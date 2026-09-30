#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal IDA: Full CPDAG vs H_ext 子图 (Python IDGraphs 版)
统一 DAG 生成 + 统一 (X, Y) 选择
"""

import os
os.environ['PYTHONUTF8'] = '1'
os.environ['LANGUAGE'] = 'en'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

import time
import traceback
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, StrVector
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface_lib.embedded import RRuntimeError

from c_decomposition_1 import CMCSA111_new

pcalg = importr('pcalg')


# =============================================================================
# 统一 DAG 生成 + (X, Y) 选择
# =============================================================================

def generate_random_dag(n, edge_density, seed, prefix="v"):
    """统一随机 DAG 生成器。
    - 随机拓扑序 + 只允许前驱 -> 后继
    - RNG: numpy.default_rng(seed)
    - 节点名: prefix + 1..n
    """
    rng = np.random.default_rng(seed)
    nodes = [f"{prefix}{i + 1}" for i in range(n)]
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    order = list(rng.permutation(nodes))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_density:
                dag.add_edge(order[i], order[j])
    return dag


def select_xy_pair(dag_true, node_names, seed, max_tries=5000):
    """统一 (X, Y) 选择器。筛选条件: X, Y 在 DAG 中不直接相邻。"""
    rng = np.random.default_rng(seed + 12345)
    for _ in range(max_tries):
        X, Y = map(str, rng.choice(node_names, 2, replace=False))
        if dag_true.has_edge(X, Y) or dag_true.has_edge(Y, X):
            continue
        return X, Y
    return None, None


# =============================================================================
# MPDAG 数据结构
# =============================================================================

class MPDAG:
    def __init__(self, nodes=None, directed=None, undirected=None):
        self.nodes = set(nodes or [])
        self.directed = set(directed or [])
        self.undirected = set(undirected or [])
        for u, v in self.directed:
            self.nodes.add(u)
            self.nodes.add(v)
        for e in self.undirected:
            self.nodes.update(e)

    def copy(self):
        return MPDAG(self.nodes.copy(), self.directed.copy(),
                     self.undirected.copy())

    def adjacent(self, u, v):
        return ((u, v) in self.directed or (v, u) in self.directed
                or frozenset({u, v}) in self.undirected)

    def is_undirected(self, u, v):
        return frozenset({u, v}) in self.undirected

    def neighbors(self, u):
        res = set()
        for a, b in self.directed:
            if a == u: res.add(b)
            if b == u: res.add(a)
        for e in self.undirected:
            if u in e: res.update(e - {u})
        return res

    def children(self, u):
        return {v for a, v in self.directed if a == u}

    def parents(self, u):
        return {a for a, v in self.directed if v == u}


def mpdag_to_amat(G, node_names):
    n = len(node_names)
    idx = {name: i for i, name in enumerate(node_names)}
    amat = np.zeros((n, n), dtype=int)
    for u, v in G.directed:
        amat[idx[u], idx[v]] = 1
    for e in G.undirected:
        u, v = tuple(e)
        amat[idx[u], idx[v]] = 1
        amat[idx[v], idx[u]] = 1
    return amat


def amat_to_mpdag(amat, node_names):
    n = len(node_names)
    directed, undirected = set(), set()
    for i in range(n):
        for j in range(i + 1, n):
            if amat[i, j] == 1 and amat[j, i] == 0:
                directed.add((node_names[i], node_names[j]))
            elif amat[i, j] == 0 and amat[j, i] == 1:
                directed.add((node_names[j], node_names[i]))
            elif amat[i, j] == 1 and amat[j, i] == 1:
                undirected.add(frozenset({node_names[i], node_names[j]}))
    return MPDAG(node_names, directed, undirected)


# =============================================================================
# pcalg 桥
# =============================================================================

def apply_bg_knowledge(cpdag_amat, node_names, oriented_edges):
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['orig_amat'] = ro.conversion.py2rpy(cpdag_amat)
    ro.globalenv['tmp_names'] = StrVector(list(node_names))
    ro.r('dimnames(orig_amat) <- list(tmp_names, tmp_names)')

    if oriented_edges:
        x_list = [u for u, v in oriented_edges]
        y_list = [v for u, v in oriented_edges]
        ro.globalenv['x_vec'] = StrVector(x_list)
        ro.globalenv['y_vec'] = StrVector(y_list)
        try:
            ro.r('''
                suppressWarnings({
                    g0 <- as(orig_amat, "graphNEL")
                    g_new <- pcalg::addBgKnowledge(g0, x = x_vec, y = y_vec, verbose = FALSE)
                    new_amat <- as(g_new, "matrix")
                    dimnames(new_amat) <- list(tmp_names, tmp_names)
                })
            ''')
        except RRuntimeError:
            return None
    else:
        ro.r('new_amat <- orig_amat')

    with localconverter(ro.default_converter + numpy2ri.converter):
        new_amat = np.array(ro.r('new_amat'))
    return amat_to_mpdag(new_amat, node_names)


# =============================================================================
# IDGraphs
# =============================================================================

def _is_possibly_causal_step(G, u, v):
    return (u, v) in G.directed or frozenset({u, v}) in G.undirected


def shortest_problem_edge(G, A, Y):
    A, Y = set(A), set(Y)
    best_edge, best_len = None, float("inf")
    for a in A:
        q = deque()
        visited = set()
        for v in G.neighbors(a):
            if v in A or not _is_possibly_causal_step(G, a, v):
                continue
            first_type = 'undir' if G.is_undirected(a, v) else 'dir'
            first = (first_type, a, v)
            q.append((v, first, 1))
            visited.add((v, first_type))
        while q:
            node, first, dist = q.popleft()
            if dist >= best_len:
                continue
            if node in Y:
                if first[0] == 'undir' and dist < best_len:
                    best_len = dist
                    best_edge = (first[1], first[2])
                continue
            for nxt in G.neighbors(node):
                if nxt in A or not _is_possibly_causal_step(G, node, nxt):
                    continue
                key = (nxt, first[0])
                if key in visited:
                    continue
                visited.add(key)
                q.append((nxt, first, dist + 1))
    return best_edge


def is_identified(G, A, Y):
    return shortest_problem_edge(G, A, Y) is None


def IDGraphs(cpdag_amat, node_names, A, Y, oriented_edges=None,
             _depth=0, _max_depth=100):
    if _depth > _max_depth:
        return []
    if oriented_edges is None:
        oriented_edges = []

    G = apply_bg_knowledge(cpdag_amat, node_names, oriented_edges)
    if G is None:
        return []

    if is_identified(G, A, Y):
        return [G]

    edge = shortest_problem_edge(G, A, Y)
    if edge is None:
        return [G]

    A1, V1 = edge
    results = []
    results.extend(IDGraphs(cpdag_amat, node_names, A, Y,
                            oriented_edges + [(A1, V1)],
                            _depth + 1, _max_depth))
    results.extend(IDGraphs(cpdag_amat, node_names, A, Y,
                            oriented_edges + [(V1, A1)],
                            _depth + 1, _max_depth))
    return results


# =============================================================================
# OLS 估计
# =============================================================================

def ols_effect_from_cov(Sigma, node_names, X, Y, parents):
    """
    用 population covariance 解 Y ~ X + parents 中 X 的系数。
    parents: X 的父集（iterable）
    """
    idx = {name: i for i, name in enumerate(node_names)}
    predictors = [X] + [p for p in parents if p != X]
    predictors = list(dict.fromkeys(predictors))

    pidx = [idx[v] for v in predictors]
    yidx = idx[Y]

    Sxx = Sigma[np.ix_(pidx, pidx)]
    Sxy = Sigma[pidx, yidx]

    try:
        beta = np.linalg.solve(Sxx, Sxy)
    except np.linalg.LinAlgError:
        return None
    return float(beta[0])


# =============================================================================
# 工具函数
# =============================================================================

def dag_to_cpdag_amat(dag, node_names):
    n = len(node_names)
    idx = {name: i for i, name in enumerate(node_names)}
    amat_dag = np.zeros((n, n), dtype=int)
    for u, v in dag.edges():
        amat_dag[idx[u], idx[v]] = 1
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['tmp_amat'] = ro.conversion.py2rpy(amat_dag)
        ro.globalenv['node_names'] = StrVector(node_names)
        ro.r('dimnames(tmp_amat) <- list(node_names, node_names)')
        ro.r('g_dag <- as(tmp_amat, "graphNEL")')
        ro.r('g_cpdag <- pcalg::dag2cpdag(g_dag)')
        cpdag_amat = np.array(ro.r('as(g_cpdag, "matrix")'))
    return cpdag_amat


def compute_true_effect_r(dag, node_names, X, Y):
    """在真实 DAG 上用 pcalg::ida 算 X -> Y 的总效应。
    输入:
      dag        : nx.DiGraph, 真实 DAG
      node_names : list, 节点名 (顺序和 dag 节点一致)
      X, Y       : 处理变量和结果变量
    返回: float
    """
    n = len(node_names)
    idx = {name: i for i, name in enumerate(node_names)}
    amat_dag = np.zeros((n, n), dtype=int)
    for u, v in dag.edges():
        amat_dag[idx[u], idx[v]] = 1

    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['true_amat'] = ro.conversion.py2rpy(amat_dag)
        ro.globalenv['true_names'] = StrVector(node_names)
        ro.r('dimnames(true_amat) <- list(true_names, true_names)')
        ro.r('g_true <- as(true_amat, "graphNEL")')
        ro.r('cov_true <- pcalg::trueCov(g_true)')
        ro.globalenv['x_pos'] = idx[X] + 1
        ro.globalenv['y_pos'] = idx[Y] + 1
        ro.r('res_true <- pcalg::ida(x.pos=x_pos, y.pos=y_pos, '
             'mcov=cov_true, graphEst=g_true, method="local")')
        eff = np.array(ro.r('as.numeric(res_true)'))
    return float(np.mean(eff))


def compute_mae(effects, true_effect):
    """MAE = mean(|effect - true_effect|)。"""
    return float(np.mean(np.abs(np.array(effects) - true_effect)))


def cpdag_to_dict_format(cpdag_amat, node_names):
    n = len(node_names)
    cpdag_dict = {name: {} for name in node_names}
    for i in range(n):
        for j in range(n):
            if cpdag_amat[i, j] != 0:
                cpdag_dict[node_names[i]][node_names[j]] = 'b'
    return cpdag_dict


def generate_linear_data_from_dag(dag, node_names, n_samples, seed):
    rng = np.random.default_rng(seed)
    topo = list(nx.topological_sort(dag))
    data = {}
    for node in topo:
        parents = list(dag.predecessors(node))
        if not parents:
            data[node] = rng.normal(size=n_samples)
        else:
            coefs = rng.uniform(0.3, 0.8, size=len(parents))
            signs = rng.choice([-1, 1], size=len(parents))
            coefs = coefs * signs
            noise = rng.normal(size=n_samples)
            data[node] = sum(c * data[p] for c, p in zip(coefs, parents)) + noise
    return np.column_stack([data[n] for n in node_names])


def neighbors_of_X_in_cpdag(cpdag_amat, node_names, X):
    x_idx = node_names.index(X)
    nbrs = set()
    n = len(node_names)
    for j in range(n):
        if cpdag_amat[x_idx, j] != 0 or cpdag_amat[j, x_idx] != 0:
            nbrs.add(node_names[j])
    return nbrs


# =============================================================================
# 单次实验
# =============================================================================

def single_experiment(n, edge_density, seed, n_samples=1000):
    result = {'n': n, 'edge_density': edge_density, 'seed': seed}

    # --- 统一: 生成 DAG ---
    dag_true = generate_random_dag(n, edge_density, seed, prefix="v")
    node_names = sorted(dag_true.nodes())

    # --- CPDAG ---
    try:
        cpdag_amat = dag_to_cpdag_amat(dag_true, node_names)
    except Exception as e:
        result['error'] = f'cpdag: {e}'
        return result

    # --- 统一: 选择 (X, Y) ---
    # --- 统一: 选择 (X, Y): 不相邻 + CPDAG 下未识别 ---
    G_full = amat_to_mpdag(cpdag_amat, node_names)
    X, Y = None, None
    rng_xy = np.random.default_rng(seed + 12345)
    for _ in range(5000):
        X_, Y_ = map(str, rng_xy.choice(node_names, 2, replace=False))
        if dag_true.has_edge(X_, Y_) or dag_true.has_edge(Y_, X_):
            continue
        if not is_identified(G_full, {X_}, {Y_}):
            X, Y = X_, Y_
            break
    if X is None:
        result['error'] = 'no unidentified (X,Y) found'
        return result

    result['X'] = X
    result['Y'] = Y


    # --- 计算真实效应 (在真实 DAG 上) ---
    try:
        true_effect = compute_true_effect_r(dag_true, node_names, X, Y)
    except Exception as e:
        result['error'] = f'true_effect: {e}'
        return result
    result['true_effect'] = true_effect

    # --- CMCSA 降维 ---
    # --- CMCSA 降维 ---
    CPDAG_dict = cpdag_to_dict_format(cpdag_amat, node_names)
    try:
        t0 = time.time()
        H_set_raw = CMCSA111_new(CPDAG_dict, [X, Y])
        t_H = time.time() - t0
        H_list = sorted(list(H_set_raw))
    except Exception as e:
        result['error'] = f'cmcsa: {e}'
        return result

    result['H_size'] = len(H_list)
    result['H_reduction'] = len(H_list) / n
    result['time_H'] = t_H

    if X not in H_list or Y not in H_list:
        result['error'] = 'H missing X or Y'
        return result

    H_set = set(H_list)

    # --- H_ext = H ∪ N(X) ---
    X_nbrs = neighbors_of_X_in_cpdag(cpdag_amat, node_names, X)
    X_nbrs_outside_H = X_nbrs - H_set
    result['X_nbrs'] = '|'.join(sorted(X_nbrs))
    result['X_nbrs_outside_H'] = '|'.join(sorted(X_nbrs_outside_H))
    result['n_X_nbrs_outside_H'] = len(X_nbrs_outside_H)

    H_ext_set = H_set | X_nbrs
    H_ext_list = sorted(H_ext_set)
    result['H_ext_size'] = len(H_ext_list)
    result['H_ext_reduction'] = len(H_ext_list) / n

    idx_ext = [node_names.index(nm) for nm in H_ext_list]

    # --- H_ext 子图 IDGraphs ---
    G_ext_cpdag_amat = cpdag_amat[np.ix_(idx_ext, idx_ext)]
    t0 = time.time()
    try:
        mpdags_ext = IDGraphs(G_ext_cpdag_amat, H_ext_list, {X}, {Y})
    except Exception as e:
        result['error'] = f'IDGraphs_ext: {e}'
        return result
    result['time_ext_IDGraphs'] = time.time() - t0
    result['n_mpdags_ext'] = len(mpdags_ext)

    if not mpdags_ext:
        result['error'] = 'H_ext IDGraphs returned zero leaves'
        return result

    # --- 完整图 IDGraphs (对照) ---
    t0 = time.time()
    try:
        mpdags_full = IDGraphs(cpdag_amat, node_names, {X}, {Y})
    except Exception as e:
        result['error'] = f'IDGraphs_full: {e}'
        return result
    result['time_full_IDGraphs'] = time.time() - t0
    result['n_mpdags_full'] = len(mpdags_full)

    # --- 数据 ---
    # --- population covariance (通过 R 的 pcalg::trueCov) ---
    try:
        n_nodes = len(node_names)
        amat_dag = np.zeros((n_nodes, n_nodes), dtype=int)
        idx_all = {name: i for i, name in enumerate(node_names)}
        for u, v in dag_true.edges():
            amat_dag[idx_all[u], idx_all[v]] = 1
        with localconverter(ro.default_converter + numpy2ri.converter):
            ro.globalenv['dag_mat'] = ro.conversion.py2rpy(amat_dag)
            ro.globalenv['dag_names'] = StrVector(node_names)
            ro.r('dimnames(dag_mat) <- list(dag_names, dag_names)')
            ro.r('g_dag_tmp <- as(dag_mat, "graphNEL")')
            ro.r('cov_tmp <- pcalg::trueCov(g_dag_tmp)')
            Sigma = np.array(ro.r('cov_tmp'))
        Sigma = (Sigma + Sigma.T) / 2
    except Exception as e:
        result['error'] = f'Sigma: {e}'
        return result

    # === 统一过滤标准：效应绝对值上限 ===
    EFFECT_MAX = 1e4

    # --- H_ext 子图 OLS（用 Sigma） ---
    Sigma_ext = Sigma[np.ix_(idx_ext, idx_ext)]
    effects_ext = []
    for g in mpdags_ext:
        eff = ols_effect_from_cov(Sigma_ext, H_ext_list,
                                  X, Y, g.parents(X))
        if eff is not None and abs(eff) < EFFECT_MAX:
            effects_ext.append(eff)

    # --- 完整图 OLS（用 Sigma） ---
    effects_full = []
    for g in mpdags_full:
        eff = ols_effect_from_cov(Sigma, node_names,
                                  X, Y, g.parents(X))
        if eff is not None and abs(eff) < EFFECT_MAX:
            effects_full.append(eff)

    # 过滤后若有一边为空，本次实验无对比意义
    if not effects_ext or not effects_full:
        result['error'] = f'all effects filtered (|effect| >= {EFFECT_MAX})'
        return result
    # --- 比较 ---
    raw_full = list(np.round(effects_full, 4))  # 原始（含重复）
    raw_ext = list(np.round(effects_ext, 4))
    set_ext = set(np.round(effects_ext, 4))
    set_full = set(np.round(effects_full, 4))

    only_in_full = sorted(set_full - set_ext)
    only_in_ext = sorted(set_ext - set_full)

    result['effects_only_in_full'] = '|'.join(map(str, only_in_full))
    result['effects_only_in_ext'] = '|'.join(map(str, only_in_ext))
    result['n_only_in_full'] = len(only_in_full)
    result['n_only_in_ext'] = len(only_in_ext)

    result['effects_full'] = '|'.join(map(str, sorted(set_full)))
    result['effects_ext'] = '|'.join(map(str, sorted(set_ext)))
    result['n_distinct_full'] = len(set_full)
    result['n_distinct_ext'] = len(set_ext)
    result['same_count'] = (len(set_full) == len(set_ext))
    result['same_values'] = (len(set_full) == len(set_ext)
                             and all(abs(a - b) < 1e-3
                                     for a, b in zip(sorted(set_full),
                                                     sorted(set_ext))))
    result['is_false'] = (not result['same_values'])

    n_ext_empty_P = sum(1 for g in mpdags_ext if not g.parents(X))
    result['n_ext_empty_P'] = n_ext_empty_P
    result['false_empty_risk'] = bool(
        len(X_nbrs_outside_H) > 0 and n_ext_empty_P > 0
    )

    # --- 时间 (端到端 + 纯枚举) ---
    result['Time_Full'] = result['time_full_IDGraphs']
    result['Time_Local'] = result['time_H'] + result['time_ext_IDGraphs']
    result['Speedup'] = (
        result['Time_Full'] / result['Time_Local']
        if result['Time_Local'] > 1e-9 else float('nan')
    )
    result['Speedup_IDG'] = (
        result['time_full_IDGraphs'] / result['time_ext_IDGraphs']
        if result['time_ext_IDGraphs'] > 1e-9 else float('nan')
    )
    result['n_raw_full'] = len(raw_full)
    result['n_raw_ext'] = len(raw_ext)
    return result


# =============================================================================
# Worker / Main
# =============================================================================

def _worker_task(args):
    n, d, seed = args
    try:
        r = single_experiment(n, d, seed)
    except Exception as e:
        return {'n': n, 'edge_density': d, 'seed': seed,
                'error': f'{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}'}
    if r is None:
        return {'n': n, 'edge_density': d, 'seed': seed, 'error': 'skipped'}
    return r


def main_parallel(n_workers=None,
                  n_list=(20, 40, 60, 80, 100),
                  density_list=(0.05, 0.10, 0.15),
                  n_seeds_per_setting=20):
    tasks = []
    for n in n_list:
        for d in density_list:
            for k in range(n_seeds_per_setting):
                seed = 10000 * n + 100 * int(round(d * 100)) + k
                tasks.append((n, d, seed))

    print(f"Total tasks: {len(tasks)}")
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, len(tasks))
    print(f"Workers:     {n_workers}")

    all_results = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_worker_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {'n': t[0], 'edge_density': t[1], 'seed': t[2],
                     'error': f'future: {e}'}
            if r is not None and 'error' not in r:
                all_results.append(r)
                status = (f"X={r.get('X')} Y={r.get('Y')}  "
                          f"same={r.get('same_values','?'):>5}  "
                          f"Speedup={r.get('Speedup', 0):.2f}")
            else:
                status = f"skip ({r.get('error', '?')[:60]})"
            print(f"[{i:4d}/{len(tasks)}] n={t[0]:3d} d={t[1]:.2f} "
                  f"seed={t[2]:6d}  {status}  "
                  f"(elapsed {time.time() - t_start:.0f}s)")

    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")

    if not all_results:
        print("No valid results.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv('verify_minimal_ida_hext_results.csv', index=False)
    print(f"Saved: verify_minimal_ida_hext_results.csv ({len(df)} rows)")


if __name__ == "__main__":
    # main_parallel(
    #     n_workers=6,
    #     n_list=(20, 40,60,80,100),
    #     density_list=(0.05, 0.10, 0.15),
    #     n_seeds_per_setting=50,
    # )

    import pandas as pd
    import numpy as np

    # ============================================================
    # 1. 读取两个 CSV
    # ============================================================
    df_py = pd.read_csv('verify_minimal_ida_hext_results.csv')  # Minimal IDA
    df_r = pd.read_csv('verify_pcalg_ida_results.csv')  # IDA (pcalg)

    # 去掉 error 行
    for df in (df_py, df_r):
        if 'error' in df.columns:
            df.dropna(subset=['error'], inplace=True)


    # ============================================================
    # 2. 计算 REC / PREC（每个 seed 一行）
    # ============================================================
    def compute_rec_prec(full_str, ext_str):
        """按论文定义计算：
           REC  = |E_full ∩ E_local| / |E_full|
           PREC = |E_local ∩ E_full| / |E_local|
           效应值先 round 到 4 位小数去重。
        """
        try:
            full_vals = [float(x) for x in str(full_str).split('|')]
            ext_vals = [float(x) for x in str(ext_str).split('|')]
        except (ValueError, AttributeError):
            return np.nan, np.nan

        E_full = set(np.round(full_vals, 4))
        E_local = set(np.round(ext_vals, 4))

        if not E_full or not E_local:
            return np.nan, np.nan

        inter = E_full & E_local
        rec = len(inter) / len(E_full)
        prec = len(inter) / len(E_local)
        return rec, prec


    for df in (df_py, df_r):
        df[['REC', 'PREC']] = df.apply(
            lambda row: pd.Series(compute_rec_prec(row['effects_full'],
                                                   row['effects_ext'])),
            axis=1
        )


    # ============================================================
    # 3. 按 (n, edge_density) 求均值
    # ============================================================
    def agg(df):
        return df.groupby(['n', 'edge_density']).agg(
            Local=('Time_Local', 'mean'),
            Full=('Time_Full', 'mean'),
            Speedup=('Speedup', 'mean'),
            REC=('REC', 'mean'),
            PREC=('PREC', 'mean'),
        ).reset_index()


    agg_py = agg(df_py).rename(columns={
        'Local': 'Local_py', 'Full': 'Full_py', 'Speedup': 'Speedup_py',
        'REC': 'REC_py', 'PREC': 'PREC_py'})
    agg_r = agg(df_r).rename(columns={
        'Local': 'Local_r', 'Full': 'Full_r', 'Speedup': 'Speedup_r',
        'REC': 'REC_r', 'PREC': 'PREC_r'})

    merged = agg_py.merge(agg_r, on=['n', 'edge_density'])
    merged = merged.sort_values(['n', 'edge_density']).reset_index(drop=True)

    # ============================================================
    # 4. 输出 LaTeX
    # ============================================================
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Minimal IDA vs IDA: local and full time (s), speedup, "
          r"and Recall / Precision, averaged over seeds.}")
    print(r"\label{tab:time-comparison}")
    print(r"\begin{tabular}{lccccccccccc}")
    print(r"\toprule")
    print(r" & & \multicolumn{5}{c}{Minimal IDA} & \multicolumn{5}{c}{IDA} \\")
    print(r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}")
    print(r" & & \multicolumn{3}{c}{Time (s)} & \multicolumn{2}{c}{Accuracy} "
          r"& \multicolumn{3}{c}{Time (s)} & \multicolumn{2}{c}{Accuracy} \\")
    print(r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}")
    print(r"Nodes & Density & Local & Full & Speedup & REC & PREC "
          r"& Local & Full & Speedup & REC & PREC \\")
    print(r"\midrule")

    prev_n = None
    for _, row in merged.iterrows():
        if prev_n is not None and prev_n != row['n']:
            print(r"\addlinespace")
        prev_n = row['n']
        print(
            f"{int(row['n'])} & {row['edge_density']:.2f} & "
            f"{row['Local_py']:.4f} & {row['Full_py']:.4f} & {row['Speedup_py']:.2f} & "
            f"{row['REC_py']:.3f} & {row['PREC_py']:.3f} & "
            f"{row['Local_r']:.4f} & {row['Full_r']:.4f} & {row['Speedup_r']:.2f} & "
            f"{row['REC_r']:.3f} & {row['PREC_r']:.3f} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    #
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator
    import os

    # ==========================================
    # 1. 读取数据并展平
    # ==========================================
    # 注意：这里用 Minimal IDA 的 CSV
    file_path = 'verify_minimal_ida_hext_results.csv'
    # 如果你要画 IDA 的对应图，把上面换成 'verify_pcalg_ida_results.csv' 即可

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    df_raw = pd.read_csv(file_path)

    flat_data = []
    for _, row in df_raw.iterrows():
        if 'error' in row and pd.notna(row['error']):
            continue

        full_str = str(row['effects_full'])
        local_str = str(row['effects_ext'])

        if full_str == 'nan' or local_str == 'nan':
            continue

        # 按 '|' 切割，展平为散点坐标
        full_vals = full_str.split('|')
        local_vals = local_str.split('|')

        # 注意：有些 seed 可能 full 和 local 的效应个数不一样，这里按最小长度配对
        n_pairs = min(len(full_vals), len(local_vals))

        for i in range(n_pairs):
            try:
                flat_data.append({
                    'Nodes': int(row['n']),
                    'Density': float(row['edge_density']),
                    'Clean_Full': float(full_vals[i]),
                    'Clean_Local': float(local_vals[i])
                })
            except ValueError:
                continue

    if not flat_data:
        raise ValueError("未提取到有效数据，请检查 CSV 内容。")

    df_clean = pd.DataFrame(flat_data)
    df_clean = df_clean.sort_values(by=['Density', 'Nodes'])

    # ==========================================
    # 2. 画图矩阵
    # ==========================================
    sns.set_theme(style="ticks", rc={"axes.facecolor": "white", "figure.facecolor": "white"})

    g = sns.relplot(
        data=df_clean,
        x="Clean_Full",
        y="Clean_Local",
        row="Density",
        col="Nodes",
        kind="scatter",
        color='black',
        s=12,
        alpha=0.4,
        linewidth=0,
        height=2.2,
        aspect=1,
        facet_kws={'margin_titles': True, 'sharex': False, 'sharey': False}
    )

    # 坐标轴标签
    g.set_axis_labels("(Minimal) IDA", "ISCHA + (Minimal) IDA", size=10)

    # 行列标题：用数学符号
    g.set_titles(col_template="$|V|$ = {col_name}", row_template="$d$ = {row_name}")

    # ==========================================
    # 3. 强行同步刻度，让每张图都有一致的范围
    # ==========================================
    for ax in g.axes.flat:
        # 显示黑色边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

        if ax.has_data():
            # 获取当前图中散点的真实极值
            paths = ax.collections[0].get_offsets()
            if len(paths) > 0:
                x_data, y_data = paths[:, 0], paths[:, 1]
                vmin = min(x_data.min(), y_data.min())
                vmax = max(x_data.max(), y_data.max())

                # 留 5% 边距
                margin = (vmax - vmin) * 0.05
                if margin == 0: margin = 0.05
                vmin -= margin
                vmax += margin

                ax.set_xlim(vmin, vmax)
                ax.set_ylim(vmin, vmax)
                ax.set_box_aspect(1)  # 正方形

                # 同步刻度
                locator = MaxNLocator(nbins=4, prune=None)
                ax.xaxis.set_major_locator(locator)
                ax.yaxis.set_major_locator(locator)

                # 开启所有数字显示
                ax.tick_params(labelbottom=True, labelleft=True, labelsize=8)

    # ==========================================
    # 4. 保存
    # ==========================================
    plt.tight_layout(h_pad=1.5, w_pad=2.5)
    plt.savefig('minimalida.eps', dpi=300, bbox_inches='tight')
    plt.show()


    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    import os


    # ==========================================
    # 1. 抽取数据的通用函数
    # ==========================================
    def load_data(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        df_raw = pd.read_csv(file_path)
        data_dict = {}

        for _, row in df_raw.iterrows():
            if 'error' in row and pd.notna(row['error']):
                continue

            full_str, local_str = str(row['effects_full']), str(row['effects_ext'])
            if full_str == 'nan' or local_str == 'nan':
                continue

            try:
                full_vals = [float(x) for x in full_str.split('|')]
                local_vals = [float(x) for x in local_str.split('|')]
            except ValueError:
                continue

            n_pairs = min(len(full_vals), len(local_vals))
            key = (int(row['n']), float(row['edge_density']))

            if key not in data_dict:
                data_dict[key] = {'full': [], 'local': []}

            data_dict[key]['full'].extend(full_vals[:n_pairs])
            data_dict[key]['local'].extend(local_vals[:n_pairs])

        return data_dict


    # 读取两份数据
    file_min_ida = 'verify_minimal_ida_hext_results.csv'
    file_ida = 'verify_pcalg_ida_results.csv'

    data_min = load_data(file_min_ida)
    data_ida = load_data(file_ida)

    # 获取所有存在的参数组合以确定行列
    all_keys = set(data_min.keys()).union(set(data_ida.keys()))
    nodes_list = sorted(list(set(k[0] for k in all_keys)))  # 列: |V|
    density_list = sorted(list(set(k[1] for k in all_keys)))  # 行: d

    # ==========================================
    # 2. 绘制嵌套复合图矩阵 (GridSpec)
    # ==========================================
    # 设置全局字体和大小
    plt.rcParams.update({'font.size': 9, 'axes.facecolor': 'white', 'figure.facecolor': 'white'})

    # 创建大画布，调整大小以适应 3x10 个小图
    fig = plt.figure(figsize=(14, 7.5))

    # 外层网格：3行 (Density) x 5列 (Nodes)
    outer_grid = gridspec.GridSpec(len(density_list), len(nodes_list), wspace=0.35, hspace=0.35)


    def format_axis(ax, x_data, y_data):
        """统一调整坐标轴比例与刻度"""
        if len(x_data) == 0: return
        vmin = min(np.min(x_data), np.min(y_data))
        vmax = max(np.max(x_data), np.max(y_data))
        margin = (vmax - vmin) * 0.05 if vmax != vmin else 0.05

        ax.set_xlim(vmin - margin, vmax + margin)
        ax.set_ylim(vmin - margin, vmax + margin)
        ax.set_box_aspect(1)  # 强制正方形

        # 控制刻度数量避免重叠
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)


    # 遍历填充图表
    for r, d in enumerate(density_list):
        for c, n in enumerate(nodes_list):
            # 内层网格：在每个 (d, n) 单元格内再切分为 1行 x 2列 (左侧 Min IDA, 右侧 IDA)
            inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[r, c], wspace=0.15)

            ax_left = fig.add_subplot(inner_grid[0])
            ax_right = fig.add_subplot(inner_grid[1])

            key = (n, d)

            # --- 绘制 Minimal IDA (左图，黑色) ---
            if key in data_min and data_min[key]['full']:
                ax_left.scatter(data_min[key]['full'], data_min[key]['local'], color='black', s=6, alpha=0.5,
                                linewidth=0)
                format_axis(ax_left, data_min[key]['full'], data_min[key]['local'])

            # --- 绘制 IDA (右图，灰色) ---
            if key in data_ida and data_ida[key]['full']:
                ax_right.scatter(data_ida[key]['full'], data_ida[key]['local'], color='gray', s=6, alpha=0.5,
                                 linewidth=0)
                format_axis(ax_right, data_ida[key]['full'], data_ida[key]['local'])

            # --- 坐标轴 Y 轴位置调整 ---
            # 为避免左右图 Y 轴标签打架，右侧图的 Y 轴刻度放到右边
            ax_right.yaxis.tick_right()
            ax_right.yaxis.set_label_position("right")

            # --- 添加标签与标题 ---
            # 1. 顶部标题 (|V|)
            if r == 0:
                # 利用坐标点，将标题居中放置在两个小图中间
                ax_left.set_title(f'$|V| = {n}$', x=1.075, ha='center', pad=12, fontsize=11)

            # 2. 右侧标题 (Density)
            if c == len(nodes_list) - 1:
                ax_right.set_ylabel(f'$d = {d}$', rotation=270, labelpad=20, fontsize=11)

    # ==========================================
    # 3. 添加全局说明 & 保存
    # ==========================================
    fig.supxlabel("Full Algorithm Effect", y=0.02, fontsize=12)
    fig.supylabel("Local (ISCHA) Effect", x=0.01, fontsize=12)

    # 自定义全局图例
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Minimal IDA'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Standard IDA')
    ]
    fig.legend(handles=custom_lines, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=10)

    plt.subplots_adjust(top=0.9, bottom=0.08, left=0.06, right=0.94)

    # 保存和显示
    # plt.savefig('combined_ida_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.show()

