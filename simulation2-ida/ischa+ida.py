#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal IDA: Full CPDAG vs H 子图 (R pcalg::ida 版, 无 n_samples 维度)
=====================================================================
- 统一 DAG 生成 + 统一 (X, Y) 选择 (不相邻 + CPDAG 下未识别)
- 每个 (n, d, seed) 只产出一行
- 用 population covariance (无噪声)
- 列名与 minimal_ida 版本对齐
"""

import os
os.environ['PYTHONUTF8'] = '1'
os.environ['LANGUAGE'] = 'en'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

import time
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, StrVector
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

from c_decomposition_1 import CMCSA111_new

pcalg = importr('pcalg')
graph_pkg = importr('graph')


# =============================================================================
# 统一 DAG 生成
# =============================================================================

def generate_random_dag(n, edge_density, seed, prefix="v"):
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


# =============================================================================
# MPDAG + is_identified
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


# =============================================================================
# pcalg 桥
# =============================================================================

def dict_to_graphNEL(G_dict, fixed_node_names=None):
    if fixed_node_names is None:
        all_nodes = set(G_dict.keys())
        for node, neighbors in G_dict.items():
            all_nodes.update(neighbors.keys())
        node_names = sorted(list(all_nodes))
    else:
        node_names = fixed_node_names
    n = len(node_names)
    adj_matrix = np.zeros((n, n))
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    for u, neighbors in G_dict.items():
        if u not in node_to_idx: continue
        for v in neighbors:
            if v in node_to_idx:
                adj_matrix[node_to_idx[u], node_to_idx[v]] = 1
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_matrix = ro.r['as.matrix'](adj_matrix)
        ro.r['dimnames<-'](r_matrix, ro.r['list'](StrVector(node_names),
                                                   StrVector(node_names)))
        graphNEL = ro.r['as'](r_matrix, "graphNEL")
    return graphNEL, node_names


def dag_to_dict_format(dag):
    return {node: {nbr: 'b' for nbr in dag.neighbors(node)}
            for node in dag.nodes()}


def cpdag_to_dict_format(cpdag_r, node_names):
    ro.globalenv['tmp_cpdag'] = cpdag_r
    with localconverter(ro.default_converter + numpy2ri.converter):
        amat = np.array(ro.r('as(tmp_cpdag, "matrix")'))

    n = len(node_names)
    cpdag_dict = {name: {} for name in node_names}
    for i in range(n):
        for j in range(n):
            if amat[i, j] != 0:
                cpdag_dict[node_names[i]][node_names[j]] = 'b'
    return cpdag_dict


# =============================================================================
# 单个 (n, d, seed) 实验
# =============================================================================

def run_one_seed(n, d, seed):
    """对单个 (n, d, seed) 跑实验, 无 n_samples 维度。"""
    result = {'n': n, 'edge_density': d, 'seed': seed}

    # --- 统一: 生成 DAG ---
    G_nx = generate_random_dag(n, d, seed, prefix="v")
    node_names = sorted(G_nx.nodes())

    # --- DAG -> CPDAG ---
    G_dict = dag_to_dict_format(G_nx)
    with localconverter(ro.default_converter + numpy2ri.converter):
        myDAG, node_names_r = dict_to_graphNEL(G_dict,
                                               fixed_node_names=node_names)
        myCPDAG = pcalg.dag2cpdag(myDAG)
        cov_true_np = np.array(pcalg.trueCov(myDAG))

    # --- CPDAG 邻接矩阵, 用于 (X, Y) 检查 ---
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['cpdag_tmp'] = myCPDAG
        cpdag_amat = np.array(ro.r('as(cpdag_tmp, "matrix")'))
    G_cpdag = amat_to_mpdag(cpdag_amat, node_names)

    # --- 统一: 选择 (X, Y): 不相邻 + CPDAG 下未识别 ---
    X_, Y_ = None, None
    rng_xy = np.random.default_rng(seed + 12345)
    for _ in range(5000):
        Xc, Yc = map(str, rng_xy.choice(node_names, 2, replace=False))
        if G_nx.has_edge(Xc, Yc) or G_nx.has_edge(Yc, Xc):
            continue
        if not is_identified(G_cpdag, {Xc}, {Yc}):
            X_, Y_ = Xc, Yc
            break
    if X_ is None:
        return None

    result['X'] = X_
    result['Y'] = Y_

    CPDAG_dict = cpdag_to_dict_format(myCPDAG, node_names)
    x_idx_true = node_names.index(X_) + 1
    y_idx_true = node_names.index(Y_) + 1

    # --- 真实效应 (真实 DAG + population covariance) ---
    with localconverter(ro.default_converter + numpy2ri.converter):
        res_true = pcalg.ida(**{'x.pos': x_idx_true, 'y.pos': y_idx_true,
                                'mcov': cov_true_np, 'graphEst': myDAG,
                                'method': 'local'})
        true_effect = float(np.mean(np.asarray(res_true)))
    result['true_effect'] = true_effect

    # --- 无噪声 population covariance ---
    cov_sample = (cov_true_np + cov_true_np.T) / 2

    # --- A. Full CPDAG 上的 ida ---
    start_f = time.time()
    with localconverter(ro.default_converter + numpy2ri.converter):
        res_f = pcalg.ida(**{'x.pos': x_idx_true, 'y.pos': y_idx_true,
                             'mcov': cov_sample, 'graphEst': myCPDAG,
                             'method': 'local'})
        eff_full = np.asarray(res_f)
    time_full = time.time() - start_f

    # --- B. H 子图上的 ida ---
    start_l = time.time()
    H_set = CMCSA111_new(CPDAG_dict, [X_, Y_])
    H_list = sorted(list(H_set))
    idx_h = [node_names.index(nm) for nm in H_list]
    cov_h = cov_sample[np.ix_(idx_h, idx_h)]

    ro.globalenv['amat'] = ro.r['as'](myCPDAG, "matrix")
    ro.globalenv['node_names'] = StrVector(node_names)
    ro.r('dimnames(amat) <- list(node_names, node_names)')
    sub_amat_r = ro.r['amat'].rx(StrVector(H_list), StrVector(H_list))
    subCPDAG_H = ro.r['as'](sub_amat_r, "graphNEL")

    x_idx_l = H_list.index(X_) + 1
    y_idx_l = H_list.index(Y_) + 1
    with localconverter(ro.default_converter + numpy2ri.converter):
        res_l = pcalg.ida(**{'x.pos': x_idx_l, 'y.pos': y_idx_l,
                             'mcov': cov_h, 'graphEst': subCPDAG_H,
                             'method': 'local'})
        eff_local = np.asarray(res_l)
    time_local = time.time() - start_l

    # --- 清理 NaN ---
    # --- 清理 NaN + 过滤病态爆炸值 ---
    # pcalg::ida 内部对 (X, P) 直接解 OLS，P 大且共线时 Sxx 病态，
    # 会输出 ~1e5 量级的爆炸值。这里用效应幅度上限兜底，
    # 等价于 file1 里的 cond(Sxx) > 1e10 的过滤效果。
    # === 统一过滤标准：效应绝对值上限 ===
    EFFECT_MAX = 1e4

    eff_full = eff_full[~np.isnan(eff_full)]
    eff_full = eff_full[np.abs(eff_full) < EFFECT_MAX]
    eff_local = eff_local[~np.isnan(eff_local)]
    eff_local = eff_local[np.abs(eff_local) < EFFECT_MAX]

    if len(eff_full) == 0 or len(eff_local) == 0:
        return None  # 全被过滤，丢弃该 (n, d, seed)

    # --- 与 minimal_ida 版本的列名对齐 ---
    raw_full = list(np.round(eff_full, 4))  # 原始（含重复）
    raw_ext = list(np.round(eff_local, 4))

    set_full = set(np.round(eff_full, 4))
    set_ext = set(np.round(eff_local, 4))

    only_in_full = sorted(set_full - set_ext)
    only_in_ext = sorted(set_ext - set_full)

    result['H_size'] = len(H_list)
    result['H_reduction'] = len(H_list) / n
    result['time_H'] = time_local   # ISCHA + H 子图 ida 总时间

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

    result['Time_Full'] = time_full
    result['Time_Local'] = time_local
    result['Speedup'] = (time_full / time_local
                         if time_local > 1e-9 else float('nan'))
    result['n_raw_full'] = len(raw_full)
    result['n_raw_ext'] = len(raw_ext)
    result['raw_full'] = '|'.join(map(str, sorted(raw_full)))
    result['raw_ext'] = '|'.join(map(str, sorted(raw_ext)))
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    N_LIST = (20, 40,60,80,100)
    DENSITY_LIST = (0.05, 0.10, 0.15)
    N_SEEDS = 50

    all_results = []
    t_start = time.time()

    total = len(N_LIST) * len(DENSITY_LIST) * N_SEEDS
    done = 0

    for n in N_LIST:
        for d in DENSITY_LIST:
            for k in range(N_SEEDS):
                seed = 10000 * n + 100 * int(round(d * 100)) + k
                done += 1
                print(f"[{done:4d}/{total}] n={n} d={d:.2f} seed={seed}")
                try:
                    r = run_one_seed(n, d, seed)
                    if r:
                        all_results.append(r)
                except Exception as e:
                    print(f"  error: {type(e).__name__}: {e}")

    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("verify_pcalg_ida_results.csv", index=False)
        print(f"Saved: verify_pcalg_ida_results.csv ({len(df)} rows)")
    else:
        print("No valid results.")


if __name__ == "__main__":
    main()