import os
import time
import random
import pandas as pd
import numpy as np
import networkx as nx
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, IntVector, StrVector
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import json
import matplotlib.pyplot as plt
from pgmpy.readwrite import BIFReader

# --- Windows 环境配置 ---
os.environ['PYTHONUTF8'] = '1'
os.environ['LANGUAGE'] = 'en'

# 导入 CMCSA 算法及 R 包
from c_decomposition_1 import CMCSA111_new, CMCSA111, CMCSA

pcalg = importr('pcalg')
graph_pkg = importr('graph')


# ==========================================
# 1. 核心工具函数
# ==========================================
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
        ro.r['dimnames<-'](r_matrix, ro.r['list'](StrVector(node_names), StrVector(node_names)))
        graphNEL = ro.r['as'](r_matrix, "graphNEL")
    return graphNEL, node_names


def generate_random_lwf(n, edge_density, seed=None):
    if seed is not None: random.seed(seed)
    graph = nx.DiGraph()
    nodes = [str(i) for i in range(1, n + 1)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_density:
                graph.add_edge(nodes[i], nodes[j])
    return graph


def dag_to_dict_format(dag):
    return {node: {nbr: 'b' for nbr in dag.neighbors(node)} for node in dag.nodes()}


def dag_to_parent_dict_format(pgmpy_model):
    """
    专门为 pgmpy 模型设计的函数。
    将 pgmpy 模型的结构转换为适合 dict_to_graphNEL 的格式。
    输出格式: {'child_node': {'parent_node1': 'dummy_val', 'parent_node2': 'dummy_val', ...}}
    """
    structure_dict = {}
    for node in pgmpy_model.nodes():
        parents = list(pgmpy_model.predecessors(node))
        # 将父节点列表转换为字典格式，值可以是任意的，因为我们只关心键（节点名称）
        structure_dict[node] = {parent: 'p' for parent in parents}
    return structure_dict


def calculate_recall(eff_full, eff_local, precision=3):
    set_f = set(np.round(eff_full.flatten(), precision))
    set_l = set(np.round(eff_local.flatten(), precision))
    if not set_f: return 1.0
    intersection = set_f.intersection(set_l)
    return len(intersection) / len(set_f)


def cpdag_to_dict_format(cpdag_r, node_names):
    """将 R 的 CPDAG 转换为双向边表示的嵌套字典"""
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


def save_cpdag_image(amat, node_names, filename, title):
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    n = len(node_names)

    # 遍历邻接矩阵添加边
    for i in range(n):
        for j in range(n):
            if amat[i, j] != 0:
                if amat[j, i] != 0:
                    G.add_edge(node_names[i], node_names[j], color='red', style='dashed')  # 无向边
                else:
                    G.add_edge(node_names[i], node_names[j], color='blue', style='solid')  # 有向边

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # 固定布局方便对比

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    styles = [G[u][v]['style'] for u, v in edges]

    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color=colors,
            style=styles, node_size=600, font_weight='bold', arrows=True)

    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# ==========================================
# 2. 实验参数与全局图生成
# ==========================================
SAMPLE_SIZES = [500, 1000, 2500, 5000, 7500, 10000]
N_EXPERIMENTS = 10
N_NODES = 20
EDGE_PROB = 0.15
SEED = 257
THRESHOLD = 1e-3
# NET = 'sachs'
## 生成真实 DAG 并转换为 CPDAG
G_nx = generate_random_lwf(N_NODES, EDGE_PROB, seed=SEED)
G_dict = dag_to_dict_format(G_nx)
# reader = BIFReader("bif_file//sachs.bif")
# model = reader.get_model()
# G_dict = dag_to_parent_dict_format(model)
with localconverter(ro.default_converter + numpy2ri.converter):
    myDAG, node_names_original = dict_to_graphNEL(G_dict)
    myCPDAG = pcalg.dag2cpdag(myDAG)
    cov_true_np = np.array(pcalg.trueCov(myDAG))

# 生成供局部算法使用的 CPDAG 字典
CPDAG_dict = cpdag_to_dict_format(myCPDAG, node_names_original)

results = []

# ==========================================
# 3. 实验循环
# ==========================================
for n_samples in SAMPLE_SIZES:
    print(f"\n>>> Running Sample Size: {n_samples}")

    valid_count = 0  # 记录当前样本量下的有效实验次数
    attempt_count = 0  # 记录尝试总次数（可选，用于监控死循环）

    # while valid_count < N_EXPERIMENTS:
    #     attempt_count += 1
    #     try:
    #         # 选取不相邻节点对，针对bif文件
    #         while True:
    #             targets = random.sample(node_names_original, 2)
    #             t, y = targets[0], targets[1]
    #
    #             # 检查 t 和 y 是否在模型中有直接连接
    #             t_has_y_as_child = y in list(model.successors(t))
    #             t_has_y_as_parent = y in list(model.predecessors(t))
    #
    #             if not t_has_y_as_child and not t_has_y_as_parent:
    #                 X_nodes = [t]
    #                 Y_node = y
    #                 break

    while valid_count < N_EXPERIMENTS:
        attempt_count += 1
        try:
            # 选取不相邻节点对，适配你随机生成的 DAG（G_nx）
            while True:
                targets = random.sample(node_names_original, 2)
                t, y = targets[0], targets[1]


                # 检查 t 和 y 在 DAG 中是否有直接边（任意方向）
                has_edge = G_nx.has_edge(t, y) or G_nx.has_edge(y, t)

                if not has_edge:
                    X_nodes = [t]
                    Y_node = y
                    break
            # --- 0. 计算真实因果效应 (Ground Truth) ---
            x_idx_true = node_names_original.index(X_nodes[0]) + 1
            y_idx_true = node_names_original.index(Y_node) + 1
            with localconverter(ro.default_converter + numpy2ri.converter):
                # 必须用真实的 myDAG 和 无噪声的 cov_true_np
                res_true = pcalg.ida(x_pos=x_idx_true, y_pos=y_idx_true, mcov=cov_true_np,
                                     graphEst=myDAG, method="local")
                true_effect = float(np.mean(np.asarray(res_true)))  # myDAG下必定只有唯一值

            # 【新增判断】：如果真实因果效应为0（考虑到浮点误差，使用绝对值极小阈值），直接跳过本次，重新抽样
            if abs(true_effect) < 1e-8:
                continue

            # 模拟样本协方差，无噪声
            cov_sample = cov_true_np
            cov_sample = (cov_sample + cov_sample.T) / 2

            # 模拟样本协方差，有噪声版
            noise_n = np.random.normal(0, 1.0 / np.sqrt(n_samples), cov_true_np.shape)
            cov_sample_n = cov_true_np + noise_n
            cov_sample_n = (cov_sample_n + cov_sample_n.T) / 2

            # --- A. 全模型估计 (CPDAG) ---#无噪声
            start_f = time.time()
            with localconverter(ro.default_converter + numpy2ri.converter):
                res_f = pcalg.ida(x_pos=x_idx_true, y_pos=y_idx_true, mcov=cov_sample,
                                  graphEst=myCPDAG, method="local")
                eff_full = np.asarray(res_f)
            time_full = time.time() - start_f

            # --- A. 全模型估计 (CPDAG) ---#有噪声
            start_f_n = time.time()
            with localconverter(ro.default_converter + numpy2ri.converter):
                res_f_n = pcalg.ida(x_pos=x_idx_true, y_pos=y_idx_true, mcov=cov_sample_n,
                                    graphEst=myCPDAG, method="local")
                eff_full_n = np.asarray(res_f_n)
            time_full_n = time.time() - start_f_n

            # --- B. 子模型估计 (CMCSA) ---#无噪声
            start_l = time.time()
            H_set = CMCSA111_new(CPDAG_dict, X_nodes + [Y_node])
            H_list = sorted(list(H_set))

            idx_h = [node_names_original.index(n) for n in H_list]
            cov_h = cov_sample[np.ix_(idx_h, idx_h)]

            ro.globalenv['amat'] = ro.r['as'](myCPDAG, "matrix")
            ro.globalenv['node_names'] = StrVector(node_names_original)
            ro.r('dimnames(amat) <- list(node_names, node_names)')
            sub_amat_r = ro.r['amat'].rx(StrVector(H_list), StrVector(H_list))
            subCPDAG_H = ro.r['as'](sub_amat_r, "graphNEL")

            x_idx_l = H_list.index(X_nodes[0]) + 1
            y_idx_l = H_list.index(Y_node) + 1
            with localconverter(ro.default_converter + numpy2ri.converter):
                res_l = pcalg.ida(x_pos=x_idx_l, y_pos=y_idx_l, mcov=cov_h,
                                  graphEst=subCPDAG_H, method="local")
                eff_local = np.asarray(res_l)
            time_local = time.time() - start_l

            # --- B. 子模型估计 (CMCSA) ---#有噪声
            start_l_n = time.time()
            H_set = CMCSA111_new(CPDAG_dict, X_nodes + [Y_node])
            H_list = sorted(list(H_set))

            idx_h = [node_names_original.index(n) for n in H_list]
            cov_h_n = cov_sample_n[np.ix_(idx_h, idx_h)]  # 复用刚才的 H_list

            ro.globalenv['amat'] = ro.r['as'](myCPDAG, "matrix")
            ro.globalenv['node_names'] = StrVector(node_names_original)
            ro.r('dimnames(amat) <- list(node_names, node_names)')
            sub_amat_r = ro.r['amat'].rx(StrVector(H_list), StrVector(H_list))
            subCPDAG_H = ro.r['as'](sub_amat_r, "graphNEL")
            with localconverter(ro.default_converter + numpy2ri.converter):
                res_l_n = pcalg.ida(x_pos=x_idx_l, y_pos=y_idx_l, mcov=cov_h_n,
                                    graphEst=subCPDAG_H, method="local")
                eff_local_n = np.asarray(res_l_n)
            time_local_n = time.time() - start_l_n

            # --- 数据清理与指标计算 ---
            # 清理可能的 NaN，防止计算 MAE 报错
            eff_full = eff_full[~np.isnan(eff_full)]
            eff_local = eff_local[~np.isnan(eff_local)]
            eff_full_n = eff_full_n[~np.isnan(eff_full_n)]
            eff_local_n = eff_local_n[~np.isnan(eff_local_n)]

            if len(eff_full) == 0: eff_full = np.array([0.0])
            if len(eff_local) == 0: eff_local = np.array([0.0])
            if len(eff_full_n) == 0: eff_full_n = np.array([0.0])
            if len(eff_local_n) == 0: eff_local_n = np.array([0.0])

            # 集合大小
            set_size_f = len(np.unique(np.round(eff_full, 5)))
            set_size_l = len(np.unique(np.round(eff_local, 5)))
            set_size_f_n = len(np.unique(np.round(eff_full_n, 5)))
            set_size_l_n = len(np.unique(np.round(eff_local_n, 5)))

            # 计算平均绝对误差 (Mean-MAE)
            # 无噪声下的误差
            mean_mae_full = np.mean(np.abs(eff_full - true_effect))
            mean_mae_local = np.mean(np.abs(eff_local - true_effect))
            # 有噪声下的误差
            mean_mae_full_n = np.mean(np.abs(eff_full_n - true_effect))
            mean_mae_local_n = np.mean(np.abs(eff_local_n - true_effect))

            # --- D. 结果持久化 ---
            results.append({
                'Sample_Size': n_samples,
                'Exp_ID': valid_count,  # 这里的 ID 现在代表有效结果的索引
                'T': X_nodes[0],
                'Y': Y_node,
                'True_Effect': true_effect,

                # 无噪声情况下的指标
                'MAE_Full': mean_mae_full,
                'MAE_Local': mean_mae_local,
                'Time_Full': time_full,
                'Time_Local': time_local,
                'Eff_Full': "|".join([str(round(v, 5)) for v in np.unique(np.round(eff_full, 5))]),
                'Eff_Local': "|".join([str(round(v, 5)) for v in np.unique(np.round(eff_local, 5))]),
                'Set_Size_Full': set_size_f,
                'Set_Size_Local': set_size_l,

                # 有噪声情况下的指标
                'MAE_Full_n': mean_mae_full_n,
                'MAE_Local_n': mean_mae_local_n,
                'Time_Full_n': time_full_n,
                'Time_Local_n': time_local_n,
                'Vars_Full': len(node_names_original),
                'Vars_Local': len(H_list),
                'Eff_Full_n': "|".join([str(round(v, 5)) for v in np.unique(np.round(eff_full_n, 5))]),
                'Eff_Local_n': "|".join([str(round(v, 5)) for v in np.unique(np.round(eff_local_n, 5))]),
                'Set_Size_Full_n': set_size_f_n,
                'Set_Size_Local_n': set_size_l_n
            })

            # 只有完整走完以上流程，且顺利添加到results中，才记录为1次有效循环
            valid_count += 1
            if valid_count % 10 == 0:
                print(f"  -- Collected {valid_count}/{N_EXPERIMENTS} valid results...")

        except Exception as e:
            print(f"Error at attempt {attempt_count} (Valid so far: {valid_count}): {e}")
# ==========================================
# 4. 导出 CSV
# ==========================================
if results:
    df = pd.DataFrame(results)
    filename = f"experiment_results_{N_NODES}_{EDGE_PROB}.csv"
    # filename = f"experiment_results_{NET}.csv"
    try:
        df.to_csv(filename, index=False)
        print(f"\n[Finished] All data saved successfully to {filename}")
    except Exception as e:
        print(f"\n[Error] 无法保存 CSV 文件，文件可能被占用。详细错误: {e}")