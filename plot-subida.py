import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset  # 【新增】局部放大所需库
import os
from matplotlib.ticker import ScalarFormatter
from rpy2.robjects.lib.grid import lines

# ==========================================
# 1. 全局配置与工具函数
# ==========================================
sns.set_theme(style="white", context="paper", font_scale=1.2)

# 设置要分析的网络列表
networks = ['sachs','child','alarm', 'insurance',  'hepar2', 'pathfinder', 'munin1']
colors = sns.color_palette("Set2_r", len(networks))
pal = {net: colors[i] for i, net in enumerate(networks)}
marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']


def beautify_ax(ax):
    """学术图表美化"""
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='in', bottom=True, left=True, width=1.2, length=5)

#
# def compute_metrics_noisy_aligned(row):
#     """
#     仅用于图 (b)：基于 True_Effect 进行零效应硬对齐。
#     如果真实效应为 0，则忽略噪声引起的微小偏差，强制返回 1.0。
#     """
#     try:
#         # 1. 检查 True_Effect 是否为 0 (硬对齐判断)
#         true_eff = abs(row.get('True_Effect', 1.0))
#         if true_eff < 1e-9:
#             return pd.Series([1.0, 1.0])
#
#         if pd.isna(row['Eff_Full_n']) or pd.isna(row['Eff_Local_n']):
#             return pd.Series([1.0, 1.0])
#
#         # 2. 正常提取集合并对比
#         f_vals = [round(float(v), 5) for v in str(row['Eff_Full_n']).split('|')]
#         l_vals = [round(float(v), 5) for v in str(row['Eff_Local_n']).split('|')]
#         f_set, l_set = set(f_vals), set(l_vals)
#
#         intersect = len(f_set.intersection(l_set))
#         rec = intersect / len(f_set) if len(f_set) > 0 else 1.0
#         prec = intersect / len(l_set) if len(l_set) > 0 else 1.0
#         return pd.Series([rec, prec])
#     except:
#         return pd.Series([1.0, 1.0])

def compute_metrics(row):
    """直接根据 CSV 字符串提取集合并计算匹配率"""
    try:
        # 提取集合并舍入防止浮点数精度跳变
        f_vals = [round(float(v), 5) for v in str(row['Eff_Full_n']).split('|')]
        l_vals = [round(float(v), 5) for v in str(row['Eff_Local_n']).split('|')]
        f_set, l_set = set(f_vals), set(l_vals)
        intersect = len(f_set.intersection(l_set))
        rec = intersect / len(f_set) if len(f_set) > 0 else 1.0
        prec = intersect / len(l_set) if len(l_set) > 0 else 1.0
        return pd.Series([rec, prec])
    except:
        return pd.Series([1.0, 1.0])

# ==========================================
# 2. 自动化数据读取
# ==========================================
df_list = []
for net in networks:
    file_path = f'experiment_results_{net}.csv'
    if os.path.exists(file_path):
        df_temp = pd.read_csv(file_path)
        df_temp['Network'] = net
        df_temp['Sample_Size'] = pd.to_numeric(df_temp['Sample_Size'], errors='coerce')

        # 计算对齐后的指标 (用于图 b)
        df_temp[['Recall', 'Precision']] = df_temp.apply(compute_metrics, axis=1)
        # 计算加速比 (用于图 c)
        df_temp['Speedup'] = df_temp['Time_Full_n'] / df_temp['Time_Local_n']

        df_list.append(df_temp)

if not df_list:
    raise ValueError("未找到 CSV 数据文件。")

df_all = pd.concat(df_list, ignore_index=True)
valid_networks = df_all['Network'].unique()

# ==========================================
# 3. 独立绘图
# ==========================================
#
# # --- 图 (a): Exactness Scatter Plot (保持原始噪声数据) + 局部放大 ---
# fig1, ax1 = plt.subplots(figsize=(7, 6))
# axis_max = 10
# axis_min = 0
# # 【修改点1】将放大图从 40% 缩小到 35%，并将 borderpad 从 2 增加到 4，使其远离主图坐标轴
# axins = inset_axes(ax1, width="35%", height="35%", loc='upper left', borderpad=4)
#
# for net in valid_networks:
#     sub_df = df_all[df_all['Network'] == net]
#     for _, row in sub_df.iterrows():
#         if pd.isna(row['Eff_Full_n']): continue
#
#         f_v = [float(v) for v in str(row['Eff_Full_n']).split('|')]
#         l_v = [float(v) for v in str(row['Eff_Local_n']).split('|')]
#
#         marker = '^' if len(f_v) > 1 else 'o'
#         for lv in l_v:
#             x_val = min(f_v, key=lambda x: abs(x - lv))
#             y_val = lv
#
#             # 画在主图
#             ax1.scatter(x_val, y_val, alpha=0.6, edgecolor='white', s=80, marker=marker,
#                         color=pal[net], linewidth=0.6)
#             # 同步画在放大图
#             axins.scatter(x_val, y_val, alpha=0.6, edgecolor='white', s=40, marker=marker,
#                           color=pal[net], linewidth=0.6)
#
# ax1.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', alpha=0.9, linewidth=2.0, zorder=0)
# axins.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', alpha=0.9, linewidth=1.5, zorder=0)
#
# ax1.set(xlim=[axis_min, axis_max], ylim=[axis_min, axis_max],
#         xlabel='Local IDA Effect (Noisy)', ylabel='Subgraph IDA Effect (Noisy)')
#
# axins.set_xlim(0.5, 1)
# axins.set_ylim(0.5, 1)
# axins.tick_params(axis='both', which='major', labelsize=6.5)
#
# mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec="0.5", alpha=0.6, lw=1.0, linestyle='--')
# # loc1=2 (左上角), loc2=3 (左下角)
#
# # 3. 图例整理 (按列顺序强制排列)
# col1 = [Line2D([0], [0], color='r', ls='--', lw=2, label='y = x')] # 左上
# col2 = [Line2D([], [], color='none', label='')]                    # 右上 (透明占位)
#
# # 将网络均匀分给左右两列
# for i, net in enumerate(valid_networks):
#     item = Line2D([], [], color=pal[net], marker='s', ls='', ms=8, label=net.capitalize())
#     if i % 2 == 0:
#         col1.append(item)
#     else:
#         col2.append(item)
#
# # 形状标志分列底部
# col1.append(Line2D([], [], color='none', markeredgecolor='black', markerfacecolor='white', marker='o', ls='', ms=8, label='Unique effect'))
# col2.append(Line2D([], [], color='none', markeredgecolor='black', markerfacecolor='white', marker='^', ls='', ms=8, label='Multiset effect'))
#
# # 合并后 ncol=2 就会完美展示
# ax1.legend(handles=col1 + col2, loc='lower right', fontsize=9, frameon=True, ncol=2)
# beautify_ax(ax1)
# plt.savefig('R_1.eps', bbox_inches='tight')

#
# import numpy as np
#
# # --- 图 (b): Matching Rate (Recall & Precision) ---
# fig2, (ax2_rec, ax2_pre) = plt.subplots(1, 2, figsize=(11, 5.5))
# global_min_recall = 1.0
#
# # 【新增】获取全局唯一的 Sample Size 并生成等间距坐标
# x_labels = sorted(df_all['Sample_Size'].unique())
# x_positions = np.arange(len(x_labels))
#
# for i, net in enumerate(valid_networks):
#     sub_df = df_all[df_all['Network'] == net]
#     sum_b = sub_df.groupby('Sample_Size')[['Recall', 'Precision']].mean().sort_index().reset_index()
#
#     global_min_recall = min(global_min_recall, sum_b['Recall'].min())
#
#     # 【修改点】将 sum_b['Sample_Size'] 替换为 x_positions
#     ax2_rec.plot(x_positions, sum_b['Recall'], color=pal[net], ls='-', lw=2.5,
#                  marker=marker_list[i % len(marker_list)], ms=8, alpha=0.7, label=net.capitalize())
#     ax2_pre.plot(x_positions, sum_b['Precision'], color=pal[net], ls='-', lw=2.5,
#                  marker=marker_list[i % len(marker_list)], ms=8, alpha=0.7, label=net.capitalize())
#
# # 画最低参考线
# ax2_rec.axhline(global_min_recall, color='gray', ls='--', lw=1.5, zorder=0)
#
# # ================= 修改点：Y轴范围与 X轴刻度 =================
# ymin, ymax = 0.0, 1.05
# min_val = round(global_min_recall, 2)
#
# ax2_rec.set_ylim([ymin, ymax])
# ax2_rec.set_yticks(sorted(list(set([ymin, 0.9, 1.0, min_val]))))
# # 【新增】应用等间距刻度并显示真实标签
# ax2_rec.set_xticks(x_positions)
# ax2_rec.set_xticklabels([int(x) for x in x_labels])
# ax2_rec.set(ylabel='Recall', xlabel='Sample Size ($N$)')
# beautify_ax(ax2_rec)
#
# ax2_pre.set_ylim([ymin, ymax])
# ax2_pre.set_yticks([ymin, 1.0])
# # 【新增】应用等间距刻度并显示真实标签
# ax2_pre.set_xticks(x_positions)
# ax2_pre.set_xticklabels([int(x) for x in x_labels])
# ax2_pre.set(ylabel='Precision', xlabel='Sample Size ($N$)')
# beautify_ax(ax2_pre)
#
# # ================= 图例与布局 =================
# handles, labels = ax2_rec.get_legend_handles_labels()
#
# # 1. 自动对齐子图
# plt.tight_layout()
#
# # 2. 强制把子图底部抬高到 15% 的位置，腾出绝对空间
# plt.subplots_adjust(bottom=0.15)
#
# # 3. 将图例放在腾出的底部空间
# fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.07),
#             ncol=6, frameon=True, fontsize=10)
#
# plt.savefig('R_2.eps',  bbox_inches='tight')
# plt.show()
#
# # --- 图 (c): Speedup Scaling ---
# fig3, ax3 = plt.subplots(figsize=(7, 5))
#
# # 【新增】获取全局唯一的 Sample Size 并生成等间距坐标
# x_labels = sorted(df_all['Sample_Size'].unique())
# x_positions = np.arange(len(x_labels))
#
# for i, net in enumerate(valid_networks):
#     sub_df = df_all[df_all['Network'] == net]
#     sum_c = sub_df.groupby('Sample_Size')['Speedup'].mean().sort_index().reset_index()
#
#     # 【修改点】将 sum_c['Sample_Size'] 替换为 x_positions
#     ax3.plot(x_positions, sum_c['Speedup'],
#              color=pal[net], marker=marker_list[i % len(marker_list)],
#              lw=2.5, ms=8, mfc='white', mew=1.5, label=net.capitalize())
#
# ax3.axhline(1.0, color='gray', ls='--', lw=1.5, zorder=0)
#
# ax3.set_yscale('log')
# ax3.yaxis.set_major_formatter(ScalarFormatter())
# ax3.set_yticks([1, 2, 5, 10, 20, 50])
#
# # 【新增】应用等间距刻度并显示真实标签
# ax3.set_xticks(x_positions)
# ax3.set_xticklabels([int(x) for x in x_labels])
#
# ax3.set(ylabel='Speedup ($S$)', xlabel='Sample Size ($N$)')
# beautify_ax(ax3)
#
# # 1. 先自动收紧布局
# plt.tight_layout()
#
# # 2. 强制抬高整个画布的底部，腾出 20% 的空间
# plt.subplots_adjust(bottom=0.2)
#
# # 3. 将图例的 y 坐标往下移到 -0.12，彻底避开横坐标标签
# ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True, fontsize=10)
#
# plt.savefig('R_3.eps',bbox_inches='tight')
# plt.show()
#
# import os
# import pandas as pd
#
#
# # ==========================================
# # 计算 Recall & Precision
# # ==========================================
# def compute_metrics(row):
#     try:
#         f_vals = [round(float(v), 5) for v in str(row['Eff_Full_n']).split('|')]
#         l_vals = [round(float(v), 5) for v in str(row['Eff_Local_n']).split('|')]
#         f_set, l_set = set(f_vals), set(l_vals)
#         intersect = len(f_set.intersection(l_set))
#         rec = intersect / len(f_set) if len(f_set) > 0 else 1.0
#         prec = intersect / len(l_set) if len(l_set) > 0 else 1.0
#         return pd.Series([rec, prec])
#     except:
#         return pd.Series([1.0, 1.0])
#
#
# # ==========================================
# # 网络节点数（用于排序）
# # ==========================================
# network_nodes = {
#     "sachs": 11,
#     "insurance": 27,
#     "alarm": 37,
#     "win95pts": 44,
#     "hepar2": 70,
#     "pathfinder": 109,
#     "munin1": 186
# }
#
# # ==========================================
# # 读取所有数据
# # ==========================================
# df_list = []
#
# for file in os.listdir('.'):
#     if file.startswith('experiment_results_') and file.endswith('.csv'):
#         net = file.replace('experiment_results_', '').replace('.csv', '')
#         df = pd.read_csv(file)
#         df['Network'] = net
#         df[['Recall', 'Precision']] = df.apply(compute_metrics, axis=1)
#
#         # 正确加速比
#         df['Speedup'] = df['Time_Full_n'] / df['Time_Local_n']
#
#         df_list.append(df)
#
# df_all = pd.concat(df_list, ignore_index=True)
#
# # ==========================================
# # ✅ 按【网络】求总平均
# # ==========================================
# avg = df_all.groupby('Network')[['Recall', 'Precision', 'Time_Local_n', 'Time_Full_n', 'Speedup']].mean()
#
# # 加入节点数并按节点排序
# avg['Nodes'] = avg.index.map(network_nodes)
# avg = avg.sort_values('Nodes')
#
# # 美化输出
# avg = avg.round(4)
# avg['Speedup'] = avg['Speedup'].round(2)
#
# # ==========================================
# # 输出最终平均结果
# # ==========================================
# print("各网络平均结果（正确加速比）")
# print("=" * 85)
# print(avg[['Recall', 'Precision', 'Time_Local_n', 'Time_Full_n', 'Speedup']])
# print("=" * 85)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from matplotlib.ticker import MaxNLocator

# ==========================================
# 1. 数据解析与展平 (提取网络名称)
# ==========================================
file_pattern = 'experiment_results_*.csv'
all_files = glob.glob(file_pattern)

flat_data = []
for f in all_files:
    # 提取文件路径中的纯文件名部分（防止路径干扰）
    filename = f.split('\\')[-1].split('/')[-1]

    # 【核心修改】：匹配 experiment_results_ 后面的网络名称
    match = re.search(r'experiment_results_(.+)\.csv', filename)
    if match:
        network_name = match.group(1).capitalize()  # 首字母大写，例如 "Sachs"

        try:
            temp_df = pd.read_csv(f)
            # 注意：这里使用的是上一版代码里的 Eff_Full 和 Eff_Local
            # 如果你的新表格里列名变了，请把这俩改成对应的名字（比如 Eff_Full_n）
            for _, row in temp_df.iterrows():
                full_vals = str(row['Eff_Full_n']).split('|')
                local_vals = str(row['Eff_Local_n']).split('|')
                for val_f, val_l in zip(full_vals, local_vals):
                    try:
                        flat_data.append({
                            'Network': network_name,
                            'Clean_Full': float(val_f),
                            'Clean_Local': float(val_l)
                        })
                    except ValueError:
                        continue
        except Exception as e:
            print(f"读取 {filename} 出错: {e}")

if not flat_data:
    raise ValueError("未找到CSV文件或解析失败，请检查路径和文件名。")

df_clean = pd.DataFrame(flat_data)
# 按网络名称按字母顺序排序
df_clean = df_clean.sort_values(by=['Network'])

# ==========================================
# 2. 绘制散点图矩阵 (自动换行网格)
# ==========================================
sns.set_theme(style="ticks", rc={"axes.facecolor": "white", "figure.facecolor": "white"})

g = sns.relplot(
    data=df_clean,
    x="Clean_Full",
    y="Clean_Local",
    col="Network",  # 以网络名称作为分类维度
    col_wrap=3,  # 【核心修改】：每行最多 5 个图，多的自动放到下一行
    kind="scatter",
    color='black',
    s=12,
    alpha=0.25,
    linewidth=0,
    height=2.2,
    aspect=1,
    facet_kws={'sharex': False, 'sharey': False}
)
# 【修改 1】：通过 size 参数调整 X轴 和 Y轴 的标签字体大小
g.set_axis_labels("IDA", "SIDA", size=10)

# 【修改 2】：通过 size 参数调整每个子图标题（网络名称）的大小
g.set_titles(col_template="{col_name}", size=10)
# g.set_axis_labels("IDA", "SIDA")
# # 标题直接显示网络名称
# g.set_titles(col_template="{col_name}")

# ==========================================
# 3. 终极暴力修复：强行同步刻度，严禁隐藏！
# ==========================================
for ax in g.axes.flat:
    # 强制显示四个黑色边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)

    if ax.has_data():
        paths = ax.collections[0].get_offsets()
        if len(paths) > 0:
            x_data, y_data = paths[:, 0], paths[:, 1]
            vmin = min(x_data.min(), y_data.min())
            vmax = max(x_data.max(), y_data.max())

            margin = (vmax - vmin) * 0.05
            if margin == 0: margin = 0.05
            vmin -= margin
            vmax += margin

            # 强行设置相同的物理边界框
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_box_aspect(1)

            # 严禁隐藏刻度
            locator = MaxNLocator(nbins=4, prune=None)
            ax.xaxis.set_major_locator(locator)
            ax.yaxis.set_major_locator(locator)

            # 强行显示所有坐标轴标签，字号调为 8 防止碰撞
            ax.tick_params(labelbottom=True, labelleft=True, labelsize=8)

# ==========================================
# 4. 紧凑排版并保存
# ==========================================
# 留出足够的文字呼吸空间
plt.tight_layout(h_pad=1.5, w_pad=2.5)
plt.savefig('scatter1.eps', dpi=800, bbox_inches='tight')
plt.show()