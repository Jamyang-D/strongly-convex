import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# 指定的样本量
SAMPLE_SIZES = [500, 1000, 2500, 5000, 7500, 10000]

# ==========================================
# 1. 数据读取与预处理
# ==========================================
files = glob.glob('MLE_KL_Divergence_*_K5.csv')
res_list = []

for f in files:
    net_name = re.search(r'Divergence_(.*)_K5\.csv', f).group(1).upper()
    df_t = pd.read_csv(f)

    if 'Sample_Size' not in df_t.columns:
        raise ValueError(f"CSV文件 {f} 中找不到 'Sample_Size' 列，请检查列名。")

    for sample_size, group in df_t.groupby('Sample_Size'):
        if sample_size in SAMPLE_SIZES:
            res_list.append({
                'Network': net_name,
                'Sample_Size': str(int(sample_size)),
                'Global_Nodes': group['Global_Nodes'].iloc[0],
                'Local_Nodes': group['Local_Nodes'].mean(),
                'Speedup': group['Speedup'].mean(),
                'KL': group['KL_Divergence'].abs().mean(),
                'Time_Global': group['Time_Global(s)'].mean(),
                'Time_Local': group['Time_Local(s)'].mean()
            })

df_plot = pd.DataFrame(res_list)

# ==========================================
# 2. 跨样本量整体平均，对应 Table 1
# ==========================================
df_table1 = df_plot.groupby('Network').agg({
    'Global_Nodes': 'mean',
    'Local_Nodes': 'mean',
    'Time_Global': 'mean',
    'Time_Local': 'mean',
    'KL': 'mean'
}).reset_index()

# 基于总平均时间重新计算 Speedup
df_table1['Speedup'] = df_table1['Time_Global'] / df_table1['Time_Local']

# 格式化列名以匹配 Table 1
df_table1 = df_table1.rename(columns={
    'Global_Nodes': 'Full',
    'Local_Nodes': 'Local',
    'Time_Global': 'Time_Full',
    'Time_Local': 'Time_Local',
    'Speedup': 'Speedup',
    'KL': 'KL-divergence'
})

# 科学计数法格式化 KL
df_table1['KL-divergence'] = df_table1['KL-divergence'].apply(lambda x: f"{x:.2e}")

# 打印结果
print(df_table1[['Network', 'Full', 'Local',
                 'Time_Full', 'Time_Local',
                 'Speedup', 'KL-divergence']].to_string(index=False))


#
# # 全局样式：无网格背景
# sns.set_theme(style="ticks")
# #
# # ==========================================
# # --- 图 2：Node Reduction (节点缩减) ---
# # ==========================================
# plt.figure(figsize=(8, 6))
# df_nodes_agg = df_plot.groupby('Network')[['Global_Nodes', 'Local_Nodes']].mean().reset_index()
#
# df_nodes = df_nodes_agg.melt(id_vars='Network', value_vars=['Global_Nodes', 'Local_Nodes'],
#                              var_name='Type', value_name='Count')
# df_nodes['Type'] = df_nodes['Type'].map({'Global_Nodes': 'Global', 'Local_Nodes': 'Local'})
#
# ax2 = sns.barplot(data=df_nodes, x='Network', y='Count', hue='Type', palette='muted')
# plt.title('', fontsize=14, fontweight='bold')
# plt.ylabel('Number of Nodes')
# plt.xlabel('Network')
# plt.xticks(rotation=0, fontsize=9)
#
# for container in ax2.containers:
#     ax2.bar_label(container, fmt='%.1f', padding=3, fontsize=9)
#
# ax2.tick_params(axis='both', direction='in', length=5)
# sns.despine()
#
# plt.legend(title=None)
# plt.tight_layout()
# plt.savefig('node.eps')
# plt.show()
#
# # ==========================================
# # --- 图 3：KL Divergence (横轴等距，精度损失) ---
# # ==========================================
# plt.figure(figsize=(8, 6))
# ax3 = sns.lineplot(data=df_plot, x='Sample_Size', y='KL', hue='Network',
#                    marker='D', markersize=8, linestyle='--', palette='tab10',
#                    hue_order=df_plot['Network'].unique())
#
# plt.axhline(0, color='black', linestyle='-', linewidth=1)
# plt.title('', fontsize=14, fontweight='bold')
# plt.ylabel('KL Divergence Value')
# plt.xlabel('Sample Size')
#
# # 设置极小的刻度范围，突出“趋于零”
# plt.ylim(-1e-15, 1e-15)
# plt.annotate('', xy=(0.5, 0.5), xycoords='axes fraction',
#              ha='center', va='center', color='blue', alpha=0.3, fontsize=15, weight='bold')
#
# ax3.tick_params(axis='both', direction='in', length=5)
# sns.despine()
#
# plt.legend(title='Network')
# plt.tight_layout()
# plt.savefig('KL.eps')
# plt.show()
#
#
# plt.figure(figsize=(8, 6))
#
# # 画箱线图看分布和中位数 (showfliers=False 会自动隐藏离群点)
# sns.boxplot(data=df_plot, x='Network', y='Speedup', palette='bone', width=0.6, showfliers=False)
#
# # 画一条 Y=1.0 的基准线，代表 "没有加速" 的基线
# plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, zorder=0)
# plt.text(len(df_plot['Network'].unique())-1.11, 1.01, 'Baseline (1.0x)', color='black', ha='right')
#
# plt.ylabel('Speedup ($S$)')
# plt.xlabel('Network')
#
# # 样式美化
# sns.despine()
# plt.xticks(rotation=0, fontsize=9)
# plt.tight_layout()
# # plt.savefig('time1.eps', dpi=800)
# plt.show()

