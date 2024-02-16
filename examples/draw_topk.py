import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

# 数据
k_add_num = [4, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 4096]
add_1024 = [0.375, 0.373, 0.383, 0.387, 0.397, 0.385, 0.370, 0.360, 0.383, None, None, None]
add_4096 = [0.387, 0.421, 0.397, 0.411, 0.425, None, 0.401, None, 0.403, None, 0.391, 0.407]

# 移除没有数据的点
k_add_num_1024 = [k for k, v in zip(k_add_num, add_1024) if v is not None]
add_1024 = [v for v in add_1024 if v is not None]

k_add_num_4096 = [k for k, v in zip(k_add_num, add_4096) if v is not None]
add_4096 = [v for v in add_4096 if v is not None]

# 创建折线图
plt.semilogx(k_add_num_1024, add_1024, label='1024', marker='o', base=2)
plt.semilogx(k_add_num_4096, add_4096, label='4096', marker='o', base=2)

# 添加最后一个数据点的水平虚线
plt.axhline(y=add_1024[-1], color='blue', linestyle='--', linewidth=1)
plt.axhline(y=add_4096[-1], color='orange', linestyle='--', linewidth=1)

# 添加标题和标签
# plt.title('k_add_num vs. Values (Logarithmic Scale)')
plt.xlabel('# Activated key-value pairs per token.')
plt.ylabel('Accuracy')
plt.legend()

import matplotlib.ticker as mtick
fmt = '%.0f' # 定义一个字符串格式，%.0f表示不显示小数，%%表示显示百分号
xticks = mtick.FormatStrFormatter(fmt) # 创建一个格式化程序
plt.gca().xaxis.set_major_formatter(xticks)

# 显示网格
plt.grid(True, which="both")

plt.savefig('topk_progress.pdf', format='pdf')
# 显示图表
plt.show()
