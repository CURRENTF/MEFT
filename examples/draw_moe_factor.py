import matplotlib.pyplot as plt

# Your data
# 3600
factor = [0.1, 0.25, 0.5, 1, 2, 4, 10]
factor = [int(60/f) for f in factor]
acc = [0.313, 0.320, 0.261, 0.215, 0.287, 0.261, 0.287]

# 6400
factor_1 = [0.1, 0.25, 0.5, 1, 2, 4, 10]
factor_1 = [int(80/f) for f in factor_1]
acc_1 = [0.325, 0.327, 0.321, 0.311, 0.319, 0.315, 0.317]

# Create the plot
plt.plot(factor, acc, marker='o', label='3600')
plt.plot(factor_1, acc_1, marker='^', label='6400')

# Add title and labels
# plt.title('Line Plot of Accuracy by Factor')
plt.xlabel('Number of Key-Experts', fontsize='large')
plt.ylabel('Accuracy', fontsize='large')
plt.grid(True)
plt.xscale('log')

# import matplotlib.ticker as mtick
# fmt = '%.2f' # 定义一个字符串格式，%.0f表示不显示小数，%%表示显示百分号
# xticks = mtick.FormatStrFormatter(fmt) # 创建一个格式化程序
# plt.gca().xaxis.set_major_formatter(xticks)
# plt.gca().xaxis.set_major_locator(mtick.FixedLocator(factor))

plt.savefig("factor_progress.pdf", format='pdf')

# Show the plot
plt.show()

# x, 1-x
# x, 2-2x
# x/(2-x)=