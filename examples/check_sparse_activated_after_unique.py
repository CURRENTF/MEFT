import matplotlib.pyplot as plt
import pickle


def mean_of_window(lst, k):
    # 创建一个新的列表用于存放结果
    result = []
    # 获取列表的长度
    n = len(lst)

    # 遍历列表
    for i in range(k, n, k // 2):
        # 计算窗口的起始和结束索引
        start_index = max(0, i - k)
        end_index = min(n, i + k + 1)  # 加1因为range不包括结束索引

        # 计算窗口内元素的均值
        window_sum = sum(lst[start_index:end_index])
        window_count = end_index - start_index
        window_mean = window_sum / window_count

        # 将均值添加到结果列表
        result.append(window_mean)

    return result


d = {}
for i in range(32):
    d[i] = pickle.load(open(f"llama-nq_v1-kv-topk16-add6400-MOE-lookout_acts-ep1-topk16-{i}.pk", "rb"))
    d[i] = mean_of_window(d[i], 600)

# 假设所有列表共享相同的X轴数据
x = range(len(d[0]))
lis = list(range(0, 32, 4))
lis += [31]
for i in lis:
    plt.plot(x, d[i], label=i)
# 绘制折线图

# 添加图例
plt.legend()

# 添加标题和轴标签
# plt.title('Line Graph of 10 Lists')
plt.xlabel('Training Steps')
plt.ylabel('Total Activated Neurons')

# 显示图表
plt.show()

a = 0
for i in range(0, 32, 1):
    a+=d[i][-1]
print(a/32)
print(a/32/6400)