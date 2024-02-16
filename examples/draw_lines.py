import matplotlib.pyplot as plt

# Updated Parallel Adapter data
parallel_adapter_rank = [16, 128, 512, 1024, 1536, 2048, 3072, 4096, 6144]
parallel_adapter_Accuracy = [0.285, 0.319, 0.379, 0.375, 0.387, 0.399, 0.425, 0.407, 0.401]
parallel_adapter_params = [p / 7e9 * 100 for p in [4_194_304, 33_554_432, 134_217_728, 268_435_456, 402_653_184,
                                                   536_870_912, 805_306_368, 1_073_741_824, 1_610_612_736]]

# Updated LoRA data
lora_rank = [4, 8, 32, 64, 128, 256, 384, 512, 784, 1024, 2048]
lora_Accuracy = [0.287, 0.289, 0.301, 0.301, 0.287, 0.305, 0.299, 0.281, 0.293, 0.287, 0.291]
lora_params = [p / 7e9 * 100 for p in [2_097_152, 4_194_304, 16_777_216, 33_554_432, 67_108_864, 134_217_728,
                                        201_326_592, 268_435_456, 411_041_792, 536_870_912, 1_073_741_824]]

# Updated AdapterH data
adapterH_rank = [4, 32, 128, 256, 384, 512, 1024, 1536]
adapterH_Accuracy = [0.267, 0.297, 0.351, 0.377, 0.371, 0.377, 0.331, 0.317]
adapterH_params = [p / 7e9 * 100 for p in [3_145_728, 25_165_824, 100_663_296, 201_326_592, 301_989_888,
                                            402_653_184, 805_306_368, 1_207_959_552]]

# Plotting Parallel Adapter, LoRA, and AdapterH curves
plt.plot(parallel_adapter_params, parallel_adapter_Accuracy, marker='o', linestyle='-', color='blue', label='Parallel Adapter')
plt.plot(lora_params, lora_Accuracy, marker='s', linestyle='--', color='red', label='LoRA')
plt.plot(adapterH_params, adapterH_Accuracy, marker='^', linestyle='-.', color='green', label='AdapterH')

# Highlighting the area on the plot (as a percentage of 7e9)
MAXX_PERCENT = 1.8e9 / 7e9 * 100
plt.axvspan(3e8 / 7e9 * 100, MAXX_PERCENT, facecolor='peachpuff', alpha=0.5)
plt.axvspan(0, 7e6 / 7e9 * 100, facecolor='blue', alpha=0.1)
plt.xlim(left=1.5e6 / 7e9 * 100, right=MAXX_PERCENT)

# Adding labels and legend
plt.xlabel('Percentage of Trainable Parameters (relative to Llama-7B)', fontsize='large')
plt.ylabel('Accuracy', fontsize='large')
plt.legend(fontsize='large')

# Showing grid and setting x-axis to log scale
plt.grid(True)
plt.xscale('log')

import matplotlib.ticker as mtick
fmt = '%.1f%%' # 定义一个字符串格式，%.0f表示不显示小数，%%表示显示百分号
xticks = mtick.FormatStrFormatter(fmt) # 创建一个格式化程序
plt.gca().xaxis.set_major_formatter(xticks)

plt.savefig("nq_progress.pdf", format='pdf')
# Displaying the plot
plt.show()
