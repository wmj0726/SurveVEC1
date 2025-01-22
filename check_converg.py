import math
import numpy as np
# 文件名列表
file_names = ['DQN_select_utility.txt', 'SafeQ1_utility.txt', 'Q1_select_utility.txt', 'WBAN1_utility.txt']
data = [[], [], [], []]
for i in range(4):
    # 读取文件数据
    with open(file_names[i], 'r') as file:
        data[i] = [float(line.strip()) for line in file]

select, random1, random2, random3 = 0, 0, 0, 0
select0, random11, random22, random33 = [], [], [], []
for i in range(100000):
    if i % 999 != 0:
        select += data[0][i]
        random1 += data[1][i]
        random2 += data[2][i]
        random3 += data[3][i]
    else:
        select0.append(select/1000)
        random11.append(random1/1000)
        random22.append(random2/1000)
        random33.append(random3/1000)
        select, random1, random2, random3 = 0, 0, 0, 0

def check_convergence(rewards, episode, window_size, threshold):
    if episode < window_size:
        return None
    # 移动平均窗口大小
    moving_average_window = window_size

    # 计算移动平均值
    moving_averages = [sum(rewards[i:i + moving_average_window]) / moving_average_window for i in range(len(rewards) - moving_average_window + 1)]

    # 收敛阈值（移动平均值变化的百分比）
    convergence_threshold = threshold  # 比如 1%

    # 检查移动平均值的相对变化
    for i in range(1, len(moving_averages)):
        change_percentage = abs((moving_averages[i] - moving_averages[i - 1]) / moving_averages[i - 1])
        if change_percentage < convergence_threshold:
            print(f"在episode={episode - len(moving_averages) + i + moving_average_window - 1}时，算法可能已收敛（移动平均值变化百分比：{change_percentage:.3%}）")
            return episode - len(moving_averages) + i + moving_average_window - 1
    else:
        print("遍历完所有移动平均值，算法尚未收敛或数据不足以判断。")
        return episode


convergence_threshold = 0.0003
convergence_window = 15
converged_episode, converged_episode1, converged_episode2, converged_episode3 = None, None, None, None

if __name__ == "__main__":
    converged_episode = check_convergence(select0, 100, convergence_window, convergence_threshold)
    stability = np.std(select0[-convergence_window:])
    #print(f"稳定性: {stability}")
    converged_episode1 = check_convergence(random11, 100, convergence_window, convergence_threshold)
    stability = np.std(random11[-convergence_window:])
    #print(f"稳定性: {stability}")
    converged_episode2 = check_convergence(random22, 100, convergence_window, convergence_threshold)
    stability = np.std(random22[-convergence_window:])
    #print(f"稳定性: {stability}")
    converged_episode3 = check_convergence(random33, 100, convergence_window, convergence_threshold)
    stability = np.std(random33[-convergence_window:])
    #print(f"稳定性: {stability}")
