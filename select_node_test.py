
import math
import numpy as np
import pandas as pd
from env import Car_node, RSU_node, Translation
import env
import DRLVEC
import jammer
import torch
from tqdm import tqdm
import os

import time
from sensors2graph import get_distance, get_adjacency_matrix, get_hacking_distance

pwd = os.getcwd() + '/' + '2024_4_35'
print(pwd)
os.makedirs(pwd)

BLOCK_REWARD = 1
seq_len = 3
rsu_time_slot = 3
max_time_slot = 100000  # 每个节点训练次数
max_epsilon = 1.0
min_epsilon = 0.02
annealing_rate = 0.9987
max_action = 1
action_space = 3
expl_noise = 0.1
Car = []
vir_node= []
act_node = []

with open("./data/sensor_graph/graph_sensor_ids.txt") as f:
    sensor_ids = f.read().strip().split(",")
# 读取距离csv文件中的内容，并以字符串的形式存储为一个Pandas DataFrame对象
distance_df = pd.read_csv("./data/sensor_graph/distances_la_2012.csv", dtype={"from": "str", "to": "str"})
adj_mx, sensor_id_to_ind = get_adjacency_matrix(distance_df, sensor_ids)  # 调用sensors2graph.py文件生成毗连矩阵，
#sp_mx = coo_matrix(adj_mx)  # 毗连矩阵稀疏化
#G = dgl.from_scipy(sp_mx)  # 将毗连矩阵转化为图
# print(sp_mx)
car_id = '717469'#np.random.choice(sensor_ids) #范围内9个rsu

car_state = 'true'
car_msg = [1 * 1000 *1000, 6.4 * 1000 *1000, 64 * 1000 *1000, 130, 5, 5, 50]

#rsu informat
Markov_b_up, Markov_gain = [0.9, 0.1], [0.9, 0.1]
b_up_set, b_up_band = [100.0, 60.0], 100.0
car_id_v = sensor_id_to_ind['{}'.format(car_id)]

for i in range(207):
    if adj_mx[car_id_v, i] != 0 and adj_mx[car_id_v, i] != 1:  #毗连矩阵中0代表超出通信范围，1代表节点本身
        vir_node.append(i) #范围内rsu的虚拟id编号

for i in range(len(vir_node)):
    id = next(k for k, v in sensor_id_to_ind.items() if v == vir_node[i])
    distance = get_distance(distance_df, car_id, id)
    act_node.append(id) #范围内rsu的实际id编号

hacking_car_id = act_node[-1] #最后一个节点作为窃听车辆

#
rsu = []
for i in range(len(act_node) - 1):
    rsu1= RSU_node(act_node[i], Markov_b_up, Markov_gain, b_up_set, b_up_band, distance[i], 0)
    rsu.append(rsu1)


for i in range(1):
    car = Car_node(car_id, car_state, car_msg, publicKey, BLOCK_REWARD)
    Car.append(car)



device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device_gpu)
save_uility = np.zeros((len(act_node), rsu_time_slot))
utilitys = np.zeros((9, max_time_slot))
save_delay = np.zeros((9, max_time_slot))
save_energy = np.zeros((9, max_time_slot))
save_sinr = np.zeros((9, max_time_slot))
save_power = np.zeros((9, max_time_slot))
save_offloadrate = np.zeros((9, max_time_slot))
save_R = np.zeros(((9, max_time_slot)))

def initialize_state_seq(select_rsu, jam_power, jam_gain):
    delay = vehicle.message[-2]
    energy = vehicle.message[-1]
    sinr = vehicle.sinr_set()[episode+2]
    h_sinr = vehicle.sinr_set()[episode]
    state = np.hstack([band_rsu_up, delay, energy, sinr, h_sinr, 200])  # len=9
    state_seq = []
    for j in range(8):
        action = np.random.uniform(-1, 1, size=action_space)
        offload_rate, power = vehicle.action_to_actual(action)
        state_seq.extend(state.tolist())
        state_seq.extend(action.tolist())
        b_up = band_rsu_up[select_rsu]
        gain = rsu_channel_gain[select_rsu]
        interfer_power, interfer_gain = interfer.interference()
        noise = interfer_power * interfer_gain
        utility, delay, energy, sinr, h_sinr, R = vehicle.sim_calculate_utility(offload_rate=offload_rate,
                                                                             b_up=b_up,
                                                                             power=power,
                                                                             gain=gain,
                                                                             jam_power=jam_power, jam_gain=jam_gain, noise=noise)

        next_state = np.hstack((band_rsu_up, delay, energy, sinr, h_sinr, R))
        state = next_state
    return state_seq

def initialize_select_state_seq(episode, hacking_car_id):
    delay = vehicle.message[-2]
    energy = vehicle.message[-1]
    sinr = vehicle.sinr_set()[episode+2]
    h_sinr = vehicle.sinr_set()[episode]
    state = np.hstack([band_rsu_up, delay, energy, sinr, h_sinr, 200])  # len=9
    state_seq = []
    jam_id_to_select_rsu = []
    car_id = Car[episode].id
    action = np.random.uniform(-1, 1, size=action_space)#[-0.17053033, 0.64351244, 0.12009159]
    offload_rate, power = 0.839580599, 73.6027477#vehicle.action_to_actual(action)

    for i in range(len(rsu)):  # rsu数量
        min_jam_hacking_dis = 10000
        #power = 100
        #offload_rate = 0.9
        state_seq.extend(state.tolist())
        state_seq.extend(action.tolist())
        jam_index = [num for num in range(0, len(rsu)) if num != i]
        #获取干扰器到窃听器的最小距离和干扰器的id
        for jam in range(len(jam_index)):
            jam_hacking_dis = get_hacking_distance(distance_df, hacking_car_id, rsu[jam_index[jam]].rsu_id)
            if jam_hacking_dis < min_jam_hacking_dis:
                min_jam_hacking_dis = jam_hacking_dis
                min_jam_hacking_dis_index = jam_index[jam]
                jam_id = rsu[min_jam_hacking_dis_index].rsu_id

        #将选中的jam_id和RSU对应
        jam_id_to_select_rsu.append(min_jam_hacking_dis_index)

        for j in range(rsu_time_slot):
            b_up = band_rsu_up[i]
            gain = rsu_channel_gain[i]

            #attacker_power = 50
            #attacker_gain = 0.5

            #jam_to_rsu_dis = get_hacking_distance(distance_df, jam_id, rsu[i].rsu_id)
            #jam_to_rsu_gain = get_gain(jam_to_rsu_dis) #干-RSU

            jam_to_hacking_dis = min_jam_hacking_dis  #干-窃的距离
            jam_gain = get_gain(jam_to_hacking_dis)

            #car_to_rsu_dis = get_hacking_distance(distance_df, car_id, rsu[i].rsu_id) #车-RSU
            #car_to_rsu_gain = get_gain(car_to_rsu_dis)

            car_to_hacking_dis = get_hacking_distance(distance_df, car_id, hacking_car_id) #车-窃
            car_to_hacking_gain = get_gain(car_to_hacking_dis)

            #sinr = car_to_rsu_gain * power / 0.5
            #h_sinr = car_to_hacking_gain * power / (jam_gain * jam_power + noise)
            interfer_power, interfer_gain = interfer.interference()
            noise = interfer_power * interfer_gain
            jam_power = get_jam_power(jam_gain, car_to_hacking_gain, power, sinr, noise) #获取最小干扰功率

            utility, delay, energy, sinr, h_sinr, R = vehicle.sim_calculate_utility(offload_rate=offload_rate,
                                                                         b_up=b_up,
                                                                         power=power,
                                                                         gain=gain,
                                                                         jam_power=jam_power, jam_gain=jam_gain, noise=noise)

            utility = torch.FloatTensor([[utility]]).to(device_gpu)
            save_uility[i, j] = utility

        next_state = np.hstack((band_rsu_up, delay, energy, sinr, h_sinr, R))
        state = next_state

    utilitys_rsu = save_uility.mean(axis=1)
    max0, num0 = utilitys_rsu[0], 0
    for le in range(len(utilitys_rsu) - 1):
        if max0 < utilitys_rsu[le + 1]:
            max0 = utilitys_rsu[le + 1]
            num0 = le + 1
    # print('episode:{}, max utility and num is :'.format(episode), utilitys, max0, num0)
    return state_seq, num0, jam_id_to_select_rsu[num0]



def get_gain(hacking_dis):
    gaindB = [128.1 + 37.5 * math.log(hacking_dis, 10), 128.1 + 37.5 * math.log(hacking_dis + 30, 10)]
    gain = np.random.choice(gaindB, p=Markov_gain) + 30
    return gain



def get_jam_power(jam_gain, car_to_hacking_gain, power, sinr, noise):
    jam_power = (((car_to_hacking_gain * power * 10) / sinr) - noise) / jam_gain
    return jam_power

if __name__=="__main__":
    #publicKey, privateKey = Authentication.generate_key_pair()
    for episode in range(len(Car)):
        trsan = Translation(Car[episode].id, [], [], None)
        band_rsu_up = []
        rsu_channel_gain = []
        select_rsu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        vehicle = Car[episode]
        interfer = env.Interference()
        jammer = jammer.Jammer()
        for i in range(len(rsu)):
            band_rsu_up.append(rsu[i].rsu_band_up())
            rsu_channel_gain.append(rsu[i].rsu_channel_gain())
        if trsan.authenticate(Car, trsan) and len(rsu) > 0 : #验证车辆是否注册，并且state是否为true,并且当前节点范围内有其他通信的rsu
            for k in range(9):
                state_dim = len(rsu)
                print(state_dim)
                agent = DRLVEC.drlvec(state_dim=state_dim, action_dim=3, max_action=1)
                delay = vehicle.message[-2]
                energy = vehicle.message[-1]
                sinr = vehicle.sinr_set()[episode+2]
                h_sinr = vehicle.sinr_set()[episode]
                state = np.hstack([band_rsu_up, delay, energy, sinr, h_sinr, 200])
                print(state)
                # print('state len', len(state))
                car_to_hacking_dis = get_hacking_distance(distance_df, car_id, hacking_car_id)
                car_to_hacking_gain = get_gain(car_to_hacking_dis)
                interfer_power, interfer_gain = interfer.interference()
                noise = interfer_power * interfer_gain
                if k != 0:
                    select_rsu[k] = np.random.choice(len(rsu))
                    select_jam = np.random.choice(sensor_ids)
                    jam_hacking_dis = get_hacking_distance(distance_df, hacking_car_id, select_jam)
                    jam_gain = get_gain(jam_hacking_dis)
                    jam_power = (((car_to_hacking_gain * 50 * 10) / sinr) - noise) / jam_gain
                    # 得到状态空间和选择的rsu
                    state_seq_t = initialize_state_seq(episode, jam_power, jam_gain)
                else:
                    state_seq_t, select_rsu[k], select_jam = initialize_select_state_seq(episode, hacking_car_id)
                # 将训练得到的选择rsu节点与当前car的信息到区块中


                b_up = b_up[select_rsu]
                gain = rsu_channel_gain[select_rsu]
                state_seq_t.extend(state.tolist())  # 将state数组转化为列表并扩展至state_seq_t的末尾，使用扩展合成的新列表代替原列表
                state_seq = torch.FloatTensor([[state_seq_t]]).to(device_gpu)

                epsilon = max_epsilon
                begin_time = time.time()

                # 源传感器到协调器和窃听器的距离,干扰传感器到协调器和窃听器的距离
                # jam_id = rsu[select_jam].rsu_id  #
                # jam_to_rsu_dis = get_hacking_distance(distance_df, jam_id, rsu[select_rsu].rsu_id)
                # jam_to_hacking_dis = get_hacking_distance(distance_df, jam_id, hacking_car_id)
                # car_to_rsu_dis = get_hacking_distance(distance_df, car_id, rsu[select_rsu].rsu_id)

                for time_slot in tqdm(range(max_time_slot)):
                    epsilon = min_epsilon + annealing_rate * (epsilon - min_epsilon)
                    if np.random.random() > epsilon:
                        action = (agent.select_action(np.array(state_seq)) + np.random.normal(0,
                                                                                              max_action * expl_noise,
                                                                                              size=action_space)).clip(
                            -max_action, max_action)
                    else:
                        action = np.random.uniform(-1, 1, size=action_space)

                    offload_rate, power = vehicle.action_to_actual(action)
                    jammer_state = [h_sinr / 10, h_sinr / 10]  # np.random.uniform(30, 90, size=2)
                    interfer_power, interfer_gain = interfer.interference()
                    noise = interfer_power * interfer_gain
                    jammer_action_index, jam_power, jam_gain = jammer.jamming_power(jammer_state[1], epsilon,
                                                                                    car_to_hacking_gain, noise)

                    utility, delay, energy, sinr, h_sinr, R = vehicle.act_calculate_utility(offload_rate=offload_rate,
                                                                                         b_up=b_up, power=power,
                                                                                         gain=gain, jam_power=jam_power,
                                                                                         jam_gain=jam_gain,
                                                                                         noise=noise)
                    jammer_utility = jammer.calculate_jammer_utility(vehicle_utility=utility,
                                                                     jammer_power=jam_power)

                    next_jammer_state = [sinr, jammer_state[0]]
                    next_state = np.hstack((band_rsu_up, delay, energy, sinr, h_sinr, R))
                    next_state_seq_t = initialize_state_seq(select_rsu[k], jam_power, jam_gain)
                    next_state_seq_t.extend(next_state.tolist())

                    next_state_seq = torch.FloatTensor([[next_state_seq_t]]).to(device_gpu)
                    action = torch.FloatTensor([[action]]).to(device_gpu)
                    utility = torch.FloatTensor([[utility]]).to(device_gpu)
                    done = torch.FloatTensor([[0.0]]).to(device_gpu)

                    exp = agent.experience_pool.experience(state_seq, action, utility, done, next_state_seq)

                    agent.experience_pool.save_experience(exp)

                    agent.train()
                    jammer.update_Qtable(jammer_state[1], jammer_action_index, jammer_utility,
                                         jammer_state[0])

                    jammer_state = next_jammer_state
                    state_seq_t = next_state_seq_t
                    state_seq = next_state_seq

                    if utility < 0:
                        utilitys[k, time_slot] = utilitys[k, time_slot - 1]
                    else:
                        utilitys[k, time_slot] = utility
                    save_delay[k, time_slot] = delay
                    save_energy[k, time_slot] = energy
                    save_sinr[k, time_slot] = sinr
                    save_power[k, time_slot] = power
                    save_R[k, time_slot] = R
                    #save_offloadrate[k, time_slot] = offload_rate
                    # print('car {} utility is :'.format(episode), save_uility[episode])

                endtime = time.time()

    for i in range(9):
        np.savetxt(pwd + f"/DQN_random{i}_utility.txt", utilitys[i])
        np.savetxt(pwd + f"/DQN_random{i}_delay.txt", save_delay[i])
        np.savetxt(pwd + f"/DQN_random{i}_energy.txt", save_energy[i])
        np.savetxt(pwd + f"/DQN_random{i}_sinr.txt", save_sinr[i])
        np.savetxt(pwd + f"/DQN_random{i}_power.txt", save_power[i])
        np.savetxt(pwd + f"/DQN_random{i}_R.txt", save_R[i])


