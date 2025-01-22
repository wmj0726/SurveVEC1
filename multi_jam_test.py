
import math
import numpy as np
import pandas as pd
from env import Car_node, RSU_node, Translation
import env
import DRLVEC
import jammer
import torch
from tqdm import tqdm
import time
from sensors2graph import get_distance, get_adjacency_matrix, get_hacking_distance
from PoH_test import Blockchain, generate_keys, Node, simulate_hybrid, asynci, simulate_pow

blockchain = Blockchain()
publicKey, privateKey = generate_keys()

BLOCK_REWARD = 1
seq_len = 3
rsu_time_slot = 3
max_time_slot = 100  # 每个节点训练次数
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
car_id = '717469'#np.random.choice(sensor_ids) #范围内9个rsu

car_state = 'true'
car_msg = ['Car', 1 * 1000 *1000, 6.4 * 1000 *1000, 64 * 1000 *1000, 130, 5, 5, 50]

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

'''
hybrid_nodes = [Node(1, blockchain, 10, Car, publicKey, privateKey)]
for i in range(len(act_node) - 1):
    rsu1= RSU_node(act_node[i], Markov_b_up, Markov_gain, b_up_set, b_up_band, distance[i], 0)
    nodes = Node(i+2, blockchain, (i+2) * 10, rsu1, publicKey, privateKey)
    hybrid_nodes.append(rsu1)
'''
#
rsu = []
for i in range(len(act_node) - 1):
    rsu1= RSU_node(act_node[i], Markov_b_up, Markov_gain, b_up_set, b_up_band, distance[i], 0)
    rsu.append(rsu1)

for i in range(1):
    car = Car_node(car_id, car_state, car_msg)
    Car.append(car)
rsu1 = ['RSU1', rsu[0].b_up_set, rsu[0].b_up_band, rsu[0].distance]
rsu2 = ['RSU2', rsu[1].b_up_set, rsu[1].b_up_band, rsu[1].distance]
rsu3 = ['RSU3', rsu[2].b_up_set, rsu[2].b_up_band, rsu[2].distance]
rsu4 = ['RSU4', rsu[3].b_up_set, rsu[3].b_up_band, rsu[3].distance]
msg = [Car[0].message, rsu1, rsu2, rsu3, rsu4]
hybrid_nodes = [
    Node(1, blockchain, 10, msg[0], publicKey, privateKey),
    Node(2, blockchain, 20, msg[1], publicKey, privateKey),
    Node(3, blockchain, 30, msg[2], publicKey, privateKey),
    Node(4, blockchain, 40, msg[3], publicKey, privateKey),
    Node(5, blockchain, 50, msg[4], publicKey, privateKey)]
asynci(blockchain, hybrid_nodes, msg, privateKey)
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device_gpu)
save_uility = np.zeros((len(act_node), rsu_time_slot))
save_power1 = []
utilitys = np.zeros((9, max_time_slot))
save_delay = np.zeros((9, max_time_slot))
save_energy = np.zeros((9, max_time_slot))
save_sinr = np.zeros((9, max_time_slot))
save_power = np.zeros((9, max_time_slot))
save_offloadrate = np.zeros((9, max_time_slot))
save_R = np.zeros(((9, max_time_slot)))

def brodcast_msg(eposide, car_msg, rsu_msg1, rsu_msg2, rsu_id, select_jam_id):
    rsu_msg = [rsu_msg1, rsu_msg2]
    Car_id = Car[eposide].id
    trsan = Translation(Car_id, car_msg, rsu_msg, rsu_id, select_jam_id).generate_transaction()
    trsan_msg = [trsan['Car_ID'], trsan['car_request'], trsan['rsu_send'], trsan['Rsu_ID'], trsan['select_jam_id'],
                 BLOCK_REWARD]  # 广播的交易内容
    #publicKey, privateKey = generate_keys()
    node = Node(2, blockchain, 100, trsan_msg, publicKey, privateKey)
    node.mine_block_dag()
    return blockchain

def initialize_state_seq(block_trans_msg, select_rsu, jam_power, jam_gain, num_jam, total_power, rsu_h_gain):
    delay = vehicle.message[-2]
    energy = vehicle.message[-1]
    sinr = vehicle.sinr_set()[episode+2]
    state = np.hstack([band_rsu_up, delay, energy, sinr, 50, 0])  # len=9
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
        utility, delay, energy, sinr, h_sinr, power, R = vehicle.act_calculate_utility(block_trans_msg=block_trans_msg, offload_rate=offload_rate,
                                                                             b_up=b_up,total_power=total_power,
                                                                             gain=gain,jam_power=jam_power,
                                                                             jam_gain=jam_gain, rsu_h_gain=rsu_h_gain,
                                                                                       noise=noise, num_jam=num_jam)

        next_state = np.hstack((band_rsu_up, delay, energy, sinr, power, R))
        state = next_state
    return state_seq, delay, energy, sinr

def initialize_select_state_seq(episode, hacking_car_id, blockchain, num_jam, band_rsu_up, rsu_channel_gain):
    delay = vehicle.message[-2]
    energy = vehicle.message[-1]
    sinr = vehicle.sinr_set()[episode+2]
    h_sinr = vehicle.sinr_set()[episode]
    state = np.hstack([band_rsu_up, delay, energy, sinr, h_sinr, 200])  # len=9
    state_seq = []
    jam_id_to_select_rsu = [0]*8
    jam_power = [5, 5, 5]
    jam_gain = np.zeros(3)
    jam_select_dis = np.zeros(3)
    car_id = Car[episode].id
    action = np.random.uniform(-1, 1, size=action_space)#[-0.17053033, 0.64351244, 0.12009159]
    offload_rate, power = 0.839580599, 73.6027477#vehicle.action_to_actual(action)

    for i in range(len(rsu)):  # rsu数量
        min_jam_hacking_dis = 10000
        #power = 100
        #offload_rate = 0.9
        state_seq.extend(state.tolist())
        state_seq.extend(action.tolist())
        select_jamming = [0] * (3)
        jam_index = [num for num in range(0, len(rsu)) if num != i]
        #获取干扰器到窃听器的最小距离和干扰器的id
        for number in range(num_jam):
            min_jam_hacking_dis = 10000
            for jam in range(len(jam_index)):
                jam_hacking_dis = get_hacking_distance(distance_df, hacking_car_id, rsu[jam_index[jam]].rsu_id)
                if jam_hacking_dis < min_jam_hacking_dis:
                    min_jam_hacking_dis = jam_hacking_dis
                    x = jam_index[jam]
                    jam_id = rsu[jam_index[jam]].rsu_id
                    select_jamming[number] = jam_id
            jam_select_dis[number] = min_jam_hacking_dis
            jam_index = [num for num in jam_index if num != x]
            # 将选中的jam_id和RSU对应
        jam_id_to_select_rsu[i] = select_jamming
        b_up = band_rsu_up[i]
        gain = rsu_channel_gain[i]

        jam_to_hacking_dis = jam_select_dis  # 干-窃的距离
        for s in range(num_jam):
            jam_gain[s] = get_gain(jam_to_hacking_dis[s])
        car_to_hacking_dis = get_hacking_distance(distance_df, car_id, hacking_car_id)  # 车-窃
        car_to_hacking_gain = get_gain(car_to_hacking_dis)
        rsu_to_hacking_dis = get_hacking_distance(distance_df, rsu[i].rsu_id, hacking_car_id)
        rsu_h_gain = get_gain(rsu_to_hacking_dis)
        #sinr = car_to_rsu_gain * power / 0.5
        #h_sinr = car_to_hacking_gain * power / (jam_gain * jam_power + noise)
        interfer_power, interfer_gain = interfer.interference()
        noise = interfer_power * interfer_gain
        jam_power = get_jam_power(jam_gain, car_to_hacking_gain, power, sinr, noise) #获取最小干扰功率
        for block in blockchain.chain:
            if block.data[0] == 'Car':
                block_msg = block.data
        for j in range(rsu_time_slot):
            utility, delay, energy, sinr, h_sinr, R = vehicle.sim_calculate_utility(block_msg=block_msg,
                                                                                    offload_rate=offload_rate,
                                                                                    b_up=b_up,
                                                                                    power=power, gain=gain,
                                                                                    jam_power=jam_power,
                                                                                    jam_gain=jam_gain,
                                                                                    rsu_h_gain=rsu_h_gain,
                                                                                    noise=noise, num_jam=2)

            utility = torch.FloatTensor([[utility]]).to(device_gpu)
            save_uility[i, j] = utility
            save_power1.append(jam_power)
            next_state = np.hstack((band_rsu_up, delay, energy, sinr, h_sinr, R))
            state = next_state

        utilitys_rsu = save_uility.mean(axis=1)
        select_rsu = np.argmax(utilitys_rsu)
        # print('episode:{}, max utility and num is :'.format(episode), utilitys, max0, num0)
    return state_seq, select_rsu, jam_id_to_select_rsu[select_rsu], save_power1[
            select_rsu], delay, energy, sinr, h_sinr


def get_gain(hacking_dis):
    gaindB = [128.1 + 37.5 * math.log(hacking_dis, 10), 128.1 + 37.5 * math.log(hacking_dis + 30, 10)]
    gain = np.random.choice(gaindB, p=Markov_gain) + 30
    return gain

def get_jam_power(jam_gain, car_to_hacking_gain, power, sinr, noise):
    jam_power = (((car_to_hacking_gain * power * 10) / sinr) - noise) / jam_gain
    return jam_power

def get_jam_to_h_sinr(h_sinr, jam_power, jam_gain, noise):
    for i in range(2):
        h_sinr[i] = jam_power[i] * jam_gain[i] / noise
    return  h_sinr[0], h_sinr[1]

if __name__=="__main__":
    #publicKey, privateKey = Authentication.generate_key_pair()
    for episode in range(len(Car)):
        trsan = Translation(Car[episode].id, [], [], None, Node)
        band_rsu_up = []
        rsu_channel_gain = []
        vehicle = Car[episode]
        interfer = env.Interference()
        jammer = jammer.Jammer()
        for i in range(len(rsu)):
            band_rsu_up.append(rsu[i].rsu_band_up())
            rsu_channel_gain.append(rsu[i].rsu_channel_gain())
        if trsan.authenticate(Car, trsan) and len(rsu) > 0 : #验证车辆是否注册，并且state是否为true,并且当前节点范围内有其他通信的rsu
            action = np.random.uniform(-1, 1, size=action_space)
            state_dim = len(rsu)
            jam_gain = [0] * (3)
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
            state_seq_t, select_rsu, select_jam, jam_power, delay, energy, sinr, h_sinr = \
                initialize_select_state_seq(episode, hacking_car_id, blockchain, 3, band_rsu_up, rsu_channel_gain)            # 将训练得到的选择rsu节点与当前car的信息到区块中
            for s in range(3):
                jam_to_hacking_dis = get_hacking_distance(distance_df, car_id, select_jam[s])
                jam_gain[s] = get_gain(jam_to_hacking_dis)
            total_power = vehicle.action_to_actual(action)[1] + sum(jam_power)
            #print(select_rsu, select_jam)
            blockchain = brodcast_msg(episode, vehicle.message, band_rsu_up[select_rsu],
                         rsu_channel_gain[select_rsu], rsu[select_rsu].rsu_id, select_jam)

            b_up = blockchain.chain[-1].data[2][0]  # band_rsu_up
            gain = blockchain.chain[-1].data[2][1]  # channel_rsu
            rsu_to_hacking_dis = get_hacking_distance(distance_df, rsu[select_rsu].rsu_id, hacking_car_id)
            rsu_h_gain = get_gain(rsu_to_hacking_dis)

            car_to_hacking_dis = get_hacking_distance(distance_df, car_id, hacking_car_id)
            car_to_hacking_gain = get_gain(car_to_hacking_dis)

            interfer_power, interfer_gain = interfer.interference()
            noise = interfer_power * interfer_gain
            state_seq_t.extend(state.tolist())  # 将state数组转化为列表并扩展至state_seq_t的末尾，使用扩展合成的新列表代替原列表
            state_seq = torch.FloatTensor([[state_seq_t]]).to(device_gpu)
            epsilon = max_epsilon
            begin_time = time.time()
            init_h_sinr = [0, 0]
            block_trans_msg = blockchain.chain[-1].data
            for time_slot in tqdm(range(max_time_slot)):
                epsilon = min_epsilon + annealing_rate * (epsilon - min_epsilon)
                if np.random.random() > epsilon:
                    action = (agent.select_action(np.array(state_seq)) + np.random.normal(0,
                                                                                          max_action * expl_noise,
                                                                                          size=action_space)).clip(
                        -max_action, max_action)

                else:
                    action = np.random.uniform(-1, 1, size=action_space)
                if total_power > 100:
                    offload_rate, total_power = vehicle.action_to_actual(action)
                else:
                    offload_rate = vehicle.action_to_actual(action)[0]
                jam1_h_sinr, jam2_h_sinr = get_jam_to_h_sinr(init_h_sinr, jam_power, jam_gain, noise)
                jammer_state = [jam1_h_sinr / 10, jam2_h_sinr / 10]  # np.random.uniform(30, 90, size=2)
                interfer_power, interfer_gain = interfer.interference()
                noise = interfer_power * interfer_gain
                jammer_action_index, jam_power = jammer.jamming_power(jammer_state[1], epsilon, car_to_hacking_gain,
                                                                      noise, jam_gain, 2)
                utility, delay, energy, sinr, h_sinr, rel_power, R = vehicle.act_calculate_utility(block_trans_msg=block_trans_msg,
                    offload_rate=offload_rate,
                    b_up=b_up, total_power=total_power,
                    gain=gain, jam_power=jam_power,
                    jam_gain=jam_gain, rsu_h_gain=rsu_h_gain,
                    noise=noise, num_jam=3)
                jammer_utility = jammer.calculate_jammer_utility(vehicle_utility=utility,
                                                                 jammer_power=jam_power)
                save_jampower = sum(jam_power)
                total_power = rel_power + save_jampower
                next_jammer_state = [sinr / 10, jammer_state[0]]
                next_state = np.hstack((band_rsu_up, delay, energy, sinr, total_power, R))
                next_state_seq_t = initialize_state_seq(block_trans_msg, select_rsu, jam_power, jam_gain, 3, total_power,
                                                        rsu_h_gain)[0]
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
            endtime = time.time()


