import numpy as np
import math
import random
import json
import blockchain1
import blockchain3
from authentication import Authentication

# 设定参与边缘计算的车辆参数
class Car_node:
    def __init__(self, id, state, message, reward):
        self.id = id
        self.state = state
        if state == 'true':
            self.message = message
            self.reward = reward
        else:
            self.message = 'null'
            self.reward = 0
        self.energy = 5  # 设定能耗权重
        self.delay = 5    # 设定时延权重
        self.sinr = 50    # 设定信噪比权重
        self.R = 1.5   #设定保密权重

    # 此处设定的信噪比集合用于构建Q学习的状态
    def sinr_set(self):
        self.sinr1 = math.floor(128.1 + 37.5 * math.log(10, 10) + 30)
        self.sinr2 = math.floor(128.1 + 37.5 * math.log(50, 10) + 30)
        self.sinr3 = math.floor(128.1 + 37.5 * math.log(90, 10) + 30)
        self.sinr4 = math.floor(128.1 + 37.5 * math.log(130, 10) + 30)
        self.sinr5 = math.floor(128.1 + 37.5 * math.log(170, 10) + 30)
        self.sinr6 = math.floor(128.1 + 37.5 * math.log(210, 10) + 30)
        return [self.sinr1, self.sinr2, self.sinr3, self.sinr4, self.sinr5, self.sinr6]

    #系统模型
    def sim_calculate_utility(self, offload_rate, b_up, power, gain, jam_power, jam_gain, noise):
        C = 1 * 1000 *1000
        loc_cal_rate = 6.4 * 1000 *1000
        clo_cal_rate = 64 * 1000 *1000
        loc_cal_energy = 130
        h_sinr = power * gain / (jam_gain * jam_power + noise) # noise = interfer * interfer_gain 源车-窃听车 ；jam * jam_gain干扰车 此处通过计算得出信噪比的值 用与距离相关的公式
        rel_power = h_sinr * noise / gain
        n = np.random.uniform(-0.4, 0.1)
        rel_sinr = h_sinr - (n * rel_power * gain / noise)
        if 0 < offload_rate <= 1:
            share_gain = offload_rate * C / 1000
        else:
            share_gain = C / 1000

        loc_cal_delay = (1 - offload_rate) * C / loc_cal_rate * 1000  # 车辆的本地计算延迟
        clo_trans_up_delay = offload_rate * C / (b_up * 1000)

        # RSU上的计算延迟
        if 0 < offload_rate <= 1:
            clo_cal_delay = offload_rate * C/ clo_cal_rate * 1000
        else:
            clo_cal_delay = 0
        total_delay = max(loc_cal_delay, clo_cal_delay + clo_trans_up_delay)  # 计算总延时

        loc_cal_energy = (1 - offload_rate) * loc_cal_energy  # 车辆的本地计算能量消耗
        loc_trans_up_energy = rel_power * offload_rate * C / (b_up * 100)  # 卸载传输能量消耗
        jam_trans_up_energy = jam_power * offload_rate * C / (b_up * 1000)#干扰器发送的干扰能量消耗
        loc_energy = loc_cal_energy + loc_trans_up_energy + jam_trans_up_energy # 计算计算能量消耗

        #powerdBm = 10 * math.log(power, 10)

        #rel_sinr = power * gain / noise #源车-rsu

        R = b_up * (math.log((1 + rel_sinr), 2) - math.log((1 + h_sinr), 2)) #保密能力值
        utility = share_gain - self.delay * total_delay - self.energy * loc_energy - self.R * R  # 多目标优化函数
        # utility = math.log(utility)
        return utility, total_delay, loc_energy, rel_sinr, power, R



    # 生成强化学习动作向量，调整卸载策略和能量
    def action_to_actual(self, action):
        offload_rate = 0.45 * action[1] + 0.55
        power = 30 * action[2] + 70
        return offload_rate, power

    def action2_to_actual(self, action):
        offload_rate = 0.45 * action[1] + 0.55
        return offload_rate

    def act_calculate_utility(self, offload_rate, b_up, power, gain, jam_power, jam_gain, noise):
        C = 1 * 1000 *1000
        loc_cal_rate = 6.4 * 1000 *1000
        clo_cal_rate = 64 * 1000 *1000
        loc_cal_energy = 130

        total_sum = jam_power * jam_gain + noise
        #powerdBm = 10 * math.log(power, 10)
        h_sinr = 0.1 * power * gain / total_sum
        rel_power = 10 * h_sinr * noise / gain
        n = np.random.uniform(-0.4, 0.1)
        rel_sinr = h_sinr - (0.1 * n * rel_power * gain / noise)
        #rel_sinr = power * gain / noise  # 此处通过计算得出信噪比的值 用与距离相关的公式
        if 0 < offload_rate <= 1:
            share_gain = offload_rate * C / 1000
        else:
            share_gain = C / 1000

        loc_cal_delay = (1 - offload_rate) * C / loc_cal_rate * 1000  # 车辆的本地计算延迟
        clo_trans_up_delay = offload_rate * C / (b_up * 1000)

        # RSU上的计算延迟
        if 0 < offload_rate <= 1:
            clo_cal_delay = offload_rate * C / clo_cal_rate * 1000
        else:
            clo_cal_delay = 0

        total_delay = max(loc_cal_delay, clo_cal_delay + clo_trans_up_delay)  # 计算总延时

        loc_cal_energy = (1 - offload_rate) * loc_cal_energy  # 车辆的本地计算能量消耗
        loc_trans_up_energy = rel_power * offload_rate * C / (b_up * 1000000)  # 卸载传输能量消耗
        jam_trans_up_energy = jam_power * offload_rate * C / (b_up * 100000)  # 干扰器发送的干扰能量消耗
        loc_energy = loc_cal_energy + loc_trans_up_energy + jam_trans_up_energy  # 计算计算能量消耗
        R = b_up * (math.log((1 + self.sinr * rel_sinr), 2) - math.log((1 + self.sinr * h_sinr), 2)) #保密能力值
        utility = share_gain - total_delay - self.energy * loc_energy - R  # 多目标优化函数, 加入jam_energy, jam_sinr
        # utility = math.log(utility)
        return utility, total_delay, loc_energy, rel_sinr, h_sinr, rel_power, R




# 定义RSU类
class RSU_node:
    def __init__(self, rsu_id, Markov_b_up, Markov_gain, b_up_set, b_up_band, distance, reward):
        self.rsu_id = rsu_id
        self.Markov_b_up = Markov_b_up #马尔科夫选择带宽
        self.b_up_set = b_up_set #传输带宽集合
        self.b_up_band = b_up_band #实际带宽
        self.distance = distance
        self.Markov_gain = Markov_gain #马尔科夫信道增益
        self.reward = reward

    # rsu与车辆通信带宽
    def rsu_band_up(self):
        b_up_band = np.random.choice(self.b_up_set, p=self.Markov_b_up)

        return b_up_band

    #车辆与RSU之间的信道增益
    def gain(self):
        gaindB = [128.1 + 37.5 * math.log(self.distance, 10), 128.1 + 37.5 * math.log(self.distance + 30, 10)]
        return gaindB

    #选择信道增益
    def rsu_channel_gain(self):
        up_gain = np.random.choice(self.gain(), p=self.Markov_gain) + 30
        return up_gain

    def hacking_gain(self, distance):
        gaindB = [128.1 + 37.5 * math.log(distance, 10), 128.1 + 37.5 * math.log(distance + 30, 10)]
        gain = np.random.choice(gaindB, p=self.Markov_gain) + 30
        return gain

class Translation:
    def __init__(self, car_id, car_msg, rsu_msg, rsu_id):
        self.car_id = car_id
        self.car_msg = car_msg
        self.rsu_msg = rsu_msg
        self.rsu_id = rsu_id

    def authenticate(self, Car, trans):
        for i in range(len(Car)):
            #if car_id == Car[i].id:
            if Car[i].state == 'true':
                trans.car_msg = Car[i].message
                #trans.rsu_msg = [rsu[i].rsu_band_up(), rsu[i].rsu_channel_gain()]
                return True
            else:
                print('state is false')
                return False

        print("need register car_node")#未知车辆注册
        '''
                publicKey, privateKey = Authentication.generate_key_pair()
        car_id = '75261'
        Car.append(Car_node(car_id, 'true', [1 * 1000 *1000, 6.4 * 1000 *1000, 64 * 1000 *1000, 130, 5, 5, 50], publicKey, 0))
        trans.car_msg = Car[-1].message
        blockchain.add_message(Car[len(Car)-1], privateKey, publicKey)
        last_block = blockchain.chain[-1]
        last_proof = last_block['proof']
        proof = blockchain.proof_of_work(last_proof)
        blockchain.add_block(proof)
        '''
        return True



    def generate_transaction(self):
        # 生成计算卸载请求的交易
        transaction = {
            'Car_ID': self.car_id,
            'car_request': self.car_msg,
            'rsu_send': self.rsu_msg,
            'Rsu_ID': self.rsu_id,
            # 其他交易信息
        }
        return transaction




class Interference:
    def __init__(self):
        super(Interference, self).__init__()
        self.Markov_interfer_power = [[0.2, 0.7, 0.1],
                                      [0.1, 0.8, 0.1],
                                      [0.1, 0.7, 0.2]]
        self.inter_power_set = [8, 10, 12]
        self.inter_power = 10

        self.Markov_interfer_gain = [[0.1, 0.9],
                                     [0.9, 0.1]]
        self.inter_gain_set = [0.12, 0.15]
        self.interfer_gain = 0.12

    def interference(self):
        interfer_power = np.random.choice(self.inter_power_set, p=self.Markov_interfer_power[
           self.inter_power_set.index(self.inter_power)])
        self.inter_powerdBm = 10 * math.log(interfer_power, 10)

        interfer_gain = np.random.choice(self.inter_gain_set, p=self.Markov_interfer_gain[
            self.inter_gain_set.index(self.interfer_gain)])
        self.interfer_gaindBm = 10 * math.log(interfer_gain, 10) + 30
        return self.inter_powerdBm, self.interfer_gaindBm
        #return self.interfer_gaindBm

