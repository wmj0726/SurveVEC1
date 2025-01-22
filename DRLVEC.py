import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.initial_seed()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.c1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)      # 一维卷积建立，建立卷积层。输入通道数为1，输出通道数为16，内核为5
        self.c2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        #input = ((state_dim * state_dim) + (7 * state_dim) - 4) * 32
        print(input)
        self.l1 = nn.Linear(4288, 500)      # 构建全连接隐藏层
        self.l2 = nn.Linear(500, 200)
        self.l3 = nn.Linear(200, 150)
        self.l4 = nn.Linear(150, action_dim)

        self.max_action =max_action

    def forward(self, state):
        hidden = self.c1(state)
        hidden = self.c2(hidden)
        hidden = hidden.view(hidden.size(0), -1)
        a = F.tanh(self.l1(hidden))
        a = F.tanh(self.l2(a))
        a = F.tanh(self.l3(a))
        return self.max_action * torch.tanh(self.l4(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.c1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.c2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        input = ((state_dim * state_dim) + (7 * state_dim)) * 32
        #print(input)
        self.l1 = nn.Linear(4384, 500)
        self.l2 = nn.Linear(500, 200)
        self.l3 = nn.Linear(200, 150)
        self.l4 = nn.Linear(150, 1)

        self.c11 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.c22 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.l5 = nn.Linear(4384, 500)
        self.l6 = nn.Linear(500, 200)
        self.l7 = nn.Linear(200, 150)
        self.l8 = nn.Linear(150, 1)

    def forward(self, state, action):
        sa = torch.cat((state, action), -1)  # sa's size [128, 1, 66]

        c1_value = self.c1(sa)
        c1_value = self.c2(c1_value)
        c1_value = c1_value.view(c1_value.size(0), -1)   # [128, 1920]
        c1_value = F.relu(self.l1(c1_value))
        c1_value = F.relu(self.l2(c1_value))
        c1_value = F.relu(self.l3(c1_value))
        c1_value = self.l4(c1_value)

        c2_value = self.c11(sa)
        c2_value = self.c22(c2_value)
        c2_value = c2_value.view(c2_value.size(0), -1)    # [128, 1920]
        c2_value = F.relu(self.l5(c2_value))
        c2_value = F.relu(self.l6(c2_value))
        c2_value = F.relu(self.l7(c2_value))
        c2_value = self.l8(c2_value)
        return c1_value, c2_value

    def Q1(self, state, action):
        sa = torch.cat((state, action), -1)

        c1_value = self.c1(sa)
        c1_value = self.c2(c1_value)
        c1_value = c1_value.view(c1_value.size(0), -1)
        c1_value = F.relu(self.l1(c1_value))
        c1_value = F.relu(self.l2(c1_value))
        c1_value = F.relu(self.l3(c1_value))
        c1_value = self.l4(c1_value)
        return c1_value


class N_step_experience_pool(object):
    def __init__(self, experience_size=5000, n_multi_step=8, discount=0.5):
        self.experience = namedtuple("agent_experience", ['state', 'action', 'reward', 'done', 'next_state'])
        self.experience_pool = []
        self.pool_size = experience_size
        self.n_multi_step = n_multi_step
        self.discount = discount
        self.next_idx = 0

    def save_experience(self, exp):
        date = exp
        if len(self.experience_pool) <= self.pool_size:
            self.experience_pool.append(date)
        else:
            self.experience_pool[self.next_idx] = date
        self.next_idx = (self.next_idx + 1) % self.pool_size

    def sample(self, batch_size=128):
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for i in range(batch_size):
            finish = random.randint(self.n_multi_step, self.pool_size - 1)
            begin = finish-self.n_multi_step
            sum_reward = 0
            data = self.experience_pool[begin:finish]
            # print(data.index(max(data)), data.index(max(data)))
            state = data[0][0]
            action = data[0][1]
            for j in range(self.n_multi_step):
                sum_reward += (self.discount**j) * data[j][2]
                if data[j][4]:
                    states_look_ahead = data[j][3]
                    done_look_ahead = True
                    break
                else:
                    states_look_ahead = data[j][3]
                    done_look_ahead = False

            states.append(state)
            actions.append(action)
            rewards.append(sum_reward)
            next_states.append(states_look_ahead)
            dones.append(done_look_ahead)

        return np.concatenate(states), actions, rewards, dones, np.concatenate(next_states)
'''
    def size(self):
        return len(self.experience_pool)
'''

class drlvec(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.5, tau=0.1, polic_noise=0.2, noise_clip=0.5, policy_freq=2, experience_size=5000):

        self.actor_train = Actor(state_dim, action_dim, max_action).to(device)   # 将状态参数传入Actor网络进行运算，得出神经网络输出值
        self.actor_target = copy.deepcopy(self.actor_train)  # 构建完全相同的目标评估网络
        self.actor_optimizer = torch.optim.Adam(self.actor_train.parameters(), lr=0.001)  # 使用Adam梯度下降算法更新神经网络权值

        self.critic_train = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic_train)
        self.critic_optimizer = torch.optim.Adam(self.critic_train.parameters(), lr=0.005)

        self.experience_pool = N_step_experience_pool(experience_size)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = polic_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor_train(state).cpu().data.numpy().flatten()

    def train(self, batch_size=128, n_multi_step=8, sum_r=0):  # 以当前测试结果来看，在1至15步内，当步数为8时最优
        self.total_it += 1

        # 从经验池中获取以保存的内容
        batch_experience = random.choices(self.experience_pool.experience_pool, k=batch_size)   # 从经验池中随机选择一个数据输出，重复128次
        batch_experience = self.experience_pool.experience(*zip(*batch_experience))
        state = torch.cat(batch_experience.state)  # state's size [128, 1, 63]
        action = torch.cat(batch_experience.action)
        reward = torch.cat(batch_experience.reward)
        done = torch.cat(batch_experience.done)
        next_state = torch.cat(batch_experience.next_state)  # [128, 1, 63]
        with torch.no_grad():
            next_action = self.actor_target(next_state)

            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            next_action = next_action.view(next_action.size(0), -1, next_action.size(1))
            target_Q1 , target_Q2= self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            for i in range(1, n_multi_step+1):
                r = self.discount**i
                sum_r += r

            target_Q = reward + (1-done)*sum_r * target_Q

        current_Q1, current_Q2 = self.critic_train(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            action = self.actor_train(state)
            action = action.view(action.size(0), -1, action.size(1))
            actor_loss = -self.critic_train.Q1(state, action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for train_parameters, target_parameters in zip(self.critic_train.parameters(), self.critic_target.parameters()):
                train_parameters.data.copy_(self.tau * train_parameters.data + (1 - self.tau) * target_parameters.data)

            for train_parameters, target_parameters in zip(self.actor_train.parameters(), self.actor_target.parameters()):
                train_parameters.data.copy_(self.tau * train_parameters.data + (1 - self.tau) * target_parameters.data)

        return critic_loss
