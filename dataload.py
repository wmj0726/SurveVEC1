import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse import coo_matrix
import torch
import dgl
from sklearn.preprocessing import StandardScaler
from sensors2graph import get_adjacency_matrix


def data_transform(data, n_his, n_pred, device):
    # produce data slices for training and testing
    n_route = data.shape[1] # 获取数据集对应的采样数量
    l = len(data)
    num = l - n_his - n_pred
    x = np.zeros([num, 1, n_his, n_route])
    y = np.zeros([num, n_route])

    cnt = 0
    for i in range(l - n_his - n_pred):
        head = i
        tail = i + n_his
        x[cnt, :, :, :] = data[head:tail].reshape(1, n_his, n_route)
        y[cnt] = data[tail + n_pred - 1]
        cnt += 1
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Dataload():
    with open("./data/sensor_graph/graph_sensor_ids.txt") as f:
        sensor_ids = f.read().strip().split(",")
    '''读取距离csv文件中的内容，并以字符串的形式存储为一个Pandas DataFrame对象'''
    distance_df = pd.read_csv("./data/sensor_graph/distances_la_2012.csv", dtype={"from": "str", "to": "str"})

    adj_mx = get_adjacency_matrix(distance_df, sensor_ids)  # 调用sensors2graph.py文件生成毗连矩阵
    #adj_mx = torch.tensor(adj_mx)
    sp_mx = coo_matrix(adj_mx)  # 毗连矩阵稀疏化
    G = dgl.from_scipy(sp_mx)  # 将毗连矩阵转化为图

    # 读取metr-la.h5文件内容，并获取数据集中的样本数量和节点数量
    df = pd.read_hdf("./data/metr-la.h5")
    num_samples, num_nodes = df.shape#34272
    print(num_samples, num_nodes)
    n_his = 60
    n_pred = 5
    n_route = num_nodes
    batch_size = 5
    # 划分数据集
    len_val = round(num_samples * 0.1)
    len_train = round(num_samples * 0.7)
    train = df[:len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]

    # 标准化数据集
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    # 划分数据样本，生成Pytorch的Tensor对象,调用load_data.py文件
    x_train, y_train = data_transform(train, n_his, n_pred, device)
    x_val, y_val = data_transform(val, n_his, n_pred, device)
    x_test, y_test = data_transform(test, n_his, n_pred, device)

    # 生成训练集、验证集和测试集。使用迭代器输出小批量数据
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)  # len=2385
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size)  # len=671

    # 训练模型
    #adj_mx1 = torch.randn(207, 207)  # 邻接矩阵
    return train_iter, test_iter, val_iter, G, n_route, adj_mx
