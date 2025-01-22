import numpy as np
import pandas as pd

distance = []
def get_distance(distance_df, rsu_id, target_id):
    for row in distance_df.values:
        # print(row[0], row[1])
        if row[0] == rsu_id and row[1] == target_id:
            distance.append(row[2])
    return distance

def get_hacking_distance(distance_df, car_id, hacking_car_id, dis=19999):
    for row in distance_df.values:
        #print(row[0], row[1], row[2])
        if row[0] == car_id and row[1] == hacking_car_id:
            dis = row[2]
    return dis

def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.3):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return: adjacency matrix
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf  # 将所有元素均赋值为正无穷进行占位
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i  # {'77386': 0, '767541': 1,  ......}
    '''

    Fills cells in the matrix with distances.
    如果其中任意一个传感器 ID 没有对应的节点索引，则跳过当前行的处理。
    如果两个传感器 ID 都有对应的节点索引，则将距离数据框中对应行的距离值 row[2] 赋值给距离矩阵 dist_mx 中对应的元素。
    '''
    # print(sensor_id_to_ind)
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
    '''
    Calculates the standard deviation as theta.
    首先计算距离的标准差，然后依据公式w_{i,j} = \exp(-\frac{(d_{i,j}/\sigma)^2}{2})计算得到毗连矩阵
    '''
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    # 距离大于3000的节点使之为0
    adj_mx[adj_mx < normalized_k] = 0  # 距离：2504.6 值：3.9095539e-01 ；值越大，距离越小
    # print(adj_mx)
    return adj_mx, sensor_id_to_ind

if __name__=="__main__":
    with open("./data/sensor_graph/graph_sensor_ids.txt") as f:
        sensor_ids = f.read().strip().split(",")
    # 读取距离csv文件中的内容，并以字符串的形式存储为一个Pandas DataFrame对象
    distance_df = pd.read_csv("./data/sensor_graph/distances_la_2012.csv", dtype={"from": "str", "to": "str"})
    adj_mx, sensor_id_to_ind = get_adjacency_matrix(distance_df, sensor_ids)  # 调用sensors2graph.py文件生成毗连矩阵，
    car_id = np.random.choice(sensor_ids)
    hacking_car_id = np.random.choice(sensor_ids)
    print(car_id, hacking_car_id)
    hacking_dis = get_hacking_distance(distance_df, car_id, hacking_car_id, dis=9999)
    print(hacking_dis)