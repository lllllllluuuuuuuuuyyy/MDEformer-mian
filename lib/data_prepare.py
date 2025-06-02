import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import csv
from .utils import print_log, StandardScaler, vrange

def get_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False, batch_size=16, log=None
):
    # ------------------------ 邻接矩阵 ----------------------------
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
    csv_file_path = 'D:\MDEformer\MDEformer\data\PEMS08\PeMS08.csv'
    num_nodes = 170
    graph_type = "connect"  # 设置为使用距离作为权重
    A = np.zeros([int(num_nodes), int(num_nodes)])  # 构造全0的邻接矩阵
    with open(csv_file_path, "r") as f_d:
        f_d.readline()  # 表头，跳过第一行
        reader = csv.reader(f_d)  # 读取.csv文件
        for item in reader:  # 将每一行组成列表赋值给item
            if len(item) != 3:  # 长度应为3，不为3则数据有问题，跳过
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])
            if graph_type == "connect":  # 这个就是将两个节点的权重都设为1，也就相当于不要权重
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":  # 这个是有权重，下面是权重计算方法
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")
    # 将邻接矩阵保存为 Numpy 文件
    np.save('pems08_01_adj.npy', A)
    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    data = data[..., features]  # （17856，170，3）3指的是流量，日周期，周周期，月周期

    # ---------------------------- 节假日标签 --------------------------------
    date_types = [
        # ---------------- 08 -------------------------
        3, 1, 2, 2, 3, 0, 0,  # 7月1日-7日
        3, 1, 2, 3, 0, 0, 0, # 7月8日-14日
        3, 1, 2, 3, 0, 0, 0, # 7月15日-21日
        3, 1, 2, 3, 0, 0, 0,  # 7月22日-28日
        3, 1, 2,  # 7月29日-31日
        3, 0, 0, 0, 3, 1, 2,  # 8月1日-7日
        3, 0, 0, 0, 3, 1, 2,  # 8月8日-14日
        3, 0, 0, 0, 3, 1, 2,  # 8月15日-21日
        3, 0, 0, 0, 3, 1, 2,  # 8月22日-28日
        3, 0, 0  # 8月29日-31日

        # 0, 1, 2, 2, 0, 0, 0,  # 7月1日-7日
        # 0, 1, 2, 0, 0, 0, 0,  # 7月8日-14日
        # 0, 1, 2, 0, 0, 0, 0, # 7月15日-21日
        # 0, 1, 2, 0, 0, 0, 0,  # 7月22日-28日
        # 0, 1, 2,  # 7月29日-31日
        # 0, 0, 0, 0, 0, 1, 2,  # 8月1日-7日
        # 0, 0, 0, 0, 0, 1, 2,  # 8月8日-14日
        # 0, 0, 0, 0, 0, 1, 2,  # 8月15日-21日
        # 0, 0, 0, 0, 0, 1, 2,  # 8月22日-28日
        # 0, 0, 0  # 8月29日-31日

        # 0, 1, 2, 2, 0, 0, 0,  # 7月1日-7日
        # 0, 1, 2, 0, 0, 0, 0,  # 7月8日-14日
        # 0, 1, 2, 0, 0, 0, 0,  # 7月15日-21日
        # 0, 1, 2, 0, 0, 0, 0,  # 7月22日-28日
        # 0, 1, 2,  # 7月29日-31日
        # 0, 0, 0, 0, 0, 1, 2,  # 8月1日-7日
        # 0, 0, 0, 0, 0, 1, 2,  # 8月8日-14日
        # 0, 0, 0, 0, 0, 1, 2,  # 8月15日-21日
        # 0, 0, 0, 0, 0, 1, 2,  # 8月22日-28日
        # 0, 0, 0  # 8月29日-31日
    ]
    encoded_labels = []
    for date_type in date_types:
        encoded_labels.extend([date_type] * 288)

    encoded_labels = np.array(encoded_labels)
    encoded_labels = encoded_labels[:, np.newaxis]

    # 将 encoded_labels 扩展为 (17856, 170)，复制170次
    encoded_labels_expanded = np.repeat(encoded_labels, 170, axis=1)
    encoded_labels_expanded = encoded_labels_expanded[..., np.newaxis]
    data = np.concatenate((data, encoded_labels_expanded), axis=-1)

    index = np.load(os.path.join(data_dir, "index.npz"))
    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]
    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = torch.utils.data.TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainset_loader, valset_loader, testset_loader, scaler
