import pandas as pd
import torch
import numpy as np
import json

def load_config(config_file):
    # set default config file
    if config_file is None:
        config_file = "./config/config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


# metrics
def metrics(pred, label):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    mae = torch.tensor(mae, dtype=torch.float32)
    rmse = torch.tensor(rmse, dtype=torch.float32)
    mape = torch.tensor(mape, dtype=torch.float32)
    return mae, rmse, mape


def Seq2Instance(data, num_his, num_pred):
    """
        将时间序列数据 data 转换为模型训练所需的输入输出（滑动窗口构建）
        INPUT:
            data(num_step, num_vertex)
        OUTPUT:
            X(num_sample, num_his, num_vertex)
            Y(num_sample, num_pred, num_vertex)
    """
    num_step, num_vertex = data.shape
    num_sample = num_step - num_his - num_pred + 1
    X = torch.zeros(num_sample, num_his, num_vertex)
    Y = torch.zeros(num_sample, num_pred, num_vertex)
    for i in range(num_sample):
        X[i] = data[i:i+num_his]
        Y[i] = data[i+num_his:i+num_his+num_pred]
    return X,Y


def count_parameters(model):
    """
    打印模型中可训练参数的数量。

    参数：
        model: 模型对象。

    返回：
        可训练参数的总数量。
    """
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable parameters: {:,}'.format(parameters))
    return