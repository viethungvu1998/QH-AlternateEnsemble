import pandas as pd
import numpy as np
from utils.ssa import SSA

def normalize_data(dataframe, mode):
    if mode == 'abs':
        from sklearn.preprocessing import MaxAbsScaler
        max_abs = MaxAbsScaler(copy=True)  #save for retransform later
        max_abs.fit(dataframe)
        data_norm = max_abs.transform(dataframe)

        return data_norm, max_abs

    if mode == 'robust':
        from sklearn.preprocessing import RobustScaler
        robust = RobustScaler(copy=True)  #save for retransform later
        robust.fit(dataframe)
        data_norm = robust.transform(dataframe)

        return data_norm, robust

    if mode == 'min_max':
        from sklearn.preprocessing import MinMaxScaler
        minmax = MinMaxScaler(feature_range=(0, 1), copy=True)  #save for retransform later
        minmax.fit(dataframe)
        data_norm = minmax.transform(dataframe)

        return data_norm, minmax
    if mode == 'std':
        from sklearn.preprocessing import StandardScaler
        stdscaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        stdscaler.fit(dataframe)
        data_norm = stdscaler.transform(dataframe)

        return data_norm, stdscaler


def extract_data(dataframe, window_size=5, target_timstep=1, cols_x=[], cols_y=[], cols_gt=[],mode='std'):
    '''
    The function for splitting the data
    '''
    dataframe, scaler = normalize_data(dataframe, mode)

    xs = [] # return input data
    ys = [] # return output data
    ygt = [] # return groundtruth data

    if target_timstep != 1:
        for i in range(dataframe.shape[0] - window_size - target_timstep):
            xs.append(dataframe[i:i + window_size, cols_x])
            ys.append(dataframe[i + window_size:i + window_size + target_timstep,
                                cols_y])
            ygt.append(dataframe[i + window_size:i + window_size + target_timstep,
                       cols_gt])
    else:
        for i in range(dataframe.shape[0] - window_size - target_timstep):
            xs.append(dataframe[i:i + window_size, cols_x])
            ys.append(dataframe[i + window_size, cols_y])
            ygt.append(dataframe[i + window_size, cols_gt])
    return np.array(xs), np.array(ys), scaler, np.array(ygt)

def transform_ssa(input, n, sigma_lst):
    print(input.shape)
    step = input.shape[0]
    qs = []
    hs = []
    for i in range(step):
        lst_H_ssa = SSA(input[i, :, 0], n)
        lst_Q_ssa = SSA(input[i, :, 1], n)
        q_comp = lst_Q_ssa.TS_comps
        h_comp = lst_H_ssa.TS_comps

        q_merged = lst_Q_ssa.reconstruct(sigma_lst)
        h_merged = lst_H_ssa.reconstruct(sigma_lst)
        qs.append(q_merged)
        hs.append(h_merged)
    
    qs = np.array(qs)
    hs = np.array(hs)
    result = np.concatenate((qs[:,:, np.newaxis], hs[:,:, np.newaxis]), axis=2)
    print(result.shape)
    return result
    
def ssa_extract_data(gtruth, q_ssa, h_ssa, window_size=7, target_timstep=1, mode='std'):
    '''
    generate data with separate ssa components
    '''
    gtruth, scaler_gtr = normalize_data(gtruth, mode)
    q_ssa, _ = normalize_data(q_ssa, mode)
    h_ssa, _ = normalize_data(h_ssa, mode)

    xs_q = [] # return input data
    xs_h = [] # return input data
    ygt = [] # return groundtruth data

    if target_timstep != 1:
        for i in range(gtruth.shape[0] - window_size - target_timstep):
            xs_q.append(q_ssa[i:i + window_size, :])
            xs_h.append(h_ssa[i:i + window_size, :])
            ygt.append(gtruth[i + window_size:i + window_size + target_timstep, :])
    else:
        for i in range(gtruth.shape[0] - window_size - target_timstep):
            xs_q.append(q_ssa[i:i + window_size, :])
            xs_h.append(h_ssa[i:i + window_size, :])
            ygt.append(gtruth[i + window_size, :])

    return np.array(xs_q), np.array(xs_h), scaler_gtr, np.array(ygt)

def ed_extract_data(dataframe, window_size=5, target_timstep=1, cols_x=[], cols_y=[], mode='std'):
    dataframe, scaler = normalize_data(dataframe, mode)

    en_x = []
    de_x = []
    de_y = []

    for i in range(dataframe.shape[0] - window_size - target_timstep):
        en_x.append(dataframe[i:i + window_size, cols_x])

        #decoder input is q and h of 'window-size' days before
        de_x.append(dataframe[i + window_size - 1:i + window_size + target_timstep - 1,
                              cols_y].reshape(target_timstep, len(cols_y)))
        de_y.append(dataframe[i + window_size:i + window_size + target_timstep,
                              cols_y].reshape(target_timstep, len(cols_y)))

    en_x = np.array(en_x)
    de_x = np.array(de_x)
    de_y = np.array(de_y)
    de_x[:, 0, :] = 0

    return en_x, de_x, de_y, scaler

def roll_data(dataframe, cols_x, cols_y, mode='min_max'):
    dataframe, scaler = normalize_data(dataframe, mode)
    #dataframe = dataframe.drop('time', axis=1)

    X = dataframe[:, cols_x]
    y = dataframe[:, cols_y]

    return X, y, scaler


