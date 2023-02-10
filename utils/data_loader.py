#  nhan vao input la mot file
# xu ly ssa
# tra ve dang du lieu mong muon
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from utils.ssa import SSA
from utils.reprocess_daily import extract_data, ed_extract_data, roll_data
import matplotlib.pyplot as plt


def get_input_data(input_file, default_n, sigma_lst):
    dat = pd.read_csv(input_file, header=0)
    Q = dat['Q'].to_list()
    H = dat['H'].to_list()
    # print(Q[:5])
    # print(H[:5])
    lst_H_ssa = SSA(H, default_n)
    lst_Q_ssa = SSA(Q, default_n)

    H_ssa = lst_H_ssa.reconstruct(sigma_lst)
    Q_ssa = lst_Q_ssa.reconstruct(sigma_lst)

    dat['Q_ssa'] = Q_ssa
    # print(dat['Q'][:5])
    dat['H_ssa'] = H_ssa
    # print(dat['H'][:5])

    # print(dat.head())
    result = dat[['Q', 'H', 'Q_ssa', 'H_ssa']]

    fig = plt.figure(figsize=(10, 6))
    fig.add_subplot(121)
    plt.plot(Q[:200], label='Q_raw')
    plt.plot(Q_ssa[:200], label='Q_ssa')
    plt.legend()

    fig.add_subplot(122)
    plt.plot(H[:200], label='H_raw')
    plt.plot(H_ssa[:200], label='H_ssa')
    plt.legend()

    plt.savefig('log/model/ssa_processed.png')

    return result

def get_ssa_data(input_file, default_n):
    dat = pd.read_csv(input_file, header=0)
    Q = dat['Q'].to_list()
    H = dat['H'].to_list()
    # print(Q[:5])
    # print(H[:5])
    lst_H_ssa = SSA(H, default_n)
    lst_Q_ssa = SSA(Q, default_n)

    q_comp = lst_Q_ssa.TS_comps
    h_comp = lst_H_ssa.TS_comps

    Q = np.expand_dims(dat['Q'].to_numpy(), 1)
    H = np.expand_dims(dat['H'].to_numpy(), 1)

    QH_stack = np.hstack((Q, H))
    print(QH_stack.shape)
    print(q_comp.shape)
    return QH_stack, q_comp, h_comp

if __name__ == "__main__":
    # res = get_input_data('../data/SonTay.csv', 20, [1, 2, 3])
    # res.to_csv('../data/modified_data.csv', index=False)
    # print(res.head())
    get_ssa_data('../data/SonTay.csv', 20)
