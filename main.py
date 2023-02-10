import sys
from ensemble import Ensemble
import yaml
import tensorflow.keras.backend as K
from utils.data_loader import get_input_data
from utils.ssa import SSA
import pandas as pd
from ga.GA import GA
from ga.Individual import Individual
import numpy as np
import os
import math
if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")


def get_list_sigma_result(default_n=10):
    with open('./settings/model/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_file = config['data']['data_file']
    df = pd.read_csv(data_file)
    Q = df['Q'].to_list()
    H = df['H'].to_list()
    H_ssa_L20 = SSA(H, default_n)
    Q_ssa_L20 = SSA(Q, default_n)

    lst_sigma_H = H_ssa_L20.get_lst_sigma()
    lst_sigma_Q = Q_ssa_L20.get_lst_sigma()
    return lst_sigma_Q


def reward_func(sigma_index_lst=[1, 2, 3], default_n=10, epoch_num=4):
    '''
    input
    sigma_lst - The component index from the ssa gene for example the gen [0, 1, 0] -> sigma_lst=[1] #the index where gen=1
    default_n - the window length for ssa - <= N /2 where N is the length of the time series - default 10
    output
    a tuple contain 2 value (mse_q, mse_h)
    '''
    K.clear_session()
    # override setting
    config['model']['child']['num'] = epoch_num
    config['model']['child']['epoch'] = [15*i for i in range(1, epoch_num + 1)]
    config['model']['child']['dropout'] = [0 for i in range(epoch_num)]
    config['model']['child']['batch_size'] = [128 for i in range(epoch_num)] 
    config['ssa']['sigma_lst'] = sigma_index_lst

    child_config['rnn_cnn']['conv']['n_kernels'] = [[32, 32, 128] for i in range(epoch_num)]
    child_config['rnn_cnn']['conv']['kernel_s'] = [[7, 5, 3] for i in range(epoch_num)]
    child_config['rnn_cnn']['lstm']['bi_unit'] = [64 for i in range(epoch_num)]
    child_config['rnn_cnn']['lstm']['si_unit'] = [128 for i in range(epoch_num)]
    child_config['rnn_cnn']['lr'] = [0.0001 for i in range(epoch_num)]
    # train
    model = Ensemble(mode='train', model_kind='rnn_cnn', child_option=child_config, **config)
    model.train_model_outer()
    model.retransform_prediction()
    return model.evaluate_model()


def fitness(ind):
    sigma_index_lst = []
    for i in range(ind.size):
        if ind.genes[i] == 1:
            sigma_index_lst.append(i)
    if len(sigma_index_lst) == 0:
        return 100000000
    fitnesss = reward_func(sigma_index_lst=sigma_index_lst, default_n=10,
                           epoch_num=ind.n)[0]
    return fitnesss


def sigma_init(sigma_input):
    tmp = []
    output_lst = []
    input_sum = sum(sigma_input)
    for j in sigma_input:
        output_lst.append(j / input_sum)
    print(output_lst)
    return output_lst


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    q = get_list_sigma_result()

    with open('./settings/model/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open('./settings/model/child_default.yaml', 'r') as f:
        child_config = yaml.load(f, Loader=yaml.FullLoader)

    sigma = sigma_init(q)

    print("((((((((((((((((((   q   )))))))))))))))))))")
    print(q)
    print("(((((((((((((((((( sigma )))))))))))))))))))")
    print(sigma)

    pop = GA(sigma, fitness)
    pop.run()

    # gene = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0]
    # n = 6
    # sigma_index_lst = []
    # for i in range(len(gene)):
    #     if gene[i] == 1:
    #         sigma_index_lst.append(i)
    # fitnesss = reward_func(sigma_index_lst=sigma_index_lst, default_n=20,
    #                        epoch_num=n, epoch_min=100, epoch_step=50)[0]
    # print(fitnesss)
# test2
