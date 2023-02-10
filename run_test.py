import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import yaml
import tensorflow.keras.backend as K
import tensorflow as tf
from ensemble import Ensemble
from models.transformer import Transformer
from models.rnn_cnn import RnnCnn
from models.alstm import ALSTM
from models.necplus import NecPlus
import torch 
import torch.nn as nn 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Run mode.')
    parser.add_argument('--model', default='transformer', type=str, help='Model used.', \
                            choices=['transformer', 'rnn-cnn', 'alstm', 'necplus'])
    parser.add_argument('--child_model', default='rnn_cnn', type=str, help='Model used.')
    args = parser.parse_args()
    return args 

if __name__=="__main__":
    K.clear_session()
    np.random.seed(99)
    tf.random.set_seed(99)

    sys.path.append(os.getcwd())
    
    with open('./settings/model/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open('./settings/model/child_default.yaml', 'r') as f:
        child_config = yaml.load(f, Loader=yaml.FullLoader)

    args = parse_args()
    type_class = {
        'transformer': Transformer,
        'alstm': ALSTM,
        'necplus': NecPlus
    }

    try: 
        model = type_class[args.model]
    except:
        raise RuntimeError('Mode must be train or test!')    
    
    if args.mode == 'train':
        model.train()
    elif args.mode == 'test':
        model.test()
    else:
        raise RuntimeError('Mode must be train or test!')