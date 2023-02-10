import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Dense, Input, LSTMCell, LSTM, Reshape, Concatenate, Conv1D, TimeDistributed, MultiHeadAttention, Attention, Bidirectional
# from tensorflow_addons.seq2seq import BahdanauAttention as BAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import sys
import os
import argparse
import yaml
import tensorflow.keras.backend as K
import tensorflow as tf
from utils.ssa import SSA
from utils.reprocess_daily import ssa_extract_data, extract_data, transform_ssa
from utils.data_loader import get_ssa_data, get_input_data
from tensorflow.keras.activations import softmax
from utils.custom_losses import shrinkage_loss, linex_loss


def getMonth(_str):
    return _str.split('/')[1]


def getYear(_str):
    return _str.split('/')[2]


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calcError(row):
    item_df = {}
    item_df['var_score_q'] = r2_score(row['real_q'], row['ensemble_q'])
    item_df['mse_q'] = mean_squared_error(row['real_q'], row['ensemble_q'])
    item_df['mae_q'] = mean_absolute_error(row['real_q'], row['ensemble_q'])
    item_df['mape_q'] = mean_absolute_percentage_error(row['real_q'], row['ensemble_q'])

    item_df['var_score_h'] = r2_score(row['real_h'], row['ensemble_h'])
    item_df['mse_h'] = mean_squared_error(row['real_h'], row['ensemble_h'])
    item_df['mae_h'] = mean_absolute_error(row['real_h'], row['ensemble_h'])
    item_df['mape_h'] = mean_absolute_percentage_error(row['real_h'], row['ensemble_h'])

    return pd.Series(item_df,
                     index=['var_score_q', 'mse_q', 'mae_q', 'mape_q', 'var_score_h', 'mse_h', 'mae_h', 'mape_h'])

class Ensemble:
    def __init__(self, mode, model_kind, child_option, **kwargs):
        self.mode = mode
        self.model_kind = model_kind
        self.child_config = self.merge_child_config(child_option[self.model_kind], kwargs['model']['child'])

        self.log_dir = kwargs.get('log_dir')
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._ssa_kwargs = kwargs.get('ssa')

        self.data_file = self._data_kwargs.get('data_file')
        self.dt_split_point_outer = self._data_kwargs.get('split_point_outer')
        self.dt_split_point_inner = self._data_kwargs.get('split_point_inner')
        self.cols_x = self._data_kwargs.get('cols_x')
        self.cols_y = self._data_kwargs.get('cols_y')
        self.cols_gt = self._data_kwargs.get('cols_gt')
        self.pred_factor = self._data_kwargs.get('pred_factor')
        self.target_timestep = self._data_kwargs.get('target_timestep')
        self.window_size = self._data_kwargs.get('window_size')
        self.norm_method = self._data_kwargs.get('norm_method')
        self.time_step_eval = self._data_kwargs.get('time_step_eval')

        self.batch_size = self._model_kwargs.get('batch_size')
        self.epochs_out = self._model_kwargs.get('epochs_out')
        self.input_dim = self._model_kwargs.get('in_dim')
        self.output_dim = self._model_kwargs.get('out_dim')
        self.patience = self._model_kwargs.get('patience')
        self.dropout = self._model_kwargs.get('dropout')

        self.sigma_lst = self._ssa_kwargs['sigma_lst']
        self.default_n = self._ssa_kwargs['default_n']

        self.data = self.generate_data()
        self.inner_models = self.build_model_inner()
        self.outer_model = self.build_model_outer()

    def merge_child_config(self, default_setting, override_setting):
        ''' Merge 2 dictionary for config of child models, 
            the 1st dict is the default for all,
            the 2nd for override setting for making differences between childs
        '''
        for k, v in override_setting.items():
            default_setting[k] = v
        print(default_setting)
        return default_setting

    def generate_data(self):
        '''
        Prepare train-val-test set
        train_in: data use for train child models
        test_in: data use for TRAIN main model
        test_out: data use for actual test of the total system 
        '''
        # dat = get_input_data(self.data_file, self.default_n, self.sigma_lst)
        dat = pd.read_csv(self.data_file, header=0)
        dat = dat[['Q', 'H']]
        dat = dat.to_numpy()
        # QH_stacked, Q_comps, H_comps  = get_ssa_data(self.data_file, self.default_n)

        data = {}
        # data['shape'] = QH_stacked.shape
        # test_outer = int(QH_stacked.shape[0] * self.dt_split_point_outer)
        # train_inner = int((QH_stacked.shape[0] - test_outer) * (1 - self.dt_split_point_inner))

        data['shape'] = dat.shape
        test_outer = int(dat.shape[0] * self.dt_split_point_outer)
        train_inner = int((dat.shape[0] - test_outer) * (1 - self.dt_split_point_inner))
        
        if self.model_kind == 'rnn_cnn':
            xq, xh, scaler, y_gt = extract_data(dataframe=dat, window_size=self.window_size, 
                                                target_timstep=self.target_timestep,
                                                cols_x=self.cols_x, cols_y=self.cols_y,
                                                cols_gt=self.cols_gt, mode=self.norm_method)

            xq = transform_ssa(xq, self.default_n, self.sigma_lst)
                                                
            # xq, xh, scaler, y_gt = ssa_extract_data(gtruth=QH_stacked,
            #                                         q_ssa=Q_comps,
            #                                         h_ssa= H_comps,
            #                                         window_size=self.window_size,
            #                                         target_timstep=self.time_step_eval,
            #                                         mode=self.norm_method)
            
            # xq = np.concatenate((xq, xh), axis=2)
            if self.pred_factor == 'q':
                x_train_in, y_gt_train_in = xq[:train_inner, :], y_gt[:train_inner,:, :]
                x_test_in, y_gt_test_in = xq[train_inner:-test_outer, :], y_gt[train_inner:-test_outer,:, :]
                x_test_out, y_gt_test_out = xq[-test_outer:, :],y_gt[-test_outer:, :, :]
            else:
                x_train_in, y_gt_train_in = xh[:train_inner, :], y_gt[:train_inner, :, 1][:,:, np.newaxis]
                x_test_in, y_gt_test_in = xh[train_inner:-test_outer, :], y_gt[train_inner:-test_outer,:, 1][:,:, np.newaxis]
                x_test_out, y_gt_test_out = xh[-test_outer:, :],y_gt[-test_outer:,:, 1][:,:, np.newaxis]
                
            for cat in ["train_in", "test_in", "test_out"]:
                x, y_gt = locals()["x_" + cat], locals()["y_gt_" + cat]
                print(cat, "x: ", x.shape, "ygtr: ", y_gt.shape)
                data["x_" + cat] = x
                data["y_" + cat] = y_gt

        data['scaler'] = scaler
        return data

    def plot_data_processed(self, x, y):
        '''
        use to plot the data after processed - not important
        '''
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,7))
        ax1.plot(range(len(y)), y[:, 0])
        ax1.set(xlabel='num', ylabel='H')
        ax2.plot(range(len(y)), y[:, 1])
        ax2.set(xlabel='num', ylabel='Q')
        
        plt.savefig('./log/data.png')

    def build_model_inner(self):
        '''
        func to build the child models
        '''
        models = []
        for i in range(self.child_config['num']):
            if self.model_kind == 'rnn_cnn':
                from models.multi_rnn_cnn import model_builder
                model = model_builder(i, self.child_config, self.input_dim, self.output_dim, self.window_size, self.time_step_eval)
                models.append(model)

        print(models[0].summary())
        return models

    def train_model_inner(self):
        '''
        function for train the child models
        '''
        #prepare train data for main model, which is the result of child models
        train_shape = self.data['y_test_in'].shape
        test_shape = self.data['y_test_out'].shape
        # print(train_shape)
        x_train_out = np.zeros(shape=(train_shape[0], self.target_timestep, self.child_config['num'], train_shape[2]))
        x_test_out = np.zeros(shape=(test_shape[0], self.target_timestep, self.child_config['num'], test_shape[2]))
        # print(x_train_out.shape)
        if (self.mode == 'train' or self.mode == 'train-inner'):
            for i in range(self.child_config['num']):
                if self.model_kind == 'rnn_cnn':
                    from models.multi_rnn_cnn import train_model
                    self.inner_models[i], _ = train_model(self.inner_models[i], i, 
                                                      self.data['x_train_in'],
                                                      self.data['y_train_in'],
                                                      self.child_config['batch_size'][i],
                                                      self.child_config['epoch'][i],
                                                      save_dir=self.log_dir + 'ModelPool/',
                                                      pred_factor=self.pred_factor)

                # gen train data for main model with prediction of i-th child
                train, test = self.predict_in(i)
                x_train_out[:, :, i, :] = train
                x_test_out[:, :, i, :] = test
        else:
            for i in range(self.child_config['num']):
                # for epoch in range(self.epoch_min, self.epoch_max + 1, self.epoch_step):
                if self.model_kind == 'rnn_cnn':
                    self.inner_models[i].load_weights(self.log_dir + f'ModelPool/best_model_{self.pred_factor}_{i}.hdf5')
                
                train, test = self.predict_in(i)
                x_train_out[:, :, i, :] = train
                x_test_out[:, :, i, :] = test

        self.data_out_generate(x_train_out, x_test_out)

    def predict_in(self, index=0, data=[]):
        if len(data) == 0: # this case for gen train data
            if self.model_kind == 'rnn_cnn':
                x_train_out = self.inner_models[index].predict(self.data['x_test_in'])
                x_test_out = self.inner_models[index].predict(self.data['x_test_out'])
           
            return x_train_out, x_test_out
        else: # this case when gen data during evaluation
            x_test_out = np.zeros((self.child_config['num'], self.output_dim))
            for ind in range(self.child_config['num']):
                # self.inner_models[ind].load_weights(self.log_dir + f'ModelPool/best_model_{self.pred_factor}_{ind}.hdf5')
                # print(f'output val {ind}: {self.inner_models[ind].predict(data, batch_size=1)}')
                x_test_out[ind, :] = self.inner_models[ind].predict(data, batch_size=1)
            # print('Now it go out bro')
            # here if apply attention then should not flatten out
            x_test_out = x_test_out.reshape(1, -1)
            return x_test_out

    def data_out_generate(self, x_train_out, x_test_out):
        '''
        retransform the dimension of outputs from submodels to use as the training data for main model
        '''
        # this flatten all output of submodel, should not do this when applying attention
        shape = x_train_out.shape
        self.data['x_train_out_submodel'] = x_train_out.reshape(shape[0], shape[1], -1) 
        print(self.data['x_train_out_submodel'].shape)
        self.data['y_train_out'] = self.data['y_test_in'] 

        shape = x_test_out.shape
        self.data['x_test_out_submodel'] = x_test_out.reshape(shape[0], shape[1], -1)
        self.data['y_test_out'] = self.data['y_test_out'] 

    def build_model_outer(self):
        '''
        build main model - Modify this for attention mechanism
        '''
        self.train_model_inner()
        in_shape = self.data['x_train_out_submodel'].shape
        print(f'Input shape: {in_shape}')

        # modify the dim here
        input_submodel = Input(shape=(self.target_timestep, self.output_dim * self.child_config['num']))
        input_val_x = Input(shape=(self.window_size, self.input_dim))
        
        conv = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')
        conv_out = conv(input_val_x)

        # conv2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
        # conv_out2 = conv2(conv_out)

        # rnn_1 = Bidirectional(
        #   LSTM(units=128,
        #         return_sequences=True,
        #         return_state=True,
        #         dropout=0,
        #         recurrent_dropout=0))

        rnn_1 = LSTM(units=128,
                return_sequences=True,
                return_state=True,
                dropout=self.dropout,
                recurrent_dropout=self.dropout)

        rnn_1_out, state_h, state_c = rnn_1(conv_out)
        # rnn_1_out, forward_h, forward_c, backward_h, backward_c = rnn_1(input_val_x)
        # state_h = Concatenate(axis=-1)([forward_h, backward_h])
        # state_c = Concatenate(axis=-1)([forward_c, backward_c])
        states = [state_h, state_c]

        decoder_cell = LSTMCell(
            units=128,
            dropout=self.dropout,
            recurrent_dropout=self.dropout)
        
        predictions = []
        attention = Attention()
        Wc = Dense(units=128, activation=tf.math.tanh, use_bias=False)
        for i in range(self.target_timestep):
            decoder_inp = input_submodel[:, i, :]
            output, states = decoder_cell(decoder_inp, states=states)
            output = tf.expand_dims(output, axis=1) # shape (batch, 1, hidden_size)
            context_vec = attention([output, rnn_1_out])
            context_and_rnn_2_out = Concatenate(axis=-1)([context_vec, output])
            attention_vec = Wc(context_and_rnn_2_out)
            predictions.append(attention_vec)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.squeeze(tf.transpose(predictions, [2, 1, 0, 3]), axis=0)

        # dense_3 = TimeDistributed(Dense(units=16, activation='relu'))
        # output_3 = dense_3(predictions)

        dense_4 = TimeDistributed(Dense(units=self.output_dim))
        output = dense_4(predictions)

        model = Model(inputs=[input_submodel, input_val_x], outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss=linex_loss, optimizer=optimizer, metrics=['mae', 'mape'])

        model.summary()
        return model

    def train_model_outer(self):
        '''
        func for train the main model
        '''
        if (self.mode == 'train' or self.mode == 'train-outer'):
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

            callbacks = []
            #lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
            early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
            checkpoint = ModelCheckpoint(self.log_dir + 'best_model.hdf5',
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True)

            # callbacks.append(lr_schedule)
            callbacks.append(early_stop)
            callbacks.append(checkpoint)

            history = self.outer_model.fit(x=[self.data['x_train_out_submodel'], self.data['x_test_in']],
                                            y=self.data['y_train_out'],
                                            batch_size=self.batch_size,
                                            epochs=self.epochs_out,
                                            callbacks=callbacks,
                                            validation_split=0.1)

            if history is not None:
                self.plot_training_history(history)

        elif self.mode == 'test':
            self.outer_model.load_weights(self.log_dir + 'best_model.hdf5')
            print('Load weight from ' + self.log_dir)


    def plot_training_history(self, history):
        '''
        plot training phase
        '''
        fig = plt.figure(figsize=(10, 6))
        # fig.add_subplot(121)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()

        plt.savefig(self.log_dir + 'training_phase.png')

    def roll_prediction(self):
        '''
        Roll prediction to the actual target timestep
        '''
        result = []
        gtruth = []
        for ind in range(len(self.data['x_test_out']) - self.time_step_eval):
            x = self.data['x_test_out'][ind]
            gt = []
            # print(f'input: {x.shape}')

            # gen data from child models
            res0_sub = self.predict_in(data=x[np.newaxis, :])
            # print(f'result submodel: {res0_sub.shape}')
            res0 = self.outer_model.predict(x=[res0_sub[np.newaxis, :], x[np.newaxis, :]], batch_size=1)
            # print(f'result model: {res0.shape}')
            x = x.tolist()
            x.append(res0.reshape(self.output_dim).tolist())
            gt.append(self.data['y_test_out'][ind])

            for i in range(1, self.time_step_eval):
                print(np.array(x[-self.window_size:]).shape)
                res_sub = self.predict_in(data=np.array(x[-self.window_size:])[np.newaxis, :])
                res = self.outer_model.predict(
                    x=[res_sub[np.newaxis, :], np.array(x[-self.window_size:])[np.newaxis, :]], batch_size=1)
                gt.append(self.data['y_test_out'][ind + i])
                x.append(res.reshape(self.output_dim).tolist())

            result.append(x[-self.time_step_eval:])
            gtruth.append(gt)

        result = np.array(result)
        gtruth = np.array(gtruth)
        print(f'RESULT SHAPE: {result.shape}')
        print(f'GTRUTH SHAPE: {gtruth.shape}')

        return result, gtruth
    
    def multistep_prediction(self):
        '''
        Mutistep prediction, simutaneously
        '''    
        pred = self.outer_model.predict(x=[self.data['x_test_out_submodel'], self.data['x_test_out']], 
                                        batch_size=self.batch_size)
        
        # print(pred.shape) 
        # print(self.data['y_test_out'].shape)   
        
        return pred, self.data['y_test_out']
    
    def retransform_prediction(self):
        '''
        func for convert the prediction to initial scale
        '''
        if self.target_timestep == 1:
            result, y_test = self.roll_prediction()
        else:
            result, y_test = self.multistep_prediction()

        mask = np.zeros(self.data['shape'])
        test_shape = self.data['y_test_out'].shape[0]

        lst_full_date = pd.read_csv(self.data_file)['date'].tolist()
        len_df = int(len(lst_full_date) * (1 - self.dt_split_point_outer) - 1)

        for i in range(self.time_step_eval):
            total_frame = pd.DataFrame()

            mask[-test_shape:, self.cols_gt] = y_test[:, i, :]
            actual_data = self.data['scaler'].inverse_transform(mask)[-test_shape:, self.cols_gt]

            mask[-test_shape:, self.cols_y] = result[:, i, :]
            actual_predict = self.data['scaler'].inverse_transform(mask)[-test_shape:, self.cols_y]

            total_frame['real_q'] = actual_data[:, 0]
            total_frame['real_h'] = actual_data[:, 1]
            total_frame['ensemble_q'] = actual_predict[:, 0]
            total_frame['ensemble_h'] = actual_predict[:, 1]
            total_frame['date'] = lst_full_date[len_df:len_df + len(actual_data)]

            print('SAVING CSV...')
            import pdb; pdb.set_trace()
            total_frame.to_csv('./log/data_analysis/predict_val_{}.csv'.format(i), index=None)

    def evaluate_model(self):
        '''
        Run evaluation
        '''
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        lst_data = []
        for i in range(self.time_step_eval):
            df = pd.read_csv('./log/data_analysis/predict_val_{}.csv'.format(i))
            actual_dat = df[['real_q', 'real_h']]
            actual_pre = df[['ensemble_q', 'ensemble_h']]

            item_df = {}
            item_df['var_score_q'] = r2_score(actual_dat.iloc[:, 0], actual_pre.iloc[:, 0])
            item_df['mse_q'] = mean_squared_error(actual_dat.iloc[:, 0], actual_pre.iloc[:, 0])
            item_df['mae_q'] = mean_absolute_error(actual_dat.iloc[:, 0], actual_pre.iloc[:, 0])
            item_df['mape_q'] = mean_absolute_percentage_error(actual_dat.iloc[:, 0], actual_pre.iloc[:, 0])

            item_df['var_score_h'] = r2_score(actual_dat.iloc[:, 1], actual_pre.iloc[:, 1])
            item_df['mse_h'] = mean_squared_error(actual_dat.iloc[:, 1], actual_pre.iloc[:, 1])
            item_df['mae_h'] = mean_absolute_error(actual_dat.iloc[:, 1], actual_pre.iloc[:, 1])
            item_df['mape_h'] = mean_absolute_percentage_error(actual_dat.iloc[:, 1], actual_pre.iloc[:, 1])
            lst_data.append(item_df)

        eval_df = pd.DataFrame(
            data=lst_data,
            columns=['var_score_q', 'mse_q', 'mae_q', 'mape_q', 'var_score_h', 'mse_h', 'mae_h', 'mape_h'])
        eval_df.to_csv('./log/data_analysis/total_error.csv')

        # visualize
        df_viz = pd.read_csv('./log/data_analysis/predict_val_0.csv')
        actual_dat = df_viz[['real_q', 'real_h']]
        actual_pre = df_viz[['ensemble_q', 'ensemble_h']]

        fig = plt.figure(figsize=(10, 6))
        fig.add_subplot(121)
        plt.plot(actual_dat.iloc[:, 0], label='actual_ground_truth_Q')
        plt.plot(actual_pre.iloc[:, 0], label='actual_predict_Q')
        plt.legend()

        fig.add_subplot(122)
        plt.plot(actual_dat.iloc[:, 1], label='ground_truth_H')
        plt.plot(actual_pre.iloc[:, 1], label='predict_H')
        plt.legend()

        plt.savefig(self.log_dir + 'predict_actual.png')

        # write score
        _str = f'Model: H: R2: {np.mean(eval_df["var_score_h"].tolist())} MSE: {np.mean(eval_df["mse_h"].tolist())} MAE: {np.mean(eval_df["mae_h"].tolist())} MAPE: {np.mean(eval_df["mape_h"].tolist())} \
                            \nQ: R2: {np.mean(eval_df["var_score_q"].tolist())} MSE: {np.mean(eval_df["mse_q"].tolist())} MAE: {np.mean(eval_df["mae_q"].tolist())} MAPE: {np.mean(eval_df["mape_q"].tolist()) }\n'

        print(_str)
        with open(self.log_dir + 'evaluate_score_total.txt', 'a') as f:
            f.write(_str)

        return (np.mean(eval_df["mse_q"].tolist()), np.mean(eval_df["mse_h"].tolist()))

if __name__ == '__main__':
    K.clear_session()
    np.random.seed(99)
    tf.random.set_seed(99)

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str, help='Run mode.')
    parser.add_argument('--model', default='rnn_cnn', type=str, help='Model used.')
    args = parser.parse_args()

    with open('./settings/model/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open('./settings/model/child_default.yaml', 'r') as f:
        child_config = yaml.load(f, Loader=yaml.FullLoader)
    if args.mode == 'train' or args.mode == 'train-inner' or args.mode == 'train-outer':
        model = Ensemble(args.mode, args.model, child_config, **config)
        model.train_model_outer()
        model.retransform_prediction()
        model.evaluate_model()
    elif args.mode == "test":
        model = Ensemble(args.mode, args.model, child_config, **config)
        model.train_model_outer()
        model.retransform_prediction()
        model.evaluate_model()
        # model.multistep_prediction()
    else:
        raise RuntimeError('Mode must be train or test!')
