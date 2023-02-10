import pandas as pd 
from utils.reprocess_daily import ssa_extract_data, extract_data, transform_ssa

class BaseModel:
    def __init__(self) -> None:
        pass

    def train():
        pass 

    def test():
        pass 

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
        data['shape'] = dat.shape
        xs, ys, scaler, y_gt = extract_data(dataframe=dat, window_size=self.window_size, 
                                                target_timstep=self.target_timestep,
                                                cols_x=self.cols_x, cols_y=self.cols_y,
                                                cols_gt=self.cols_gt, mode=self.norm_method)  
        for cat in ["train", "val", "test"]:
            x, y_gt = locals()["x_" + cat], locals()["y_gt_" + cat]
            data["x_" + cat] = x
            data["y_" + cat] = y_gt
        data['scaler'] = scaler
        return data

