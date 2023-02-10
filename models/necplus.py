import torch 
import  torch.nn as nn
from models.base_model import BaseModel

class NecPlus(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        pass 

    def forward(self, x):
        pass 

class E_L(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, layer_dim=8):
        super(E_L, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=0.4, bidirectional=False, batch_first=True)

    def forward(self, x, s_h, s_c):
        # Initialize hidden and cell state
        h0 = s_h
        c0 = s_c
        out, (hn, cn) = self.lstm(x, (h0,c0))
        return out, hn, cn

class E_F(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=24):
        super(E_F, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim=output_dim
        self.bn1 = nn.BatchNorm1d(num_features=self.hidden_dim)
        self.L_out1 = nn.Linear(self.hidden_dim, 512) 
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.L_out2 = nn.Linear(512, 256) 
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.L_out3 = nn.Linear(256, self.output_dim) 
    def forward(self, x, id=-1):
        linear_out = self.L_out1(self.bn1(x[:,id,:]))
        linear_out = self.L_out2(self.bn2(linear_out))
        linear_out = self.L_out3(self.bn3(linear_out))
        return linear_out
