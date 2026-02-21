import torch
import torch.nn as nn
import numpy as np
import joblib

ml_model = joblib.load("model/ml_model.pkl")
scaler = joblib.load("model/scaler.pkl")

class HeartNet(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x)

dl_model = HeartNet(13)
dl_model.load_state_dict(torch.load("model/dl_model.pth"))
dl_model.eval()

def hybrid_predict(features):

    arr = np.array(features).reshape(1,-1)
    arr_scaled = scaler.transform(arr)

    ml_pred = ml_model.predict_proba(arr_scaled)[0][1]

    tensor = torch.tensor(arr_scaled,dtype=torch.float32)
    dl_pred = dl_model(tensor).item()

    return (ml_pred + dl_pred)/2
