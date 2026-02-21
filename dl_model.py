import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

os.makedirs("model", exist_ok=True)

data = pd.read_csv("data/heart.csv")

X = data.drop("target",axis=1).values
y = data["target"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler,"model/dl_scaler.pkl")

X = torch.tensor(X,dtype=torch.float32)
y = torch.tensor(y,dtype=torch.float32).view(-1,1)

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

model = HeartNet(X.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(),"model/dl_model.pth")

print("✅ Deep Learning Model Saved")
