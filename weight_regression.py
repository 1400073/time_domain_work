import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
import jax
from jax import config
config.update("jax_enable_x64", True)

from simphony.time_domain import TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator

import json
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from jax import grad
import torch.nn as nn
import torch.optim as optim
import torch


data = np.load("X_mmi_binary_5__few_delay_lines.npz")
X_re    = data["X_re"]
X_im    = data["X_im"]
y = data["labels"]
y_data = np.asarray(y, dtype=float)
X_data    = X_re + 1j * X_im  

print(X_data.shape, y_data.shape)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data[:X_data.shape[0]], test_size=0.4, shuffle = True)

# def split_pos_neg(X_raw):
#     xpos_list = []
#     xneg_list = []
#     for i in range(0, 45, 5):
#         xpos_list.append(X_raw[:, i:i+5])  # pos: first 5 ports
#         xneg_list.append(X_raw[:, i+5:i+9])  # neg: next 5 ports
#     xpos = np.concatenate(xpos_list, axis=1)  # shape: (N, 25)
#     xneg = np.concatenate(xneg_list, axis=1)  # shape: (N, 25)
#     return xpos, xneg   
def split_pos_neg(X_raw):
    """
    X_raw: np.ndarray of shape (N, 45)
    We assume 5 blocks of 9 features each:
      - first 5 per block are 'pos'
      - next 4 per block are 'neg'
    Returns:
      xpos: (N, 5*5=25)
      xneg: (N, 5*4=20)
    """
    n_feat = X_raw.shape[1]
    block_len = 9               # 45 total ÷ 5 blocks
    pos_len   = 5
    neg_len   = block_len - pos_len  # =4

    xpos_list = []
    xneg_list = []
    for i in range(0, n_feat, block_len):
        xpos_list.append(X_raw[:, i : i + pos_len])        # 5 cols
        xneg_list.append(X_raw[:, i + pos_len : i + block_len])  # 4 cols

    xpos = np.concatenate(xpos_list, axis=1)   # (N, 25)
    xneg = np.concatenate(xneg_list, axis=1)   # (N, 20)
    return xpos, xneg

xpos_train, xneg_train = split_pos_neg(X_train)
xpos_test, xneg_test = split_pos_neg(X_test)

xpos_train_torch = torch.tensor(xpos_train, dtype=torch.cfloat)
xneg_train_torch = torch.tensor(xneg_train, dtype=torch.cfloat)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)

xpos_test_torch = torch.tensor(xpos_test, dtype=torch.cfloat)
xneg_test_torch = torch.tensor(xneg_test, dtype=torch.cfloat)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)


class InterferometricRegressor(nn.Module):
    def __init__(self, n_pos, n_neg):
        super().__init__()
        self.wpos = nn.Parameter(torch.randn(n_pos, dtype=torch.cfloat))
        self.wneg = nn.Parameter(torch.randn(n_neg, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, xpos, xneg):
        pos = torch.sum(self.wpos * xpos, dim=1)
        neg = torch.sum(self.wneg * xneg, dim=1)
        return pos.abs()**2 - neg.abs()**2 + self.bias

# instantiate with the right dims:
model = InterferometricRegressor(
    n_pos = xpos_train.shape[1],
    n_neg = xneg_train.shape[1]
)
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,            
    max_iter=30,       
    history_size=10,
    line_search_fn='strong_wolfe' 
)

def closure():
    optimizer.zero_grad()
    y_pred = model(xpos_train_torch, xneg_train_torch)
    loss = loss_fn(y_pred, y_train_torch)
    loss.backward()
    return loss


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)

loss_fn = RMSELoss()
def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


n_epochs = 500
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(xpos_train_torch, xneg_train_torch)
    loss = optimizer.step(closure) 

    model.eval()
    with torch.no_grad():
        test_pred = model(xpos_test_torch, xneg_test_torch)
        test_loss = loss_fn(test_pred, y_test_torch)

        r2_train = r2_score(y_train_torch, y_pred).item()
        r2_test = r2_score(y_test_torch, test_pred).item()

    print(f"Epoch {epoch+1}: "
          f"Train Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}, "
          f"Train R² = {r2_train:.4f}, Test R² = {r2_test:.4f}")



with torch.no_grad():
    w_pos = model.wpos 
    w_neg = model.wneg
    bias = model.bias

# with torch.no_grad():
#     w_pos = torch.exp(model.log_amp_pos) * torch.exp(1j*model.phase_pos)
#     w_neg = torch.exp(model.log_amp_neg) * torch.exp(1j*model.phase_neg)
#     bias = model.bias

print(w_pos.tolist())
print(w_neg.tolist())
print(bias.item())