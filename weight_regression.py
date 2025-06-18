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


data = np.load("X_mmi_binary_final.npz")
X_re    = data["X_re"]
X_im    = data["X_im"]
y = data["labels"]
y_data = np.asarray(y, dtype=float)
X_data    = X_re + 1j * X_im  

print(X_data.shape, y_data.shape)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle = True)
# # 4) (optional) delete the handle when you’re done
# data.close()
# with open("time_domain_paper/rings/X_mmi_10x10_mmiresults.json", "r") as f:
#     X_mmi = json.load(f)
# y = X_mmi["labels"]
# y_data = np.asarray(y, dtype=float)
# X_re_list = X_mmi["X_re"]          # list-of-lists (rows)
# X_im_list = X_mmi["X_im"]
# X_re = np.asarray(X_re_list, dtype=float)
# X_im = np.asarray(X_im_list, dtype=float)

# X_data    = X_re + 1j * X_im  

# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle = True)
# Extract pos and neg features
def split_pos_neg(X_raw):
    xpos_list = []
    xneg_list = []
    for i in range(0, 50, 10):
        xpos_list.append(X_raw[:, i:i+5])  # pos: first 5 ports
        xneg_list.append(X_raw[:, i+5:i+10])  # neg: next 5 ports
    xpos = np.concatenate(xpos_list, axis=1)  # shape: (N, 25)
    xneg = np.concatenate(xneg_list, axis=1)  # shape: (N, 25)
    return xpos, xneg

# Split features
xpos_train, xneg_train = split_pos_neg(X_train)
xpos_test, xneg_test = split_pos_neg(X_test)

# Convert to torch tensors (optional: dtype=torch.cfloat if using complex)
xpos_train_torch = torch.tensor(xpos_train, dtype=torch.cfloat)
xneg_train_torch = torch.tensor(xneg_train, dtype=torch.cfloat)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)

xpos_test_torch = torch.tensor(xpos_test, dtype=torch.cfloat)
xneg_test_torch = torch.tensor(xneg_test, dtype=torch.cfloat)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)
# ----- 4. Define the Model -----

class InterferometricRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # Learn one unconstrained complex weight per input channel for “pos” and “neg.”
        self.wpos = nn.Parameter(torch.randn(in_dim, dtype=torch.cfloat))
        self.wneg = nn.Parameter(torch.randn(in_dim, dtype=torch.cfloat))
        self.bias  = nn.Parameter(torch.tensor(0.0))  # optional scaling factor


    def forward(self, xpos, xneg):
        # xpos, xneg: each has shape (batch_size, in_dim) and dtype=torch.cfloat

        # 1) “Interferometric” readout
        pos = torch.sum(self.wpos * xpos, dim=1)  # shape = (batch_size,), complex
        neg = torch.sum(self.wneg * xneg, dim=1)  # shape = (batch_size,), complex

        # 2) Intensity‐difference
        y_pred = pos.abs()**2 - neg.abs()**2       # shape = (batch_size,), real

        return y_pred + self.bias  # add bias term for flexibility

# ----- 5. Instantiate model, loss, optimizer -----
model = InterferometricRegressor(in_dim=25)
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,            # initial step size
    max_iter=30,       # how many internal steps per call
    history_size=10,
    line_search_fn='strong_wolfe' 
)

# 6. Define the closure for LBFGS
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
        # add eps inside the sqrt for numerical stability
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)

loss_fn = nn.MSELoss()
def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# ----- 6. Training Loop -----
n_epochs = 500
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(xpos_train_torch, xneg_train_torch)
    loss = optimizer.step(closure) 
    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(xpos_test_torch, xneg_test_torch)
        test_loss = loss_fn(test_pred, y_test_torch)

        r2_train = r2_score(y_train_torch, y_pred).item()
        r2_test = r2_score(y_test_torch, test_pred).item()

    print(f"Epoch {epoch+1}: "
          f"Train Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}, "
          f"Train R² = {r2_train:.4f}, Test R² = {r2_test:.4f}")
# after training…
with torch.no_grad():
    w_pos = model.wpos 
    w_neg = model.wneg
    bias = model.bias

print(w_pos.tolist())
print(w_neg.tolist())
print(bias.item())
