import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config

config.update("jax_enable_x64", True)
import pickle
import time

from scipy import signal

from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator
from simphony.time_domain.simulation import TimeResult, TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

data = np.load("X_mmi_binary_large.npz")
X_re    = data["X_re"]
X_im    = data["X_im"]
y = data["labels"]
y_data = np.asarray(y, dtype=float)
X_data    = X_re + 1j * X_im  

T = 10.0e-11
dt = 1e-14
t = jnp.arange(0,T,dt)
netlist = {
    "instances":{
        "pm1": "phase_modulator1",
        "pm2": "phase_modulator2",
        "pm0": "phase_modulator0",
        "wg1": "waveguide",
        "wg2": "waveguide",
        "y1": "y_branch",
        "y2": "y_branch",

},
"connections":{
    "y1,port_2":"wg1,o0",
    "y1,port_3":"wg2,o0",
    "wg1,o1": "pm1,o0",
    "wg2,o1": "pm0,o0",
    "pm1,o1": "y2,port_2",
    "pm0,o1": "y2,port_3",
    "y2,port_1": "pm2,o0",

},
"ports":{
    "o0": "y1,port_1",
    "o1": "pm2,o1",

},
}

def phase_mod(amp):
    return 2*np.arccos(amp)
phase_1 = phase_mod(np.abs(weights_pos[0]/15))
phase_2 = -phase_1/2+np.angle(weights_pos[0])
phase_mod1 = Modulator(mod_signal=phase_1*jnp.ones_like(t))
phase_mod0 = Modulator(mod_signal = 0*jnp.ones_like(t))
phase_mod2 = Modulator(mod_signal=phase_2*jnp.ones_like(t))
models = {
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
}
# wavelengths = [1.548,1.549,1.55,1.551,1.552]
# for wl in wavelengths:
models["phase_modulator1"] = phase_mod1
models["phase_modulator0"] = phase_mod0
models["phase_modulator2"] = phase_mod2
wvl = np.linspace(1.50,1.60,200)
options = {"wl":wvl, "wg1":{"length":10.0},"wg2":{"length":10.0},}
time_sim = TimeSim(netlist=netlist, models=models, settings = options)
inputs = {
    "o0": X_data[:10000,0],
    "o1": jnp.zeros_like(t),
}
c = 299792458.0
results = time_sim.run(t,inputs, carrier_freq=c/(1.548*1e-6), dt=dt)
