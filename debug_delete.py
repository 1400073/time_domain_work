import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "simphony")))
import simphony
config.update("jax_enable_x64", True)
import pickle
import time

from scipy import signal

from simphony.libraries import siepic,ideal
from simphony.time_domain.ideal import Modulator
from simphony.time_domain.simulation import TimeResult, TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import simphony

splitter_netlist = {
    "instances":{
        # "wg1": "waveguide",
        "wg2": "waveguide",
        "wg3": "waveguide",
        "wg4": "waveguide",
        "wg5": "waveguide",
        "wg6": "waveguide",
        "wg7": "waveguide",
        "wg8": "waveguide",
        "wg9": "waveguide",
        "wg10": "waveguide",

        "yinput":"y_branch",
        "yb1": "y_branch",
        "yb2": "y_branch",
        "yb3": "y_branch",
        "yb4": "y_branch",
        "yb5": "y_branch",
        "yb6": "y_branch",
        "yb7": "y_branch",
        "yb8": "y_branch",
    },
    "connections":{

        "yinput,port_1":"yb1,port_1",
        "yb1,port_2":"yb2,port_1",
        "yb1,port_3":"yb3,port_1",

        "yb2,port_2":"yb4,port_1",
        "yb2,port_3":"yb5,port_1",
        "yb3,port_2":"yb6,port_1",
        "yb3,port_3":"yb7,port_1",

        "yb4,port_2":"wg2,o0",
        "yb4,port_3":"wg3,o0",
        "yb5,port_2":"wg4,o0",
        "yb5,port_3":"wg5,o0",
        "yb6,port_2":"wg6,o0",
        "yb6,port_3":"wg7,o0",

        "yb7,port_2":"yb8,port_1",
        "yb7,port_3":"wg8,o0",
        "yb8,port_2":"wg9,o0",
        "yb8,port_3":"wg10,o0",


    },
    "ports":{
        "o0":"yinput,port_2",
        "o1":"yinput,port_3",
        "o2":"wg2,o1",
        "o3":"wg3,o1",
        "o4":"wg4,o1",
        "o5":"wg5,o1",
        "o6":"wg6,o1",
        "o7":"wg7,o1",
        "o8":"wg8,o1",
        "o9":"wg9,o1",
        "o10":"wg10,o1",


    },
}
T = 80e-11
dt = 1e-14                   # Time step/resolution
t = jnp.arange(0, T, dt)
num_measurements = 200
wvl = np.linspace(1.5, 1.6, num_measurements)
options = {
    'wl': wvl,
    'wg1': {'length': 10.0},
    'wg2': {'length': 50.0},
    'wg3': {'length': 50.0},
    'wg4': {'length': 50.0},
    'wg5': {'length': 50.0},
    'wg6': {'length': 50.0},
    'wg7': {'length': 50.0},
    'wg8': {'length': 50.0},
    'wg9': {'length': 50.0},
    'wg10': {'length': 50.0},
    
}
# MultiModeInterferometer = MMI(r=10, s=10)
MultiModeInterferometer = ideal.make_mmi_model(r = 10, s = 10)
models = {
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
    "MultiModeInterferometer": MultiModeInterferometer,
}
c = 299792458.0
final_sim = TimeSim(netlist = splitter_netlist, models = models, settings= options) 
result = final_sim.run(t, {
    "o0":smooth_rectangular_pulse(t, 0.0, T+ 20.0e-11),
    'o1': smooth_rectangular_pulse(t, 0.0, T+ 20.0e-11),
    'o2': jnp.zeros_like(t),
    'o3': jnp.zeros_like(t),
    'o4': jnp.zeros_like(t),
    'o5': jnp.zeros_like(t),
    'o6': jnp.zeros_like(t),
    'o7': jnp.zeros_like(t),
    'o8': jnp.zeros_like(t),
    'o9': jnp.zeros_like(t),
    'o10': jnp.zeros_like(t),
    }, carrier_freq=c/(1.55*1e-6), dt=dt)

result.plot_sim()
