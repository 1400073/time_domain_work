import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "simphony-block")))
import simphony
from simphony.time_domain import TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse 
from simphony.libraries import siepic, ideal
from simphony.time_domain.ideal import Modulator,MMI
from simphony.utils import SPEED_OF_LIGHT
import json
from tqdm.auto import tqdm
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import contextlib
import sax

T = 80.0e-11
dt = 1e-14                   # Time step/resolution
t = jnp.arange(0, T, dt)
num_delay = 5
wavelengths = [1.548, 1.549, 1.55,1.551,1.552]

num_measurements = 1000 
wvl = np.linspace(1.5, 1.6, num_measurements)

netlist = {
    "instances":{
        "hr1":"half_ring",
        "hr2":"half_ring",
        "hr3":"half_ring",
        "hr4":"half_ring",
        "wg1":"waveguide",
        "wg2":"waveguide",

    },
    "connections":{
        "hr1,port_1":"hr2,port_1",
        "hr1,port_3":"hr2,port_3",
        # "hr3,port_1":"hr4,port_1",
        # "hr3,port_3":"hr4,port_3",

        # "hr2,port_2":"hr3,port_2",

    },
    "ports":{
        "o0":"hr1,port_2",
        "o1":"hr2,port_2",
        "o2":"hr1,port_4",
        "o3":"hr2,port_4",
        # "o4":"hr3,port_4",
        # "o5":"hr4,port_4",
    },
}

# netlist2 = {
#     "instances":{
#         "dr1":"directional_coupler",
#         "dr2":"directional_coupler",
#         "wg1":"waveguide",
#         "wg2":"waveguide",

#     },
#     "connections":{
#         "dr1,port_1":"wg1,o0",
#         "dr1,port_3":"wg2,o0",
#         "dr2,port_1":"wg1,o1",
#         "dr2,port_3":"wg2,o1",
#     },
#     "ports":{
#         "o0":"dr1,port_2",
#         "o1":"dr2,port_2",
#         "o2":"dr1,port_4",
#         "o3":"dr2,port_4",
#     },
# }
options = {
        'wl': wvl, 
        'dr1':{"coupling_length":45},
        'dr2':{"coupling_length":45},
        'hr1': {"radius": 5.00},
        'hr2': {"radius": 5.00},
        "wg1": {"length":0.00},
        "wg2": {"length":0.00} 
    }
models = {
        "waveguide": siepic.waveguide,
        "directional_coupler": siepic.directional_coupler,
        "y_branch": siepic.y_branch,
        "half_ring": siepic.half_ring,
    }  
circuit, _ = sax.circuit(netlist=netlist, models = models)
s_params_dict = circuit(**options)
# Plot the S-parameters
plt.figure(figsize=(10, 6))
plt.plot(wvl, np.abs(s_params_dict["o0","o1"])**2, label='S01')
plt.show()
# from scipy.signal import find_peaks


# port_pair = ('o0','o2')
# data = np.array(s_params_dict[port_pair])    # complex S(f) vs wavelength
# mag2 = np.abs(data)**2                       # |S|^2

# # Find local maxima in |S|^2 that exceed 0.9
# threshold = 0.95
# peaks, props = find_peaks(mag2, height=threshold)

# peak_wls   = wvl[peaks]
# peak_vals  = mag2[peaks]

# print(f"Resonant peaks for S{port_pair}:")
# for wl, val in zip(peak_wls, peak_vals):
#     print(f"  λ = {wl:.6f} μm  →  |S|^2 = {val:.3f}")

# time_sim = TimeSim(netlist = netlist, models = models, 
#                    settings = options)
# inputs = {
#     "o0": smooth_rectangular_pulse(t, 1.0e-13, T+ 20.0e-11),
#     "o1": jnp.zeros_like(t),
#     "o2": jnp.zeros_like(t),
#     "o3": jnp.zeros_like(t),
# }
# results = time_sim.run(t, inputs,carrier_freq=SPEED_OF_LIGHT/(1.548*1e-6), dt=dt)
# results.plot_sim()