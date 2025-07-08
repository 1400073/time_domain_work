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


T = 80.0e-11
dt = 1e-14                   # Time step/resolution
t = jnp.arange(0, T, dt)
num_delay = 5
wavelengths = [1.548, 1.549, 1.55,1.551,1.552]

start_idx = 5000
end_idx   = len(t) 
n_samples = end_idx - start_idx
c = 299792458.0                          
n_ports_out     = 4                           
n_wvls          = len(wavelengths) 

X = np.zeros((n_samples, n_ports_out * n_wvls), dtype=np.complex64)
num_measurements = 200
wvl = np.linspace(1.5, 1.6, num_measurements)

for wl_idx, w in tqdm(enumerate(wavelengths), desc="Processing wavelengths", total=len(wavelengths)):
    sigma, rho, beta = 10.0, 28.0, 8/3

    def lorenz(t, xyz):
        x, y, z = xyz
        return [
            sigma*(y - x),
            x*(rho - z) - y,
            x*y - beta*z,
        ]

    dtt        = 0.05
    n_train   = 400
    n_test    = 600
    n_warmup  = 100
    total     = n_warmup + n_train + n_test   # 1101
    t_eval    = np.linspace(0, dtt*total, total)

    sol = solve_ivp(lorenz, (0, t_eval[-1]), (0.,1.,1.05),
                    t_eval=t_eval, method="RK45")
    xyz = sol.y.T
    x, y_, z_ = xyz.T

    # normalize to [−1,1]
    def minmax(u):
        return 2*(u - u.min())/(u.max() - u.min()) - 1

    # coarse → fine upsampling
    N_coarse = 1000
    t_coarse = np.arange(N_coarse)*dtt
    upsample = 100
    dt_fine  = dtt/upsample
    t_fine   = np.linspace(0, t_coarse[-1], N_coarse*upsample)

    cs = CubicSpline(t_coarse, x[:N_coarse])
    x_f = cs(t_fine)
    cs = CubicSpline(t_coarse, y_[:N_coarse])
    y_f = cs(t_fine)
    cs = CubicSpline(t_coarse, z_[:N_coarse])
    z_f = cs(t_fine)

    # base signals
    signal  = minmax(x_f)
    signal2 = minmax(y_f)
    y       = minmax(z_f)

    network_splitter = {
        "instances":{
            "y1":"y_branch",
            "y2":"y_branch",
            "y3":"y_branch",
            "y4":"y_branch",
        },
        "connections":{
            "y1,port_1":"y2,port_1",
            "y2,port_2":"y3,port_1",
            "y2,port_3":"y4,port_1",
        },
        "ports":{
            "o0":"y1,port_2",
            "o1":"y1,port_3",
            "o2":"y3,port_2",
            "o3":"y3,port_3",
            "o4":"y4,port_2",
            "o5":"y4,port_3",
        },
    }
    model_settings_split = {
        'wl': wvl,
    }
    models_split = {
        "y_branch": siepic.y_branch,
    }

    network_split = TimeSim(netlist = network_splitter,models = models_split, settings = model_settings_split)

    network1 = {
        "instances":{
            "hr1":"half_ring",
            "hr2":"half_ring",

        },
        "connections":{
            "hr1,port_1":"hr2,port_1",
            "hr1,port_3":"hr2,port_3",

        },
        "ports":{
            "o0":"hr1,port_2",
            "o1":"hr2,port_2",
            "o2":"hr1,port_4",
            "o3":"hr2,port_4",
        },
    }

    
    time_sim_list = []
    models_hr = {
        "half_ring": siepic.half_ring,
    }
    model_settings = {
        'wl': wvl,
    }

    for i in range(0,1):
        time_sim_list.append(
            TimeSim(
                netlist = network1,
                models = models_hr,
                settings = model_settings,
            )
        )
    final_netlist = {
        "instances":{
            "splitter":"network_splitter",
            "wg1":"waveguide",
        },
        "connections":{

        },
        "ports":{

        },
    }

    final_models = {
        "network_splitter": network_split,
        "waveguide": siepic.waveguide,
    }

    for i,time_sim in enumerate(time_sim_list):
        final_netlist["instances"][f"hr_time_sim{i}"] = f"hr_time_sim{i}"
        final_models[f"hr_time_sim{i}"] = time_sim

    # final_netlist["connections"]["splitter,o0"] = "hr_time_sim0,o0"
    # final_netlist["connections"]["splitter,o1"] = "hr_time_sim1,o1"
    # final_netlist["connections"]["hr_time_sim0,o1"] = "wg1,o0"
    # final_netlist["connections"]["hr_time_sim1,o0"] = "wg1,o1"

    # final_netlist["connections"]["hr_time_sim0,o2"] = "hr_time_sim2,o0"
    # final_netlist["connections"]["hr_time_sim0,o3"] = "hr_time_sim3,o0"
    # final_netlist["connections"]["hr_time_sim1,o3"] = "hr_time_sim2,o1"
    # final_netlist["connections"]["hr_time_sim1,o3"] = "hr_time_sim4,o1"

    # final_netlist["connections"]["hr_time_sim2,o2"] = "hr_time_sim3,o1"
    # final_netlist["connections"]["hr_time_sim2,o3"] = "hr_time_sim4,o0"

    # final_netlist["ports"]["o0"] = "splitter,o2"
    # final_netlist["ports"]["o1"] = "splitter,o3"
    # final_netlist["ports"]["o2"] = "splitter,o4"
    # final_netlist["ports"]["o3"] = "splitter,o5"
    # final_netlist["ports"]["o4"] = "hr_time_sim3,o2"
    # final_netlist["ports"]["o5"] = "hr_time_sim3,o3"
    # final_netlist["ports"]["o6"] = "hr_time_sim4,o2"
    # final_netlist["ports"]["o7"] = "hr_time_sim4,o3"


    

    final_time_sim = TimeSim(netlist = final_netlist, models = final_models, settings = model_settings)

    with open(os.devnull, 'w') as fnull, \
        contextlib.redirect_stdout(fnull), \
        contextlib.redirect_stderr(fnull):   
            result = final_time_sim.run(t, {
            "o0": signal[1000:len(t)+1000],
            # "o1": jnp.zeros_like(t),
            'o1': signal2[1000:len(t)+1000],
            # 'o2': jnp.zeros_like(t),
            # 'o3': jnp.zeros_like(t),
            'o2': smooth_rectangular_pulse(t, 1.0e-13, T+ 20.0e-11)*jnp.sqrt(10),
            'o3': smooth_rectangular_pulse(t, 1.0e-13, T+ 20.0e-11)*jnp.sqrt(10),
            'o4': jnp.zeros_like(t),
            'o5': jnp.zeros_like(t),
            'o6': jnp.zeros_like(t),
            'o7': jnp.zeros_like(t),
            # 'o8': jnp.zeros_like(t),
            # 'o9': jnp.zeros_like(t),
            # 'o10': jnp.zeros_like(t),
            # 'o11': jnp.zeros_like(t),
            # 'o12': jnp.zeros_like(t),
            # 'o13': jnp.zeros_like(t),
            # 'o14': jnp.zeros_like(t),
            # 'o15': jnp.zeros_like(t),
            # 'o16': jnp.zeros_like(t),
            # 'o17': jnp.zeros_like(t),
            # 'o18': jnp.zeros_like(t),
            # 'o19': jnp.zeros_like(t),
            }, carrier_freq=SPEED_OF_LIGHT/(w*1e-6), dt=dt)

    outputs = result.outputs
    result.plot_sim()
    port_list = [p for p in final_netlist['ports'] if p not in ('o0','o1','o2','o3')]
    P = np.zeros((n_samples, n_ports_out), dtype=np.complex64)

    for j, p in enumerate(port_list):
        full_ts = np.array(result.outputs[p])
        P[:, j] = full_ts[start_idx:end_idx]  

    start = wl_idx * n_ports_out            
    X[:, start:start + n_ports_out] = P
    
X_re = np.real(X)
X_im = np.imag(X)

np.savez_compressed(
    "X_simple_reservoir_binary.npz",
    X_re=X_re.astype(np.float32),
    X_im=X_im.astype(np.float32),
    labels=y[start_idx:].astype(np.float32),
)


