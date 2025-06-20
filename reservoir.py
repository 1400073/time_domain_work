import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "simphony")))
import simphony
from simphony.time_domain import TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse 
from simphony.libraries import siepic, ideal
from simphony.time_domain.ideal import Modulator,MMI

import json
from tqdm.auto import tqdm
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

# --- Lorenz‐63 and integration setup --------------------------------
sigma, rho, beta = 10.0, 28.0, 8/3

def lorenz(t, xyz):
    x, y, z = xyz
    return [
        sigma*(y - x),
        x*(rho - z) - y,
        x*y - beta*z,
    ]

dt        = 0.05
n_train   = 400
n_test    = 600
n_warmup  = 100
total     = n_warmup + n_train + n_test   # 1101
t_eval    = np.linspace(0, dt*total, total)

# solve
sol = solve_ivp(lorenz, (0, t_eval[-1]), (0.,1.,1.05),
                t_eval=t_eval, method="RK45")
xyz = sol.y.T
x, y_, z_ = xyz.T

# normalize to [−1,1]
def minmax(u):
    return 2*(u - u.min())/(u.max() - u.min()) - 1

# coarse → fine upsampling
N_coarse = 1000
t_coarse = np.arange(N_coarse)*dt
upsample = 80
dt_fine  = dt/upsample
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

# --- Smooth half‐cosine ramp within the same time span --------------
ramp_time    = 0.5                        # ramp duration
ramp_samples = int(ramp_time / dt_fine)   # number of fine‐samples

def _add_smooth_ramp(u: np.ndarray, N: int) -> np.ndarray:
    # half‐cosine from 0→1 over N points, then scale by u[0]
    θ = np.linspace(0, np.pi, N)
    ramp = 0.5 * (1 - np.cos(θ)) * u[0]
    # replace the first N samples of u with the ramp
    return np.concatenate([ramp, u[N:]])

signal  = _add_smooth_ramp(signal,  ramp_samples)
signal2 = _add_smooth_ramp(signal2, ramp_samples)
y       = _add_smooth_ramp(y,       ramp_samples)

T = 80e-11
dt = 1e-14                   # Time step/resolution
t = jnp.arange(0, T, dt)

wavelengths = [1.548, 1.549, 1.55,1.551,1.552]

start_idx = 5000
end_idx   = len(t) 
n_samples = end_idx - start_idx
c = 299792458.0                          
n_ports_out     = 10                           
n_wvls          = len(wavelengths)  
                    # 5
# design matrix: rows = time samples, cols = (ports × wavelengths)
X = np.zeros((n_samples, n_ports_out * n_wvls), dtype=np.complex64)
for wl_idx, w in tqdm(enumerate(wavelengths), desc="Processing wavelengths", total=len(wavelengths)):
    
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
    final_sim_split = TimeSim(netlist = splitter_netlist, models = models, settings= options) 

    mmi_netlist = {
        "instances":{
            "mmi": "MultiModeInterferometer",
            "wg1": "waveguide",

        },
        "connections":{
            "mmi,o0":"wg1,o1",

        },
        "ports":{
            
            "o0":"wg1,o0",
            "o1":"mmi,o1",
            "o2":"mmi,o2",
            "o3":"mmi,o3",
            "o4":"mmi,o4",
            "o5":"mmi,o5",
            "o6":"mmi,o6",
            "o7":"mmi,o7",
            "o8":"mmi,o8",
            "o9":"mmi,o9",
            
            "o10":"mmi,o10",
            "o11":"mmi,o11",
            "o12":"mmi,o12",
            "o13":"mmi,o13",
            "o14":"mmi,o14",
            "o15":"mmi,o15",
            "o16":"mmi,o16",
            "o17":"mmi,o17",
            "o18":"mmi,o18",
            "o19":"mmi,o19",

        },
    }
    MultiModeInterferometer = ideal.make_mmi_model(r = 10, s = 10)

    num_delay = 20

    T =80e-11
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
    final_sim_groups = []
    for k in range(0,9):
        group_delay = []
        group_delay_ts = []
        for i in range(0, num_delay//5*(k+1)):
            netlist = {}
            netlist["instances"] = {}
            netlist["connections"] = {}
            netlist["ports"] = {}
            for j in range(2, 7):
                netlist["instances"][f"wg{j}"] = "waveguide"
            netlist["ports"]["o0"] = "wg2,o0"
            netlist["ports"]["o1"] = "wg6,o1"

            for j in range(2, 6):
                netlist["connections"][f"wg{j},o1"] = f"wg{j+1},o0"
            group_delay.append(netlist)

        models = {
            "waveguide": siepic.waveguide,
            "y_branch": siepic.y_branch,
        }

        for netlist in group_delay:
            sim = TimeSim(netlist = netlist, models = models, settings = options)
            group_delay_ts.append(sim)

        final_netlist = {
            "instances":{},
            "connections":{},
            "ports":{},
        }

        counter = 0
        final_netlist["ports"]["o0"] = f"time_sim{counter},o0"

        for i, time_sim in enumerate(group_delay_ts[:-1]):
            final_netlist["instances"][f"time_sim{counter}"] = f"time_sim{counter}"
            models[f"time_sim{counter}"] = time_sim
            final_netlist["connections"][f"time_sim{counter},o1"] = f"time_sim{counter+1},o0"
            counter +=1
        models[f"time_sim{counter}"] = group_delay_ts[-1]
        final_netlist["instances"][f"time_sim{counter}"] = f"time_sim{counter}"
        final_netlist["ports"]["o1"] = f"time_sim{counter},o1"
        l = 0
        final_sim = TimeSim(netlist= final_netlist, models= models, settings=options)
        final_sim_groups.append(final_sim)

    models = {
        "waveguide": siepic.waveguide,
        "y_branch": siepic.y_branch,
        "MultiModeInterferometer": MultiModeInterferometer,
    }
    mmi_sim = TimeSim(netlist = mmi_netlist, models=models, settings= options)

    combined_netlist = {
        "instances":{},
        "connections":{},
        "ports":{},
    }
    models_combined = {
        "final_sim_split1":final_sim_split,
        "mmi_simmer":mmi_sim,
    }
    for i,time_sim in enumerate(final_sim_groups):
        combined_netlist["instances"][f"final_sim{i}"] = f"final_sim{i}"
        models_combined[f"final_sim{i}"] = time_sim

    combined_netlist["instances"]["final_sim_split"] = "final_sim_split1"
    combined_netlist["instances"]["mmi_simmer"] = "mmi_simmer"

    combined_netlist["connections"]["final_sim0,o0"] = "final_sim_split,o2"
    combined_netlist["connections"]["final_sim1,o0"] = "final_sim_split,o3"
    combined_netlist["connections"]["final_sim2,o0"] = "final_sim_split,o4"
    combined_netlist["connections"]["final_sim3,o0"] = "final_sim_split,o5"
    combined_netlist["connections"]["final_sim4,o0"] = "final_sim_split,o6"
    combined_netlist["connections"]["final_sim5,o0"] = "final_sim_split,o7"
    combined_netlist["connections"]["final_sim6,o0"] = "final_sim_split,o8"
    combined_netlist["connections"]["final_sim7,o0"] = "final_sim_split,o9"
    combined_netlist["connections"]["final_sim8,o0"] = "final_sim_split,o10"

    combined_netlist["connections"]["final_sim0,o1"] = "mmi_simmer,o1"
    combined_netlist["connections"]["final_sim1,o1"] = "mmi_simmer,o2"
    combined_netlist["connections"]["final_sim2,o1"] = "mmi_simmer,o3"
    combined_netlist["connections"]["final_sim3,o1"] = "mmi_simmer,o4"
    combined_netlist["connections"]["final_sim4,o1"] = "mmi_simmer,o5"
    combined_netlist["connections"]["final_sim5,o1"] = "mmi_simmer,o6"
    combined_netlist["connections"]["final_sim6,o1"] = "mmi_simmer,o7"
    combined_netlist["connections"]["final_sim7,o1"] = "mmi_simmer,o8"
    combined_netlist["connections"]["final_sim8,o1"] = "mmi_simmer,o9"

    combined_netlist["ports"]["o0"] = "final_sim_split,o0"
    combined_netlist["ports"]["o1"] = "final_sim_split,o1"
    for i in range(10,20):
        combined_netlist["ports"][f"o{i-7}"] = f"mmi_simmer,o{i}"
    combined_netlist["ports"]["o2"] = "mmi_simmer,o0"
    final_sim = TimeSim(netlist = combined_netlist, models = models_combined, settings= options)   
    result = final_sim.run(t, {
    "o0":signal[:len(t)],
    'o1': signal2[:len(t)],
    'o2': smooth_rectangular_pulse(t, 0.0, T+ 20.0e-11)*jnp.sqrt(10),
    'o3': jnp.zeros_like(t),
    'o4': jnp.zeros_like(t),
    'o5': jnp.zeros_like(t),
    'o6': jnp.zeros_like(t),
    'o7': jnp.zeros_like(t),
    'o8': jnp.zeros_like(t),
    'o9': jnp.zeros_like(t),
    'o10': jnp.zeros_like(t),
    'o11': jnp.zeros_like(t),
    'o12': jnp.zeros_like(t),
    }, carrier_freq=c/(w*1e-6), dt=dt)

    outputs = result.outputs
    

    port_list = [p for p in combined_netlist['ports'] if p not in ('o0','o1','o2')]

    P = np.zeros((n_samples, n_ports_out), dtype=np.complex64)

    for j, p in enumerate(port_list):
        
        full_ts = np.array(result.outputs[p])
        
        P[:, j] = full_ts[start_idx:end_idx]  
    start = wl_idx * n_ports_out            
    X[:, start:start + n_ports_out] = P


X_re = np.real(X)
X_im = np.imag(X)

np.savez_compressed(
    "X_mmi_binary_10.npz",
    X_re=X_re.astype(np.float32),
    X_im=X_im.astype(np.float32),
    labels=y[5000:].astype(np.float32),
)