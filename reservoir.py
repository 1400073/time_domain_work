import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)


from simphony.time_domain import TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse 
from simphony.libraries import siepic, ideal
from simphony.time_domain.ideal import Modulator,MMI

import json
from tqdm.auto import tqdm
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

## Lorenz-63 parameters
sigma, rho, beta = 10.0, 28.0, 8/3

def lorenz(t, xyz):
    x, y, z = xyz
    return [sigma*(y - x),
            x*(rho - z) - y,
            x*y - beta*z]

# --- integration settings ---------------------------------
dt          = 0.05                   #   Δt used in the paper
n_train     = 400                   #   per variable
n_test      = 601
n_warmup    = 100                    #   let the attractor settle first
total_steps = n_warmup + n_train + n_test   # = 1100
t_eval      = np.linspace(0,
                          dt*total_steps,
                          total_steps)
sol = solve_ivp(lorenz,
                (0, t_eval[-1]),
                (0., 1., 1.05),          # same IC as the paper
                t_eval=t_eval,
                method="RK45")
xyz = sol.y.T 
x, y, z = xyz.T                  # shape (1101, 3)
x   = x[n_warmup:]
y_  = y[n_warmup:]
z_  = z[n_warmup:]

def minmax(u):
    return 2 *(u - u.min())/(u.max() - u.min()) - 1

N_coarse  = 1001 
t_coarse  = np.arange(N_coarse) * dt
upsample  = 80                  # 1→10
dt_fine   = dt/ upsample # 0.005
N_fine    = N_coarse * upsample  # 10 000

t_fine = np.linspace(0.0, t_coarse[-1], N_fine) 
# interp = interp1d(t_coarse, x, kind='linear')   # make a 1-D interpolator
# x_f = interp(t_fine)                              # length 10 000
# y_f = interp1d(t_coarse, y_)(t_fine)
# z_f = interp1d(t_coarse, z_)(t_fine)
cs = CubicSpline(t_coarse, x)
x_f = cs(t_fine)                              # length 10 000
y_f = CubicSpline(t_coarse, y_)(t_fine)
z_f = CubicSpline(t_coarse, z_)(t_fine)

signal = minmax(x_f)
signal2 = minmax(y_f)
y = minmax(z_f)

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

        "ytest1": "y_branch",
        "ytest2": "y_branch"

    },
    "connections":{

        

        # "mmi,o0":"wg1,o1",
        # "mmi,o1":"wg2,o1",
        # "mmi,o2":"wg3,o1",
        # "mmi,o3":"wg4,o1",
        # "mmi,o4":"wg5,o1",
        # "mmi,o5":"wg6,o1",
        # "mmi,o6":"wg7,o1",
        # "mmi,o7":"wg8,o1",
        # "mmi,o8":"wg9,o1",

        
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
        "yb7,port_2":"wg8,o0",
        "yb7,port_3":"yb8,port_1",
        "yb8,port_2":"wg9,o0",
        "yb8,port_3":"wg10,o0",


    },
    "ports":{
        "o0":"yinput,port_2",
        "o1":"yinput,port_3",
        "o2":"wg2,o0",
        "o3":"wg3,o0",
        "o4":"wg4,o0",
        "o5":"wg5,o0",
        "o6":"wg6,o0",
        "o7":"wg7,o0",
        "o8":"wg8,o0",
        "o9":"wg9,o0",
        "o10":"wg10,o0",
        # "o2":"wg1,o0",


        # "o3":"mmi,o10",
        # "o4":"mmi,o11",
        # "o5":"mmi,o12",
        # "o6":"mmi,o13",
        # "o7":"mmi,o14",
        # "o8":"mmi,o15",
        # "o9":"mmi,o16",
        # "o10":"mmi,o17",
        # "o11":"mmi,o18",
        # "o12":"mmi,o19",


    },
}

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
num_delay = 8
group_delay = {}
group_delay_ts = {}
for i in range(3,11):
    group_delay[f"{i}"] = []
    group_delay_ts[f"{i}"] = []


delay_base = 11
prev = 2
for k in range(3,11):
    for i in range(0, num_delay//8*(k-2)):
        netlist = {}
        netlist["instances"] = {}
        netlist["connections"] = {}
        netlist["ports"] = {}
        for j in range(2, 10):
            netlist["instances"][f"wg{j}"] = "waveguide"
        netlist["ports"]["o0"] = "wg2,o0"
        netlist["ports"]["o1"] = "wg9,o1"

        for j in range(2, 9):
            netlist["connections"][f"wg{j},o1"] = f"wg{j+1},o0"
        group_delay[f"{k}"].append(netlist)
 
# MultiModeInterferometer = MMI(r=10, s=10)
MultiModeInterferometer = ideal.make_mmi_model(r = 10, s = 10)

models = {
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
    "MultiModeInterferometer": MultiModeInterferometer,
}

for key,val in group_delay.items():
    for netlist in val:
        sim = TimeSim(netlist = netlist, models = models, settings = options)
        group_delay_ts[key].append(sim)

split_sim = TimeSim(netlist = splitter_netlist,models=models,settings=options)
mmi_sim = TimeSim(netlist = mmi_netlist, models=models, settings= options)

final_netlist = {
    "instances":{},
    "connections":{},
    "ports":{},
}
final_netlist["instances"]["split_sim"] = "split_sim"
models["split_sim"] = split_sim
final_netlist["instances"]["mmi_sim"] = "mmi_sim"
models["mmi_sim"] = mmi_sim
final_netlist["ports"]["o2"] = "mmi_sim,o0"
final_netlist["ports"]["o0"] = "split_sim,o0"
final_netlist["ports"]["o1"] = "split_sim,o1"
final_netlist["connections"]["mmi_sim,o1"] ="split_sim,o2"
counter = 0
for key,value in group_delay_ts.items():
    keye = int(key)
    final_netlist["connections"][f"split_sim,o{keye}"] = f"time_sim{counter},o0"

    for i, time_sim in enumerate(value[:-1]):
        final_netlist["instances"][f"time_sim{counter}"] = f"time_sim{counter}"
        models[f"time_sim{counter}"] = time_sim
        final_netlist["connections"][f"time_sim{counter},o1"] = f"time_sim{counter+1},o0"
        counter +=1
    final_netlist["instances"][f"time_sim{counter}"] = f"time_sim{counter}"
    models[f"time_sim{counter}"] = value[-1]
    final_netlist["connections"][f"time_sim{counter},o1"] = f"mmi_sim,o{keye-1}"
    counter +=1
for i in range(10,20):
    final_netlist["ports"][f"o{i-7}"] = f"mmi_sim,o{i}"

wavelengths = [1.548, 1.549, 1.55,1.551,1.552]
# wavelengths = [1.55]
port_list       = [p for p in final_netlist["ports"]             # ['o2', 'o3', … 'o6']
                   if p not in ("o0", "o1","o2")]
start_idx = 6000
end_idx   = len(t) 
n_samples = end_idx - start_idx
c = 299792458.0  # speed of light in m/s                          
n_ports_out     = len(port_list)                           # 5
n_wvls          = len(wavelengths)  
final_sim = TimeSim(netlist = final_netlist, models = models, settings= options)                       # 5

# design matrix: rows = time samples, cols = (ports × wavelengths)
X = np.zeros((n_samples, n_ports_out * n_wvls), dtype=np.complex64)
for wl_idx, w in tqdm(enumerate(wavelengths), desc="Processing wavelengths", total=len(wavelengths)):
   
    result = final_sim.run(t, {
    # "o0": impulse,
    "o0":signal[:len(t)],
    # "o0": jnp.zeros_like(t),
    # 'o0': sig_padded,
    'o1': signal2[:len(t)],
    # "o1": jnp.zeros_like(t),
    # 'o1': jnp.zeros_like(t),
    'o2': smooth_rectangular_pulse(t, 0.0, T+ 20.0e-11)*jnp.sqrt(10),
    # 'o2': jnp.zeros_like(t),
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
    

    # List of the ports you actually want:
    port_list = [p for p in final_netlist['ports'] if p not in ('o0','o1','o2')]

    # Pre-allocate a (n_samples × n_ports) array of instantaneous powers
    P = np.zeros((n_samples, n_ports_out), dtype=np.complex64)

    for j, p in enumerate(port_list):
        
        full_ts = np.array(result.outputs[p])
        
        P[:, j] = full_ts[start_idx:end_idx]  
    start = wl_idx * n_ports_out            
    X[:, start:start + n_ports_out] = P



# Split into real and imaginary parts (float32 each):
X_re = np.real(X)
X_im = np.imag(X)

# Save to a compressed .npz file:
np.savez_compressed(
    "X_mmi_binary_small.npz",
    X_re=X_re.astype(np.float32),
    X_im=X_im.astype(np.float32),
    labels=y[6075:80075].astype(np.float32),
)