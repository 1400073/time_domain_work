import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import config
import pandas
config.update("jax_enable_x64", True)
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "simphony")))
from simphony.time_domain.simulation import TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse 
from simphony.libraries import siepic, ideal
from simphony.time_domain.ideal import Modulator,MMI
from simphony.utils import SPEED_OF_LIGHT
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import time
from scipy.stats import pearsonr
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
wave_num = 3
num_measurements = 2000
model_order = 50
wvl = np.linspace(1.5, 1.6, num_measurements)
num_delay = 100
wavelengths = [
    1.5460730365182591,
    1.548624312156078,
    1.5512256128064033,
    1.5538269134567284,
    1.5564282141070536,
]
STEPS_PER_SLOT = 15000          
RAMP_LEN       = 5000 
freqmod = 0.1e9
A = jnp.pi
# mywavelengths = wavelengths[rank::size]
# for z in mywavelengths:

sigma, rho, beta = 10.0, 28.0, 8/3
COARSE_SAMPLES     = 1000          # 1000 Lorenz points
MASKS_PER_SAMPLE   = 50            # virtual nodes
dt                 = 1.0e-14       # simulator step [s]
a                  = 0.5         # mask amplitude

def lorenz(t, xyz):
    x, y, z = xyz
    return [sigma*(y - x),
            x*(rho - z) - y,
            x*y - beta*z]

coarse_t = np.linspace(0.0, 0.05 * (COARSE_SAMPLES-1), COARSE_SAMPLES)  # arbitrary 0.05 s spacing
sol      = solve_ivp(lorenz, (0, coarse_t[-1]), (0., 1., 1.05),
                    t_eval=coarse_t, method="RK45")
x_c, y_c, z_c = sol.y

def minmax(u):
    return 2*(u - u.min())/(u.max() - u.min()) - 1

x_c, y_c, z_c = map(minmax, (x_c, y_c, z_c))          # scale to [-1,1] 

            
SLOTS          = COARSE_SAMPLES * MASKS_PER_SAMPLE
TOTAL_STEPS    = SLOTS * STEPS_PER_SLOT


np.random.seed(42)
full_mask_slots = np.tile(
    np.random.uniform(-a, a, MASKS_PER_SAMPLE), COARSE_SAMPLES
)                                             
hold_len = STEPS_PER_SLOT - RAMP_LEN
mask_timed = np.empty(TOTAL_STEPS)

for s in range(SLOTS-1):

    v0 = full_mask_slots[s]+ x_c[s//50] + y_c[s//50]+1
    v1 = full_mask_slots[s+1]+ x_c[s//50] + y_c[s//50]+1
    base = s * STEPS_PER_SLOT

    mask_timed[base : base+hold_len] = v0 
    u = np.linspace(0, 1, RAMP_LEN, endpoint=False)
    ramp = v0 + (v1-v0)*0.5*(1 - np.cos(np.pi*u))

    mask_timed[base+hold_len : base+STEPS_PER_SLOT] = ramp

ramp0 = (mask_timed[0])*0.5*(1 - np.cos(np.pi*u))
mask_timed = np.concatenate((ramp0, mask_timed))
mask_timed[(SLOTS-1)*STEPS_PER_SLOT :] = full_mask_slots[-1]

signal = mask_timed
T =100000
t = jnp.arange(T)*dt
mod_signal = jnp.sin(2 * jnp.pi * freqmod * t)*A
# mod_signal = t * 0.0
timePhaseInstantiated = Modulator(mod_signal=mod_signal)


wg_netlist = {
    "instances":{
        "wg":"waveguide",
    },
    "connections":{
    },
    "ports":{
        "o0":"wg,o0",
        "o1":"wg,o1",
    }
}
hr_netlist = {
    "instances":{
        "hr":"half_ring"
    },
    "connections":{

    },
    "ports":{
        "o0":"hr,port_1",
        "o1":"hr,port_2",
        "o2":"hr,port_3",
        "o3":"hr,port_4",
        }
}

models = {
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
    "bidirectional": siepic.bidirectional_coupler,
    "phase_modulator": timePhaseInstantiated,
    "half_ring": siepic.half_ring,
}

options = {'wl': wvl, 'wg': {"length": 100.0}, 'wg2': {"length": 100.0},'wg3': {"length": 150.0}}
wg_sim = TimeSim(
    netlist=wg_netlist,
    models=models,
    settings=options,
)

hr_sim = TimeSim(
    netlist=hr_netlist,
    models=models,
    settings=options,
)

group_delay = []
group_delay_ts = []
for i in range(0, num_delay):
    netlist = {}
    netlist["instances"] = {}
    netlist["connections"] = {} 
    netlist["ports"] = {}
    
    netlist["instances"]["wg3"] = "waveguide"
    netlist["ports"]["o0"] = "wg3,o0"
    netlist["ports"]["o1"] = "wg3,o1"
    group_delay.append(netlist)


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
final_sim = TimeSim(
    netlist= final_netlist, 
    models= models, 
    settings=options)

models  = {}
models["final_sim"] = final_sim
models["waveguide1"] = wg_sim
models["half_ring1"] = hr_sim
models["phase_modulator"] = timePhaseInstantiated

reservoir_netlist = {
    "instances": {
        "wg_sim1": "waveguide1",
        "wg_sim2": "waveguide1",
        "hr_sim1": "half_ring1",
        "hr_sim2": "half_ring1",
        "mod1": "phase_modulator",
        "delay": "final_sim",
    },
    "connections":{
        "hr_sim1,o2": "wg_sim1,o0",
        "wg_sim1,o1": "mod1,o0",
        "mod1,o1":"hr_sim2,o2",

        "hr_sim2,o0": "wg_sim2,o0", 
        "wg_sim2,o1": "hr_sim1,o0",

        "hr_sim1,o3": "delay,o0",
        "delay,o1": "hr_sim2,o3",
    },
    "ports": {
        "o0": "hr_sim1,o1",
        "o1": "hr_sim2,o1",
    },
}
# reservoir_netlist = {
#    "instances": {
#         "wg_sim1": "waveguide1",
#         "wg_sim2": "waveguide1",
#         "hr_sim1": "half_ring1",
#         "hr_sim2": "half_ring1",
#         "mod1": "phase_modulator",
#     },
#     "connections":{
#         "hr_sim1,o2": "wg_sim1,o0",
#         "wg_sim1,o1": "mod1,o0",
#         "mod1,o1":"hr_sim2,o2",

#         "hr_sim2,o0": "wg_sim2,o0", 
#         "wg_sim2,o1": "hr_sim1,o0",
#     },
#     "ports": {
#         "o0": "hr_sim1,o1",
#         "o1": "hr_sim2,o1",
#         "o2": "hr_sim1,o3",
#         "o3": "hr_sim2,o3",
#     },
# }

time_sim = TimeSim(
    netlist=reservoir_netlist,
    models=models,
    settings=options,
)
#gaussian_pulse(t, 1.0e-11, 0.5e-11)


num_outputs = 4
# inputs = {
#     f'o{i}':  smooth_rectangular_pulse(t, 10.0e-11, T*1e-14 + 10e-11) if i == 0 else jnp.zeros_like(t)
#     for i in range(num_outputs)
# }
inputs = {
    f'o{i}': signal[:T] if i == 0 else jnp.zeros_like(t)
    for i in range(num_outputs)
}
modelResult = time_sim.run(t, inputs,carrier_freq=SPEED_OF_LIGHT/(wavelengths[wave_num]*1.0e-6),dt=dt)
# slice_o1 = modelResult.outputs["o1"][:]     # grab full complex output
# slice_o1.block_until_ready()                # force the compute
# o1_data = np.asarray(slice_o1)              # now a NumPy complex128 array

# output_dir = "simulation_outputs"
# os.makedirs(output_dir, exist_ok=True)

# fn = f"rank{wave_num + 1:02d}_wl{wavelengths[wave_num]:.6f}_o1.npy"
# path = os.path.join(output_dir, fn)

# np.save(path, o1_data)

# print(f"[Rank {wave_num + 1}] saved o1 ({o1_data.shape}) to {path}")
print("now plotting")
# — Pull exactly 10k samples to host in one go —
slice_jax = modelResult.outputs["o0"][:T]
slice_jax1 = modelResult.outputs["o1"][:T]
print("slice_jax", slice_jax.shape)
print("slice_jax1", slice_jax1.shape)
t1 = time.time()
slice_jax.block_until_ready()
t2 = time.time()
print("Time to pull 10k samples:", t2 - t1, "seconds")

raw_o0 = np.asarray(slice_jax)
raw_o1 = np.asarray(slice_jax1)
# — Compute power in NumPy —
o0_host = np.abs(raw_o0)**2
o1_host = np.abs(raw_o1)**2
print("o0_host", o0_host.shape)
print("o1_host", o1_host.shape)

# Compute Pearson correlation coefficient
pearson_corr, _ = pearsonr(o1_host[20000:], signal[:T-20000])
print("Pearson Correlation:", pearson_corr)
print("amplitude:",A)
print("frequency modulation:", freqmod)
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6))
ax0.plot(o0_host[20000:], label="o0");  ax0.legend()
ax1.plot(signal[:T], label="signal");  ax1.legend()
ax1.plot(o1_host, label="o1");  ax1.legend()
plt.tight_layout()
plt.show()

