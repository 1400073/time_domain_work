import time
import numpy as np
import jax
from jax import jit
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "simphony")))
import simphony
from simphony.time_domain import TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse 
from simphony.libraries import siepic, ideal
from simphony.time_domain.ideal import Modulator,MMI
import sax
import jax.numpy as jnp
from simphony.time_domain.time_system import (
    BlockModeSystem,
    SampleModeSystem,
    TimeSystem,
    TimeSystemIIR,
)
from simphony.time_domain.pole_residue_model import BVF_Options, IIRModelBaseband
from simphony.utils import dict_to_matrix
# ── your original step function ───────────────────────────────────────────────
netlist = {
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
models = {
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
}

inputs = {
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
    }

ports = sorted(inputs.keys(), key=lambda k: int(k[1:]))  



signals = [ inputs[p] for p in ports ]   


u = jnp.stack(signals, axis=1)         


inputs_per_t = tuple(
    tuple(u[t].tolist())                  
    for t in range(u.shape[0])
)

circuit, _ = sax.circuit(
                            netlist=netlist,
                            models=models,
                        )

s_params_dict = circuit(**options)
s_matrix = np.asarray(dict_to_matrix(s_params_dict))
center_wvl = 1.55
c_light = 299792458
center_freq = c_light / (center_wvl * 1e-6)
freqs = c_light / (wvl * 1e-6) - center_freq
sampling_freq = -1 / dt
beta = sampling_freq / (freqs[-1] - freqs[0])
bvf_options = BVF_Options(beta=beta)
sorted_ports = sorted(netlist["ports"].keys(), key=lambda p: int(p.lstrip('o')))

iir_model = IIRModelBaseband(
    wvl, center_wvl, s_matrix,order = 50, options=bvf_options
)

td_system = TimeSystemIIR(iir_model, sorted_ports)
initial_state = td_system.init_state()



jitted_step = jit(td_system.step)

y0 = jitted_step(initial_state, inputs_per_t[0])

y0.block_until_ready()


n_runs = len(t)
t0 = time.perf_counter()
x_next = initial_state
for ipt in inputs_per_t:
    x_next,out = td_system.step(x_next,ipt)
t1 = time.perf_counter()
avg_non_jit = (t1 - t0) / n_runs
x_next = initial_state
t0 = time.perf_counter()
for ipt in inputs_per_t:
    x_next,out = jitted_step(x_next,ipt).block_until_ready()
t1 = time.perf_counter()
avg_jit = (t1 - t0) / n_runs

# ── 3) Per-call Python→XLA dispatch overhead ─────────────────────────────────
# Total loop time includes both XLA work *and* the small Python dispatch
# If you believe avg_jit ≈ pure compute time, then:
x_next = initial_state
t0 = time.perf_counter()
for _ in range(n_runs):
    x_next,out = jitted_step(x_next,ipt)      # note: NO block_until_ready here
t1 = time.perf_counter()
total_loop = t1 - t0

# estimate overhead per .__call__()
overhead_per_call = (total_loop - avg_jit * n_runs) / n_runs

# ── report ───────────────────────────────────────────────────────────────────
print(f"avg non-jitted call   : {avg_non_jit*1e6:.2f} µs")
print(f"avg jitted   call      : {avg_jit*1e6:.2f} µs")
print(f"dispatch overhead/call : {overhead_per_call*1e6:.2f} µs")
