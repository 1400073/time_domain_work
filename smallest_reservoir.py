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
wl_log = []     # wavelengths (µm)
p1_log = []     # |o1|^2
p2_log = []     # |o2|^2
start, end = 1.545, 1.55
num_measurements = 2000
model_order = 50
wvl = np.linspace(1.5, 1.6, num_measurements)
 
mask      = (wvl >= start) & (wvl <= end)
num_delay = 50
wvl_slice = wvl[mask]

wavelengths = [
    # 1.5460730365182591,
    # 1.548624312156078,
    1.5512256128064033,
    1.5538269134567284,
    1.5564282141070536,
]

for z in wavelengths:
    T = 500.0e-11
    dt = 2.0e-14      
    t = jnp.arange(0, T, dt) 

    mod_signal = jnp.sin(2 * jnp.pi * 1.0e9 * t)*0.7-1 
    # mod_signal = 0*t
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


    num_outputs = 2
    inputs = {
        f'o{i}':  smooth_rectangular_pulse(t, 5.0e-11, T+10e-11) if i == 0 else jnp.zeros_like(t)
        for i in range(num_outputs)
    }
    
    modelResult = time_sim.run(t, inputs,carrier_freq=SPEED_OF_LIGHT/(z*1.0e-6),dt=dt)
    modelResult.plot_sim()

    # wl_log.append(float(i))
    # if 1.0 > np.abs(modelResult.outputs["o1"][-30])**2 > 0.8:
    #     print(i)
    #     print("Resonance at", i) 
        

# # ------------- write to disk -------------
# df = pd.DataFrame({
#     "wavelength_um": wl_log,
#     "power_o1":      p1_log,
#     "power_o2":      p2_log,
# })
# df.to_csv("sweep_results_1p5_to_1p6um.csv", index=False)
# # or, smaller binary:
# # np.savez_compressed("sweep_results.npz",
# #                     wavelength_um=np.array(wl_log),
# #                     power_o1=np.array(p1_log),
# #                     power_o2=np.array(p2_log))
# # -----------------------------------------

# print("Saved 1 000 points to sweep_results_1p5_to_1p6um.csv")