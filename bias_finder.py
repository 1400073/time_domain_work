import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config
import os, sys
config.update("jax_enable_x64", True)
import pickle
import time
from pathlib import Path
project_root = Path().resolve()              # e.g. /workspaces/TIME_DOMAIN_WORK
simphony_path = project_root / "simphony"    # relative clone folder
sys.path.insert(0, str(simphony_path))
import simphony
from scipy import signal
from simphony.utils import SPEED_OF_LIGHT
from simphony.libraries import siepic,ideal
from simphony.time_domain.ideal import Modulator
from simphony.time_domain.simulation import TimeResult, TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import contextlib

weights_pos =[(1.5090389251708984-0.39406076073646545j), (-0.02254176326096058+0.2515658736228943j), (-1.819382667541504-0.22460536658763885j), (-0.8253945112228394+0.9981537461280823j), (-0.5082707405090332+1.18967604637146j), (-0.0001774576521711424-0.6961514949798584j), (-0.03673146292567253+0.44644805788993835j), (-0.1459171175956726+0.06807226687669754j), (-0.5036165714263916-0.720775306224823j), (0.22833532094955444-0.6702100038528442j), (0.4913672208786011-0.5054720640182495j), (0.23964214324951172-0.17941826581954956j), (-0.8479951620101929+0.7080851793289185j), (-0.13165295124053955+1.22820246219635j), (-0.3264063894748688+0.9105650782585144j), (-0.5630364418029785+0.12412437796592712j), (0.6648470163345337-0.021696971729397774j), (0.768333911895752-0.26949793100357056j), (0.18256008625030518+1.3365470170974731j), (0.3749624788761139+0.6753222942352295j), (-0.007524069398641586+0.12470445781946182j), (0.24145781993865967-1.1117314100265503j), (-2.0713939666748047+0.13644053041934967j), (0.6786695718765259-0.9033428430557251j), (-0.7801675200462341+0.8043974041938782j)]
weights_neg =[(1.1716632843017578+0.8445451259613037j), (-0.7019909024238586-0.5123153328895569j), (1.5928280353546143-0.8183402419090271j), (-0.48552241921424866+1.1273351907730103j), (-0.4933079779148102-0.3121565878391266j), (0.5230439305305481+0.8503711819648743j), (1.3481826782226562-1.3615267276763916j), (-0.31822851300239563-0.6919068694114685j), (0.6234993934631348-0.5789737701416016j), (0.27767032384872437-0.08296317607164383j), (0.24960929155349731-0.014554109424352646j), (-0.5633884072303772+0.04182880371809006j), (-0.95609050989151-0.6565830707550049j), (0.7877309918403625+0.15527625381946564j), (0.29248183965682983+0.6491250991821289j), (0.03677031770348549+0.43369022011756897j), (0.6928834915161133-1.2170876264572144j), (-1.004181146621704-0.21675696969032288j), (-0.40289801359176636+1.09256911277771j), (-1.348107933998108-0.8007844686508179j), (-0.506431519985199+0.20231975615024567j), (-0.6783027648925781-0.6386787295341492j), (-0.9982107281684875+0.07071749120950699j), (0.5076889991760254-0.5713299512863159j), (0.4646381139755249+0.3727649748325348j)]

bias = -0.5240516066551208


data = np.load("X_mmi_binary_10.npz")
X_re    = data["X_re"]
X_im    = data["X_im"]
y = data["labels"]
y_data = np.asarray(y, dtype=float)
X_data    = X_re + 1j * X_im  


def split_pos_neg(X_raw):
    xpos_list = []
    xneg_list = []
    for i in range(0, 50, 10):
        xpos_list.append(X_raw[:, i:i+5])  # pos: first 5 ports
        xneg_list.append(X_raw[:, i+5:i+10])  # neg: next 5 ports
    xpos = np.concatenate(xpos_list, axis=1)  # shape: (N, 25)
    xneg = np.concatenate(xneg_list, axis=1)  # shape: (N, 25)
    return xpos, xneg

xpos, xneg = split_pos_neg(X_data)
Y = np.abs(np.sum(xpos*weights_pos, axis =1 ))**2 - np.abs(np.sum(xneg*weights_neg, axis = 1))**2 + bias

T = 10.0e-11
dt = 1e-14
t = jnp.arange(0,T,dt)
wavelengths = [1.548,1.549,1.55,1.551,1.552]


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
pos_amp_bias = []
pos_phase_bias = []
neg_amp_bias = []
neg_phase_bias = []
for i,w in enumerate(wavelengths):
    print("\n\npos_bias:\n")
    x_pos_current = xpos[:,i*5:i*5+5]
    x_neg_current = xneg[:,i*5:i*5+5]

    for j,weight in enumerate(weights_pos[i*5:i*5+5]):
        def phase_mod(amp):
            return 2*np.arccos(amp)
        x_pos_value = x_pos_current[:,j]

        phase_1 = phase_mod(np.abs(weight/15))
        phase_2 = -phase_1/2+np.angle(weight)
        phase_mod1 = Modulator(mod_signal=phase_1*jnp.ones_like(t))
        phase_mod0 = Modulator(mod_signal = 0*jnp.ones_like(t))
        phase_mod2 = Modulator(mod_signal=phase_2*jnp.ones_like(t))
        models = {
            "waveguide": siepic.waveguide,
            "y_branch": siepic.y_branch,
        }
        models["phase_modulator1"] = phase_mod1
        models["phase_modulator0"] = phase_mod0
        models["phase_modulator2"] = phase_mod2
        wvl = np.linspace(1.50,1.60,200)
        options = {"wl":wvl, "wg1":{"length":10.0},"wg2":{"length":10.0},}
        time_sim = TimeSim(netlist=netlist, models=models, settings = options)

        inputs = {
            "o0": x_pos_value[:len(t)],
            "o1": jnp.zeros_like(t),
        }

        c = 299792458.0
        with open(os.devnull, 'w') as fnull, \
            contextlib.redirect_stdout(fnull), \
            contextlib.redirect_stderr(fnull):
                results = time_sim.run(t,inputs, carrier_freq=c/(w*1e-6), dt=dt)
        output = results.outputs["o1"]
        ref = x_pos_value[3950]       
        desired_pos_amp = np.abs(ref * weight)
        desired_pos_phase = np.angle(ref*weight)
        pos_amp_bias.append(desired_pos_amp/np.abs(output[4000]*15))
        pos_phase_bias.append((desired_pos_phase - np.angle(output[4000])))

        print("amplitude: ", np.abs(desired_pos_amp*np.abs(weight))/np.abs(output[4000]*15))
        print("phase: ", (desired_pos_phase - np.angle(output[4000])))

    print("\n\nneg_bias:\n")
    for j,weight in enumerate(weights_neg[i*5:i*5+5]):
        def phase_mod(amp):
            return 2*np.arccos(amp)
        phase_1 = phase_mod(np.abs(weight/15))
        phase_2 = -phase_1/2+np.angle(weight)
        phase_mod1 = Modulator(mod_signal=phase_1*jnp.ones_like(t))
        phase_mod0 = Modulator(mod_signal = 0*jnp.ones_like(t))
        phase_mod2 = Modulator(mod_signal=phase_2*jnp.ones_like(t))
        models = {
            "waveguide": siepic.waveguide,
            "y_branch": siepic.y_branch,
        }
        models["phase_modulator1"] = phase_mod1
        models["phase_modulator0"] = phase_mod0
        models["phase_modulator2"] = phase_mod2
        wvl = np.linspace(1.50,1.60,200)
        options = {"wl":wvl, "wg1":{"length":10.0},"wg2":{"length":10.0},}
        time_sim = TimeSim(netlist=netlist, models=models, settings = options)
        x_neg_value = x_neg_current[:,j]

        inputs = {
            "o0": x_neg_value[:len(t)],
            "o1": jnp.zeros_like(t),
        }

        c = 299792458.0
        with open(os.devnull, 'w') as fnull, \
            contextlib.redirect_stdout(fnull), \
            contextlib.redirect_stderr(fnull):
                results = time_sim.run(t,inputs, carrier_freq=c/(w*1e-6), dt=dt)
        output = results.outputs["o1"]
        ref = x_neg_value[3950]       # or x_neg_value[3950] in the neg loop
        desired_neg_amp = np.abs(ref * weight)
        desired_neg_phase = np.angle(ref*weight)
        neg_amp_bias.append(desired_neg_amp/np.abs(output[4000]*15))
        neg_phase_bias.append(desired_neg_phase- np.angle(output[4000]))
        
        print("amplitude: ", desired_neg_amp/np.abs(output[4000]*15))
        print("phase: ", (desired_neg_phase - np.angle(output[4000])))

np.savez_compressed(
    "bias_values.npz",
    pos_amp   = np.array(pos_amp_bias,   dtype=np.float32),
    pos_phase = np.array(pos_phase_bias, dtype=np.float32),
    neg_amp   = np.array(neg_amp_bias,   dtype=np.float32),
    neg_phase = np.array(neg_phase_bias, dtype=np.float32),
)
        
