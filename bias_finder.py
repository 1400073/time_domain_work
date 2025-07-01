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

weights_pos = [(1.8519843816757202+0.7078672647476196j), (-0.604307234287262-0.1147068589925766j), (0.43786120414733887+1.7585910558700562j), (-0.6022605299949646+0.26262497901916504j), (-1.4301124811172485+2.272144317626953j), (0.44822531938552856+0.13348473608493805j), (-2.0070977210998535-0.04853450134396553j), (0.11580459773540497+0.4857751429080963j), (1.0077588558197021+0.0016502846265211701j), (-0.47071197628974915+0.5913359522819519j), (1.1564805507659912+1.1393290758132935j), (-0.9112538695335388+1.4097117185592651j), (-0.6415706872940063+2.4744179248809814j), (-0.8512062430381775-0.17953583598136902j), (0.3809644281864166+0.25280171632766724j), (0.3938139081001282+0.18153895437717438j), (-0.7374314069747925-0.990584671497345j), (0.3390156030654907-1.3638015985488892j), (-0.20807674527168274-0.2533365488052368j), (0.6081218123435974-0.3395324647426605j), (0.14982298016548157-0.18772412836551666j), (-0.8728575110435486+1.5909597873687744j), (-0.3436660170555115+0.9934117197990417j), (-0.5231232047080994-0.08409158140420914j), (-0.011310526169836521-0.6463790535926819j)]
weights_neg = [(0.5585399866104126-0.9365990161895752j), (-1.753100872039795-0.9364692568778992j), (-0.6305011510848999-1.3713420629501343j), (-5.407022476196289+2.947660207748413j), (4.356186866760254+1.2885116338729858j), (-0.5377777814865112-0.4971708059310913j), (0.8223535418510437+1.4154627323150635j), (0.9820473194122314+1.9421899318695068j), (4.265300750732422+3.2920823097229004j), (-1.7857370376586914-1.6787693500518799j), (-1.0474036931991577-0.4366137385368347j), (-1.5863134860992432-0.8493765592575073j), (-1.3785861730575562-1.931609869003296j), (1.9762458801269531-0.6362915635108948j), (-2.9167187213897705+4.2610673904418945j), (1.0422117710113525-0.03991905599832535j), (1.4107178449630737-0.9849020838737488j), (1.8710063695907593-0.4527903199195862j), (-5.314842224121094+3.4482064247131348j), (4.102719306945801-0.2235642820596695j), (-2.832637071609497+0.2938309609889984j), (-1.1224998235702515-0.3607421815395355j), (-1.6475646495819092-0.9122076630592346j), (3.9600770473480225-0.2542233467102051j), (-2.7423291206359863+2.8689873218536377j)]

bias = -0.4816352427005768


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
    "bias_values_3.0_bias.npz",
    pos_amp   = np.array(pos_amp_bias,   dtype=np.float32),
    pos_phase = np.array(pos_phase_bias, dtype=np.float32),
    neg_amp   = np.array(neg_amp_bias,   dtype=np.float32),
    neg_phase = np.array(neg_phase_bias, dtype=np.float32),
)
        
