import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from simphony.time_domain.utils import  gaussian_pulse, smooth_rectangular_pulse





data = np.load("simulation_data_QAM_signal_generation.npz", allow_pickle=True)
results      = data["results"].item()
T = 20e-11       # total time window (~40 ps)
dt = 1e-14       # time step
t = jnp.arange(0, T, dt)

def upsample_trajectory(I, Q, factor=20):
    I_list, Q_list = [], []
    n = len(I)
    for i in range(n - 1):
        i0, i1 = I[i], I[i+1]
        q0, q1 = Q[i], Q[i+1]
        for alpha in np.linspace(0, 1, factor, endpoint=False):
            I_list.append(i0 + alpha*(i1 - i0))
            Q_list.append(q0 + alpha*(q1 - q0))
    # Add the last point
    I_list.append(I[-1])
    Q_list.append(Q[-1])
    return np.array(I_list), np.array(Q_list)

def plot_sim(inputs, outputs, t):
    """
    Plot input and output signals versus time in picoseconds.

    Parameters
    ----------
    inputs : dict
        Dictionary of input signals. Keys are labels, values are complex-valued arrays.
    outputs : dict
        Dictionary of output signals. Keys are labels, values are complex-valued arrays.
    t : array-like
        Time array in seconds (will be converted to picoseconds for display).
    """
    # Convert time from seconds to picoseconds
    t_ps = t * 1e12  # 1 s = 10^12 ps
    
    input_keys = list(inputs.keys())
    output_keys = list(outputs.keys())
    
    # Determine the number of ports to plot based on the maximum number of signals
    ports = max(len(input_keys), len(output_keys))
    
    fig, axs = plt.subplots(ports, 2, figsize=(12, 3 * ports), squeeze=False)
    
    # Plot input signals on the left column
    for i, key in enumerate(input_keys):
        axs[i, 0].plot(t_ps, jnp.abs(inputs[key])**2,'k-', lw=2)
        axs[i, 0].set_title(f'Input Signal P{i+1}', fontsize=14)
        axs[i, 0].set_xlabel('Time (ps)', fontsize=12)
        axs[i, 0].set_ylabel('Intensity', fontsize=12)
        axs[i, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot output signals on the right column
    for i, key in enumerate(output_keys):
        axs[i, 1].plot(t_ps, jnp.abs(outputs[key])**2,color = 'gold', lw=2)
        axs[i, 1].set_title(f'Output Signal P{i+1}', fontsize=14)
        axs[i, 1].set_xlabel('Time (ps)', fontsize=12)
        axs[i, 1].set_ylabel('Intensity', fontsize=12)
        axs[i, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# I_out = results["I_in"]
# Q_out = results["Q_in"]

I_out = results["I_output"]
Q_out = results["Q_output"]
complex_signal = I_out + 1j*Q_out

# Q_in         = data["Q_in"]
# I_in         = data["I_in"]
t_ps = t * 1e12
# #inputs = data["modelResultinputs"].item()
# #outputs = data["modelResultoutputs"].item()
# #plot_sim(inputs,outputs,t)
# plt.figure(figsize=(10, 8))
# plt.plot(t_ps, I_out, label="I_out")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.title("I vs. Time")
# plt.xlabel("Time (ps)")
# plt.ylabel("Amplitude")

# #plt.plot(t_ps, Q_in, label="I_in")


# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 8))

# plt.plot(t_ps, Q_out, label="Q_out")
# #plt.plot(t_ps, I_in, label="Q_in")
# plt.title("Q vs. Time")
# plt.xlabel("Time (ps)")
# plt.ylabel("Amplitude")
# plt.grid(True, linestyle='--', alpha=0.7)


# plt.legend()
# plt.tight_layout()
# plt.show()
num_outputs = 2
inputs = {
            f'o{i}': smooth_rectangular_pulse(t,0.0e-11,20e-11) if i == 0 else jnp.zeros_like(t)
            for i in range(num_outputs)
        }

signal = I_out + 1j*Q_out

outputs = {
            f'o{i}': signal if i == 1 else jnp.zeros_like(t)
            for i in range(num_outputs)
        }
plot_sim(inputs,outputs,t)

I_out_up, Q_out_up = upsample_trajectory(I_out, Q_out, factor=30)
#I_in_up, Q_in_up = upsample_trajectory(I_in, Q_in, factor=8)
# plt.figure(figsize=(8,6))
# bins = 500
# #plt.hist2d(I_in_up, Q_in_up, bins=bins, cmap='jet',norm=matplotlib.colors.LogNorm() )
# plt.xlim(-3.25,3.25)
# plt.ylim(-3.25,3.25)
# plt.colorbar(label="Counts per bin")
# plt.title("Input (o0)" )
# plt.xlabel("In-Phase (I)")
# plt.ylabel("Quadrature (Q)")
# plt.show()

plt.figure(figsize=(8,6))
bins = 500
plt.hist2d( Q_out_up,I_out_up, bins=bins, cmap='jet',norm=matplotlib.colors.LogNorm() )
plt.colorbar(label="Counts per bin")
plt.title("Output (o1)")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(12, 3 * 2), squeeze=False)
fig.suptitle('Decomposed Complex Output Signal (I, Q)', fontsize=18)

# I‑trace
axs[0, 0].plot(t_ps, I_out, color='blue', lw=2, label='I_out')
axs[0, 0].grid(True, linestyle='--', alpha=0.7)
axs[0, 0].set_title("I vs. Time")
axs[0, 0].set_xlabel("Time (ps)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 0].legend(loc='upper right')   # <— legend here

# Q‑trace
axs[1, 0].plot(t_ps, Q_out, color='red', lw=2, label='Q_out')
axs[1, 0].grid(True, linestyle='--', alpha=0.7)
axs[1, 0].set_title("Q vs. Time")
axs[1, 0].set_xlabel("Time (ps)")
axs[1, 0].set_ylabel("Amplitude")
axs[1, 0].legend(loc='upper right')   # <— and here

# Zoom in on 0–25 ps
for ax in axs.flat:
    ax.set_xlim(0, 25)

# Make room for the suptitle
fig.subplots_adjust(top=0.88)

plt.tight_layout()
plt.show()
