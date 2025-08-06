# ------------------------------------------------------------
# 1.  Parameters
# ------------------------------------------------------------
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

sigma, rho, beta = 10.0, 28.0, 8/3
COARSE_SAMPLES     = 1000          # 1000 Lorenz points
MASKS_PER_SAMPLE   = 50            # virtual nodes
dt                 = 1.0e-14       # simulator step [s]
a                  = 0.1           # mask amplitude

# ------------------------------------------------------------
# 2.  Generate 1000 coarse Lorenz points (any solver step is fine here)
# ------------------------------------------------------------
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

STEPS_PER_SLOT = 4000          
RAMP_LEN       = 1000          
SLOTS          = COARSE_SAMPLES * MASKS_PER_SAMPLE
TOTAL_STEPS    = SLOTS * STEPS_PER_SLOT


np.random.seed(42)
full_mask_slots = np.tile(
    np.random.uniform(-a, a, MASKS_PER_SAMPLE), COARSE_SAMPLES
)                                             
hold_len = STEPS_PER_SLOT - RAMP_LEN
mask_timed = np.empty(TOTAL_STEPS)

for s in range(SLOTS-1):

    v0 = full_mask_slots[s]+ x_c[s//50] + y_c[s//50]
    v1 = full_mask_slots[s+1]+ x_c[s//50] + y_c[s//50]
    base = s * STEPS_PER_SLOT

    mask_timed[base : base+hold_len] = v0 
    u = np.linspace(0, 1, RAMP_LEN, endpoint=False)
    ramp = v0 + (v1-v0)*0.5*(1 - np.cos(np.pi*u))

    mask_timed[base+hold_len : base+STEPS_PER_SLOT] = ramp

mask_timed[(SLOTS-1)*STEPS_PER_SLOT :] = full_mask_slots[-1]

signal1 = mask_timed

plt.figure(figsize=(6,3))
plt.plot(signal1[:100000], lw=1)
plt.show()


