# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial#:~:text=when%20using%20matplotlib%3A-,Error%20%2315%3A%20Initializing%20libiomp5.,performance%20or%20cause%20incorrect%20results.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
from isochrones import get_ichrone

tracks = get_ichrone('mist', tracks=True)

def binary_color_mag_isochrones(
    m1,  # [Solar mass]
    q,   # m2/m1
    age, # [Gyr] 
    fe_h
):
    properties = tracks.generate_binary(
        m1,
        q * m1, 
        np.log10(age) + 9,  
        fe_h,
        bands=["G", "BP", "RP"]
    )
    b_mag = properties.BP_mag.values
    g_mag = properties.G_mag.values
    r_mag = properties.RP_mag.values
    return np.array([b_mag, g_mag, r_mag]).T[0]



example = binary_color_mag_isochrones(1.0, 0.5, 4.5, -0.1)

print(example)


# Following https://www.mackelab.org/sbi/tutorial/00_getting_started/

from sbi import utils
from sbi import analysis
from sbi.inference.base import infer

print(f"Setting prior")

bounds = torch.tensor([
    [0.1,  1.8], # M1
    [0,      1], # q
    [0.2,    6], # age (Gyr)
    [-1,   0.5]  # metallicity
])
prior = utils.BoxUniform(low=bounds.T[0], high=bounds.T[1])

print(f"Emulating posterior")
def simulator(theta):
    return torch.tensor(
        binary_color_mag_isochrones(*theta)
    )

posterior = infer(
    simulator, 
    prior, method='SNPE', num_simulations=1000)

print(f"Creating observation")
observation = binary_color_mag_isochrones(
    1.0, 
    1, 
    4.5, 
    0.0
)
b_mag, g_mag, r_mag = observation
print(observation)
print(b_mag - r_mag, g_mag)

print(f"Sampling")
samples = posterior.sample(
    (10_000,), 
    x=observation
)

print(f"Log probabilities..")
log_probability = posterior.log_prob(samples, x=observation)

print("Plotting..")
fig, ax = analysis.pairplot(
    samples, 
    limits=bounds, 
    labels=["M1", "q", "age", "[M/H]"],
    figsize=(6,6)
)

fig.show()
fig.savefig("explore.png", dpi=300)