# mc_py.py
import numpy as np

def complex_calc(mu, sigma, seed):
    # ex.: Box–Muller ou outro cálculo pesado
    rng = np.random.RandomState(seed)
    return rng.normal(mu, sigma)

def simulate_mc_py(n_samples: int, mu: float, sigma: float, seed_offset: int = 0):
    results = np.empty(n_samples, dtype=np.float32)
    for i in range(n_samples):
        results[i] = complex_calc(mu, sigma, seed_offset + i)
    return results

if __name__ == "__main__":
    res = simulate_mc_py(1_000_000, mu=0.0, sigma=1.0)
    print("Done Python MC, sample:", res[:5])
