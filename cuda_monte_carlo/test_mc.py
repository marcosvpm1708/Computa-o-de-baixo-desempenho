# test_mc.py
from mc_py import simulate_mc_py
from wrapper import simulate_mc_cuda
import numpy as np
import time

# Parâmetros de teste
N      = 1_000_000    # maior tamanho reduz ruído estatístico
MU, SD = 0.0, 1.0
SEED   = 42

# 1) Gera resultados
res_py   = simulate_mc_py(N, MU, SD, SEED)
res_cuda = simulate_mc_cuda(N, MU, SD, SEED)

# 2) Estatísticas
mean_py, std_py     = res_py.mean(),   res_py.std()
mean_cuda, std_cuda = res_cuda.mean(), res_cuda.std()
print(f"Python MC: mean={mean_py:.4f}, std={std_py:.4f}")
print(f"CUDA   MC: mean={mean_cuda:.4f}, std={std_cuda:.4f}")

# 3) Testes com tolerância aumentada
tol = 2e-2  # 2%
assert abs(mean_py   - mean_cuda) < tol,   f"Médias divergem >{tol}"
assert abs(std_py    - std_cuda)  < tol,   f"Desvios divergem >{tol}"
print("✓ Testes estatísticos OK")

# 4) Benchmark
t0 = time.time()
simulate_mc_py(N, MU, SD, SEED)
print("Python MC time:", time.time() - t0, "s")

t0 = time.time()
simulate_mc_cuda(N, MU, SD, SEED)
print("CUDA   MC time:", time.time() - t0, "s")
