import ctypes
import numpy as np
import os

# Carrega o .so gerado pelo CMake ou nvcc
_lib = os.path.join(os.path.dirname(__file__), "libmonte_carlo.so")
# se estiver em build/, faça:
# _lib = os.path.join(os.path.dirname(__file__), "build", "libmonte_carlo.so")

mc = ctypes.CDLL(_lib)

mc.launch_mc.argtypes = [
    ctypes.c_int, ctypes.c_float, ctypes.c_float,
    ctypes.c_uint, ctypes.POINTER(ctypes.c_float)
]
mc.launch_mc.restype = None

def simulate_mc_cuda(n_samples: int,
                     mu: float,
                     sigma: float,
                     seed_offset: int = 0) -> np.ndarray:
    buf_type = ctypes.c_float * n_samples
    host_buf = buf_type()
    mc.launch_mc(n_samples, mu, sigma, seed_offset, host_buf)
    return np.frombuffer(host_buf, dtype=np.float32)

if __name__ == "__main__":
    out = simulate_mc_cuda(100000, 0.0, 1.0, 42)
    print("Done CUDA MC, sample:", out[:5])
