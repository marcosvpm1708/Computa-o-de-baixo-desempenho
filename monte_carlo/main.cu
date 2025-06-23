#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// declaração do kernel (extern "C" se for lib compartilhada)
extern "C" __global__
void mc_kernel(float *d_out,
               int   n_samples,
               float mu,
               float sigma,
               unsigned int seed_offset);

// helper pra checar erros
inline void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

int main() {
    // 1) Detectar dispositivo
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::cout << "Device: " << prop.name
              << " (CC " << prop.major << "." << prop.minor << ")\n"
              << "  maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "\n";

    // 2) Descobrir memória
    size_t free_mem, total_mem;
    check(cudaMemGetInfo(&free_mem, &total_mem), "cudaMemGetInfo");
    std::cout << " Memória livre: " << (free_mem/1e6) << " MB / "
              << (total_mem/1e6) << " MB\n";

    // 3) Parâmetros MC
    int   n_samples   = 1 << 20;      // 1 048 576 amostras
    float mu          = 0.0f;
    float sigma       = 1.0f;
    unsigned seed_off = 1234;

    // 4) Alocar memória device
    float *d_out = nullptr;
    size_t bytes = n_samples * sizeof(float);
    check(cudaMalloc(&d_out, bytes), "cudaMalloc");

    // 5) Configurar grid/block dinamicamente
    int blockSize = std::min(256, prop.maxThreadsPerBlock);
    int numBlocks = (n_samples + blockSize - 1) / blockSize;
    std::cout << " Lançando kernel com grid=("
              << numBlocks << ","<<1<<","<<1<<") block=("
              << blockSize<<","<<1<<","<<1<<")\n";

    // 6) Lançar kernel
    mc_kernel<<<numBlocks, blockSize>>>(d_out,
                                        n_samples,
                                        mu, sigma,
                                        seed_off);
    check(cudaGetLastError(), "Kernel launch");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // 7) Copiar de volta só 5 primeiros para validar
    float host_out[5] = {0};
    check(cudaMemcpy(host_out, d_out, 5*sizeof(float),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H");

    std::cout << "Resultados (5 primeiras): ";
    for (float v : host_out) std::cout << v << "  ";
    std::cout << "\n";

    // 8) Cleanup
    cudaFree(d_out);
    return 0;
}


