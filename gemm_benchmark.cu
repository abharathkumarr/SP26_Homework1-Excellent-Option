#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)
#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)
struct BenchmarkResult {
    float avg_time_ms;
    float std_time_ms;
    float min_time_ms;
    float max_time_ms;
    double throughput_tflops;
};
BenchmarkResult benchmark_sgemm(cublasHandle_t handle, int M, int N, int K, 
                                 int warmup_iters, int benchmark_iters) {
    // Allocate matrices: C = A * B where A is MxK, B is KxN, C is MxN
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    // Initialize with random data
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    std::vector<float> timings;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    for (int i = 0; i < benchmark_iters; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        timings.push_back(milliseconds);
    }
    
    // Calculate statistics
    float sum = std::accumulate(timings.begin(), timings.end(), 0.0f);
    float mean = sum / timings.size();
    
    float sq_sum = 0.0f;
    for (float t : timings) {
        sq_sum += (t - mean) * (t - mean);
    }
    float std_dev = std::sqrt(sq_sum / timings.size());
    
    float min_time = *std::min_element(timings.begin(), timings.end());
    float max_time = *std::max_element(timings.begin(), timings.end());
    
    // Calculate FLOPS
    double flops = 2.0 * M * N * K;
    double throughput_tflops = (flops / (mean * 1e-3)) / 1e12;
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return {mean, std_dev, min_time, max_time, throughput_tflops};
}
BenchmarkResult benchmark_gemmex_tf32(cublasHandle_t handle, int M, int N, int K,
                                       int warmup_iters, int benchmark_iters) {
    // Allocate matrices
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    // Initialize with random data
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Set math mode to use TF32 Tensor Cores
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B, CUDA_R_32F, N,
                                  d_A, CUDA_R_32F, K,
                                  &beta,
                                  d_C, CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F_FAST_TF32,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    std::vector<float> timings;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    for (int i = 0; i < benchmark_iters; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B, CUDA_R_32F, N,
                                  d_A, CUDA_R_32F, K,
                                  &beta,
                                  d_C, CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F_FAST_TF32,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        timings.push_back(milliseconds);
    }
    
    // Calculate statistics
    float sum = std::accumulate(timings.begin(), timings.end(), 0.0f);
    float mean = sum / timings.size();
    
    float sq_sum = 0.0f;
    for (float t : timings) {
        sq_sum += (t - mean) * (t - mean);
    }
    float std_dev = std::sqrt(sq_sum / timings.size());
    
    float min_time = *std::min_element(timings.begin(), timings.end());
    float max_time = *std::max_element(timings.begin(), timings.end());
    
    // Calculate FLOPS
    double flops = 2.0 * M * N * K;
    double throughput_tflops = (flops / (mean * 1e-3)) / 1e12;
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return {mean, std_dev, min_time, max_time, throughput_tflops};
}
int main() {
    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Test configuration
    int batch_size = 128;
    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192};
    int warmup_iters = 20;
    int benchmark_iters = 100;
    
    std::cout << "========================================\n";
    std::cout << "CUDA cuBLAS Benchmark Results\n";
    std::cout << "========================================\n\n";
    
    for (int size : sizes) {
        int M = batch_size;
        int N = size;
        int K = size;
        
        std::cout << "========================================\n";
        std::cout << "Matrix Size: " << size << "x" << size << "\n";
        std::cout << "========================================\n";
        
        // Baseline: cublasSgemm (FP32)
        std::cout << "Running cublasSgemm (FP32 CUDA Cores)...\n";
        BenchmarkResult result_fp32 = benchmark_sgemm(handle, M, N, K, warmup_iters, benchmark_iters);
        
        // Tensor Core: cublasGemmEx with TF32
        std::cout << "Running cublasGemmEx (TF32 Tensor Cores)...\n";
        BenchmarkResult result_tf32 = benchmark_gemmex_tf32(handle, M, N, K, warmup_iters, benchmark_iters);
        
        float speedup = result_fp32.avg_time_ms / result_tf32.avg_time_ms;
        
        std::cout << "\nResults:\n";
        std::cout << "  FP32 (CUDA Cores):   " << result_fp32.avg_time_ms << " ms | "
                  << result_fp32.throughput_tflops << " TFLOPS\n";
        std::cout << "  TF32 (Tensor Cores): " << result_tf32.avg_time_ms << " ms | "
                  << result_tf32.throughput_tflops << " TFLOPS\n";
        std::cout << "  Speedup: " << speedup << "x\n\n";
    }
    
    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
