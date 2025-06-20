#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <builtin_types.h>

#include <cnpy.h>
#include <cuda_fp16.h>

#include <vector>
#include <fstream>
#include <cmath>
#include <cublas_v2.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

#define checkCublasErrors(call)                                                   \
    {                                                                             \
        cublasStatus_t status = call;                                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS Error: %d in %s at line %d\n",                \
                    status, __FILE__, __LINE__);                                  \
            exit(-1);                                                             \
        }                                                                         \
    }

//#define M_dim 64
//#define K_dim 256
//#define N_dim 3136

//#define A_Blocks 2
//#define C_Blocks 98

#define Tsy 1
#define Tsz (N_dim / C_Blocks)
#define ST 1
#define Fx 1
#define Fy (Tsz/Fx)

#define Usy (Tsy * Fy)
#define Gsy (Usy)

//#define Gy 2
#define Block_size (Gy * Gsy)

#ifndef HALF
#define HALF 0
#endif

inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}

// --- global variables ----------------------------------------------------
CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;
size_t     totalGlobalMem;

char       *module_file = (char*) "testing.cubin";
#if RESIDUAL
char       *kernel_name = (char*) "_Z2mmPKfS0_Pf";
#else
char       *kernel_name = (char*) "_Z2mmPKfPf";
#endif

// --- functions -----------------------------------------------------------
void initCUDA() {
    int deviceCount = 0;
    CUresult err = cuInit(0);
    int major = 0, minor = 0;

    if (err == CUDA_SUCCESS)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    // get first CUDA device
    checkCudaErrors(cuDeviceGet(&device, 0));
    char name[100];
    cuDeviceGetName(name, 100, device);
    printf("> Using device 0: %s\n", name);

    // get compute capabilities and the devicename
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device));

    err = cuCtxCreate(&context, 0, device);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.\n");
        cuCtxDetach(context);
        exit(-1);
    }
}

void initfunction() {
	auto err = cuModuleLoad(&module, module_file);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error loading the module %s\n", module_file);
        cuCtxDetach(context);
        exit(-1);
    }

    err = cuModuleGetFunction(&function, module, kernel_name);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        cuCtxDetach(context);
        exit(-1);
    }
}

void runKernel(CUdeviceptr d_B, CUdeviceptr d_residual, CUdeviceptr d_C)
{
    void *args[2] = { &d_B, &d_C};

    // grid for kernel: <<<N, 1>>>

    CUevent start, stop;
    //cudaProfilerStart();
    cuEventCreate(&start,0);
    cuEventCreate(&stop,0);
    cuEventRecord(start,0);

    std::cout << "Executing SparseRT kernel\n";
    std::cout << A_Blocks << " " << C_Blocks << " " << Block_size << std::endl;

    for(int i = 0;i < 1000; i ++){
    checkCudaErrors( cuLaunchKernel(function, A_Blocks, C_Blocks, 1,  // Nx1x1 blocks
                                    Block_size, 1, 1,            // 1x1x1 threads
                                    0, 0, args, 0) );
    }

    cuEventRecord(stop,0);
    //cudaProfilerStop();
    cuEventSynchronize(stop);
    float time;
    cuEventElapsedTime(&time,start,stop);
    cuEventDestroy(start);
    cuEventDestroy(stop);
    std::cout << "kernel used " << time / 1000.0 << std::endl;

}

void runCublas(CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C)
{

    cublasHandle_t cublasHandle;
    checkCublasErrors(cublasCreate(&cublasHandle));

    CUevent start, stop;
    //cudaProfilerStart();
    cuEventCreate(&start,0);
    cuEventCreate(&stop,0);
    cuEventRecord(start,0);

    const float alpha = 1.0;
    const float beta = 0.0;

    std::cout << "Executing cuBLAS kernel" << std::endl;

    for(int i = 0;i < 1000; i ++){
        checkCublasErrors(
            cublasSgemm(
                cublasHandle,
                CUBLAS_OP_T, // Transpose first matrix (d_B)
                CUBLAS_OP_T, // Transpose second matrix (d_A)
                N,           // Effective rows of op(A) (N rows from B_T)
                M,           // Effective cols of op(B) (M cols from A_T)
                K,           // Common dimension
                &alpha,      // Alpha scalar
                (float*)d_B, // Device pointer to B (becomes op(A))
                N,           // Leading dimension of B_T (N rows)
                (float*)d_A, // Device pointer to A (becomes op(B))
                K,           // Leading dimension of A_T (K rows)
                &beta,       // Beta scalar
                (float*)d_C, // Device pointer to C_T (result)
                N            // Leading dimension of C_T (N rows)
            )
        );
    }
    cuEventRecord(stop,0);
    //cudaProfilerStop();
    cuEventSynchronize(stop);
    float time;
    cuEventElapsedTime(&time,start,stop);
    cuEventDestroy(start);
    cuEventDestroy(stop);
    checkCublasErrors(cublasDestroy(cublasHandle));
    std::cout << "kernel used " << time / 1000.0 << std::endl;

}

void float2half(float * in, __half * out, int n) {
    for(int i=0; i<n; i++){
        out[i] = __float2half(in[i]);
    }
}

void half2float(__half * in, float * out, int n) {
    for(int i=0; i<n; i++){
        out[i] = __half2float(in[i]);
    }
}


int main(int argc, char **argv)
{
    cnpy::NpyArray arr0 = cnpy::npy_load("A.npy");
    float * A = arr0.data<float>();
    assert(arr0.word_size = sizeof(float));
    assert(arr0.shape.size()==2 && arr1.shape[0] == M_dim && arr1.shape[1] == K_dim);

    cnpy::NpyArray arr1 = cnpy::npy_load("B.npy");
    float * B = arr1.data<float>();
    assert(arr1.word_size = sizeof(float));
    assert(arr1.shape.size()==2 && arr1.shape[0] == K_dim && arr1.shape[1] == N_dim);

    cnpy::NpyArray arr2 = cnpy::npy_load("ref.npy");
    float * C = arr2.data<float>();
    assert(arr2.word_size = sizeof(float));
    assert(arr2.shape.size()==2 && arr2.shape[0] == M_dim && arr2.shape[1] == N_dim);

    cnpy::NpyArray arr3 = cnpy::npy_load("ref_transposed.npy");
    float * C_transposed = arr3.data<float>();
    assert(arr3.word_size = sizeof(float));
    assert(arr3.shape.size()==2 && arr2.shape[0] == N_dim && arr2.shape[1] == M_dim);

    __half * B_h, * C_h;
    B_h = (__half *)malloc(N_dim * K_dim *2);
    C_h = (__half *)malloc(M_dim * N_dim *2);
    float2half(B,B_h,K_dim * N_dim);

    CUdeviceptr d_A, d_B, d_C, d_residual;

    initCUDA();

    #ifndef TESTCUBLAS
    initfunction();
    #endif

    // allocate memory
    checkCudaErrors( cuMemAlloc(&d_A, sizeof(float) * M_dim * K_dim) );
    checkCudaErrors( cuMemAlloc(&d_B, sizeof(float) * K_dim * N_dim) );
    checkCudaErrors( cuMemAlloc(&d_C, sizeof(float) * M_dim * N_dim) );
    
    checkCudaErrors( cuMemcpyHtoD(d_A, A, sizeof(float) * M_dim * K_dim) );
    checkCudaErrors( cuMemcpyHtoD(d_B, B, sizeof(float) * K_dim * N_dim) );

    // run
    //printf("# Running the kernel...\n");
    #ifndef TESTCUBLAS
    runKernel(d_B, d_residual, d_C);
    #else
    runCublas(d_A, d_B, d_C);
    #endif
    //printf("# Kernel complete.\n");

    // copy results to host and report
    float *result;
    result = (float *)malloc(M_dim * N_dim *sizeof(float));
    checkCudaErrors( cuMemcpyDtoH(result, d_C, sizeof(float) * M_dim * N_dim) );

    float error = 0;
    for(int i = 0 ; i < M_dim * N_dim; i ++)
    {
        #ifndef TESTCUBLAS
        auto diff = abs(result[i] - ((float*)C)[i]);
        #else
        auto diff = abs(result[i] - ((float*)C_transposed)[i]);
        #endif
	if (diff > 0.1)
	{
	//	std::cout << i << " " << result[i] << " " << ((float*)C)[i] << std::endl;
	}
	error += diff;
    }

    std::cout << result[0] << result[1] << result[2] << std::endl;
    cnpy::npy_save("ptx_result.npy",&result[0],{M_dim,N_dim},"w");
    std::cout << "error: " << error << std::endl;
    // finish
    //printf("- Finalizing...\n");
    checkCudaErrors( cuMemFree(d_A) );
    checkCudaErrors( cuMemFree(d_B) );
    checkCudaErrors( cuMemFree(d_C) );
    cuCtxDetach(context);
    return 0;
}
