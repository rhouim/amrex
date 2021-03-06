#ifndef AMREX_GPU_DEVICE_H_
#define AMREX_GPU_DEVICE_H_

#include <cstdlib>
#include <memory>
#include <array>
#include <AMReX.H>
#include <AMReX_Utility.H>
#include <AMReX_GpuError.H>
#include <AMReX_GpuControl.H>

namespace amrex {

#if defined(AMREX_USE_HIP)
using gpuStream_t     = hipStream_t;
using gpuDeviceProp_t = hipDeviceProp_t;
#elif defined(AMREX_USE_CUDA)
using gpuStream_t     = cudaStream_t;
using gpuDeviceProp_t = cudaDeviceProp;
#endif

namespace Gpu {

class Device
{

public:

    static void Initialize ();
    static void Finalize ();

#if defined(AMREX_USE_GPU)
    static gpuStream_t gpuStream () noexcept { return gpu_stream; }
#ifdef AMREX_USE_CUDA
    // for backward compatibility
    static cudaStream_t cudaStream () noexcept { return gpu_stream; }
#endif
    static int numGpuStreams () noexcept { return max_gpu_streams; }
#endif
    static void setStreamIndex (const int idx) noexcept;
    static void resetStreamIndex () noexcept { setStreamIndex(-1); }

    static int deviceId () noexcept;

    static void synchronize () noexcept;
    static void streamSynchronize () noexcept;

#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10) )
    // Generic graph selection. These should be called by users. 
    static void startGraphRecording(bool first_iter);
    static cudaGraphExec_t stopGraphRecording(bool last_iter);

    // Capture graphs per iteration
    static void startGraphIterRecording();
    static void stopGraphIterRecording();
    static cudaGraphExec_t assembleGraphIter();

    // Capture graphs along streams
    static void startGraphStreamRecording();
    static cudaGraphExec_t stopGraphStreamRecording();

    // Instantiate a created cudaGtaph
    static cudaGraphExec_t instantiateGraph(cudaGraph_t graph);

    // Execute an instantiated cudaGraphExec
    static void executeGraph(cudaGraphExec_t &graphExec);

    // Set type of graph recording.
    // Currently 2 options, so bool used. Change later if needed. 
    static inline bool isGraphPerStream() { return graph_per_stream; }
    static inline bool isNotGraphPerIter() { return graph_per_stream; }

    static inline bool isGraphPerIter() { return !graph_per_stream; }
    static inline bool isNotGraphPerStream() { return !graph_per_stream; }

    static inline bool setGraphPerStream () {
        bool r = graph_per_stream;
        graph_per_stream = true;
        return r;
    }

    static inline bool setGraphPerIter () {
        bool r = graph_per_stream;
        graph_per_stream = false;
        return r;
    }

#endif

    static void mem_advise_set_preferred (void* p, const std::size_t sz, const int device);
    static void mem_advise_set_readonly (void* p, const std::size_t sz);

#ifdef AMREX_USE_GPU
    static void setNumThreadsMin (int nx, int ny, int nz) noexcept;
    static void n_threads_and_blocks (const long N, dim3& numBlocks, dim3& numThreads) noexcept;
    static void c_comps_threads_and_blocks (const int* lo, const int* hi, const int comps,
                                            dim3& numBlocks, dim3& numThreads) noexcept;
    static void c_threads_and_blocks (const int* lo, const int* hi, dim3& numBlocks, dim3& numThreads) noexcept;
    static void grid_stride_threads_and_blocks (dim3& numBlocks, dim3& numThreads) noexcept;
    static void box_threads_and_blocks (const Box& bx, dim3& numBlocks, dim3& numThreads) noexcept;

    static std::size_t totalGlobalMem () noexcept { return device_prop.totalGlobalMem; }
    static int numMultiProcessors () noexcept { return device_prop.multiProcessorCount; }
    static int maxThreadsPerMultiProcessor () noexcept { return device_prop.maxThreadsPerMultiProcessor; }
    static int maxThreadsPerBlock () noexcept { return device_prop.maxThreadsPerBlock; }
    static int maxThreadsPerBlock (int dir) noexcept { return device_prop.maxThreadsDim[dir]; }
    static int maxBlocksPerGrid (int dir) noexcept { return device_prop.maxGridSize[dir]; }
#endif

    static std::size_t freeMemAvailable ();

#ifdef AMREX_USE_GPU
#ifdef AMREX_USE_CUDA
    static constexpr int warp_size = 32;
#else
    static constexpr int warp_size = 64;
#endif
#endif

private:

    static void initialize_gpu ();

    static int device_id;
    static int verbose;

#ifdef AMREX_USE_GPU
    static constexpr int max_gpu_streams = 16;
    static dim3 numThreadsMin;
    static dim3 numBlocksOverride, numThreadsOverride;

    static std::array<gpuStream_t,max_gpu_streams> gpu_streams;
    static gpuStream_t gpu_stream;
    static gpuDeviceProp_t device_prop;
#if ( defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 10) )
    static bool graph_per_stream;
    static Vector<cudaGraph_t> cuda_graphs;
#endif
#endif
};

// Put these in amrex::Gpu

#if defined(AMREX_USE_GPU)
inline gpuStream_t
gpuStream () noexcept
{
    return Device::gpuStream();
}

inline int
numGpuStreams () noexcept
{
    return Device::numGpuStreams();
}
#endif

inline void
synchronize () noexcept
{
   AMREX_HIP_OR_CUDA( AMREX_HIP_SAFE_CALL(hipDeviceSynchronize());,
                      AMREX_CUDA_SAFE_CALL(cudaDeviceSynchronize()); )
}

inline void
streamSynchronize () noexcept
{
    Device::streamSynchronize();
}

inline void
htod_memcpy (void* p_d, const void* p_h, const std::size_t sz) noexcept
{
    AMREX_HIP_OR_CUDA(
        AMREX_HIP_SAFE_CALL(hipMemcpy(p_d, p_h, sz, hipMemcpyHostToDevice));,
        AMREX_CUDA_SAFE_CALL(cudaMemcpy(p_d, p_h, sz, cudaMemcpyHostToDevice)); )
}

inline void
dtoh_memcpy (void* p_h, const void* p_d, const std::size_t sz) noexcept
{
    AMREX_HIP_OR_CUDA(
        AMREX_HIP_SAFE_CALL(hipMemcpy(p_h, p_d, sz, hipMemcpyDeviceToHost));,
        AMREX_CUDA_SAFE_CALL(cudaMemcpy(p_h, p_d, sz, cudaMemcpyDeviceToHost)); )
}

inline void
htod_memcpy_async (void* p_d, const void* p_h, const std::size_t sz) noexcept
{
    AMREX_HIP_OR_CUDA(
        AMREX_HIP_SAFE_CALL(hipMemcpyAsync(p_d, p_h, sz, hipMemcpyHostToDevice, gpuStream()));,
        AMREX_CUDA_SAFE_CALL(cudaMemcpyAsync(p_d, p_h, sz, cudaMemcpyHostToDevice, gpuStream())); )
}

inline void
dtoh_memcpy_async (void* p_h, const void* p_d, const std::size_t sz) noexcept
{
    AMREX_HIP_OR_CUDA(
        AMREX_HIP_SAFE_CALL(hipMemcpyAsync(p_h, p_d, sz, hipMemcpyDeviceToHost, gpuStream()));,
        AMREX_CUDA_SAFE_CALL(cudaMemcpyAsync(p_h, p_d, sz, cudaMemcpyDeviceToHost, gpuStream())); )
}

#ifdef AMREX_USE_GPU
void callbackAdded ();
void resetNumCallbacks ();
int getNumCallbacks ();
#endif

}}

#endif
