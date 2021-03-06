#ifndef AMREX_GPU_LAUNCH_FUNCTS_G_H_
#define AMREX_GPU_LAUNCH_FUNCTS_G_H_

namespace amrex {

template<typename T, typename L>
void launch (T const& n, L f, std::size_t shared_mem_bytes=0) noexcept
{
    const auto ec = Gpu::ExecutionConfig(n);
    std::size_t sm = std::max(ec.sharedMem, shared_mem_bytes);
    amrex::launch_global<<<ec.numBlocks, ec.numThreads, sm, Gpu::gpuStream()>>>(
    [=] AMREX_GPU_DEVICE () noexcept {
        for (auto const i : Gpu::Range(n)) {
            f(i);
        }
    });
    AMREX_GPU_ERROR_CHECK();
}

template <typename T, typename L, typename M=amrex::EnableIf_t<std::is_integral<T>::value> >
void For (T n, L f, std::size_t shared_mem_bytes=0) noexcept
{
    const auto ec = Gpu::ExecutionConfig(n);
    std::size_t sm = std::max(ec.sharedMem, shared_mem_bytes);
    amrex::launch_global<<<ec.numBlocks, ec.numThreads, sm, Gpu::gpuStream()>>>(
    [=] AMREX_GPU_DEVICE () noexcept {
        for (T i = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
             i < n; i += stride) {
            f(i);
        }
    });
    AMREX_GPU_ERROR_CHECK();
}

template <typename T, typename L, typename M=amrex::EnableIf_t<std::is_integral<T>::value> >
void ParallelFor (T n, L f, std::size_t shared_mem_bytes=0) noexcept
{
    For(n,f,shared_mem_bytes);
}

template <typename L>
void For (Box const& box, L f, std::size_t shared_mem_bytes=0) noexcept
{
    int ncells = box.numPts();
    const auto lo  = amrex::lbound(box);
    const auto len = amrex::length(box);
    const auto ec = Gpu::ExecutionConfig(ncells);
    std::size_t sm = std::max(ec.sharedMem, shared_mem_bytes);
    amrex::launch_global<<<ec.numBlocks, ec.numThreads, sm, amrex::Gpu::gpuStream()>>>(
    [=] AMREX_GPU_DEVICE () noexcept {
        for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
             icell < ncells; icell += stride) {
            int k =  icell /   (len.x*len.y);
            int j = (icell - k*(len.x*len.y)) /   len.x;
            int i = (icell - k*(len.x*len.y)) - j*len.x;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            f(i,j,k);
        }
    });
    AMREX_GPU_ERROR_CHECK();
}

template <typename L>
void ParallelFor (Box const& box, L f, std::size_t shared_mem_bytes=0) noexcept
{
    For(box,f,shared_mem_bytes);
}

template <typename T, typename L, typename M=amrex::EnableIf_t<std::is_integral<T>::value> >
void For (Box const& box, T ncomp, L f, std::size_t shared_mem_bytes=0) noexcept
{
    int ncells = box.numPts();
    const auto lo  = amrex::lbound(box);
    const auto len = amrex::length(box);
    const auto ec = Gpu::ExecutionConfig(ncells);
    std::size_t sm = std::max(ec.sharedMem, shared_mem_bytes);
    amrex::launch_global<<<ec.numBlocks, ec.numThreads, sm, amrex::Gpu::gpuStream()>>>(
    [=] AMREX_GPU_DEVICE () noexcept {
        for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
             icell < ncells; icell += stride) {
            int k =  icell /   (len.x*len.y);
            int j = (icell - k*(len.x*len.y)) /   len.x;
            int i = (icell - k*(len.x*len.y)) - j*len.x;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            for (T n = 0; n < ncomp; ++n) {
                f(i,j,k,n);
            }
        }
    });
    AMREX_GPU_ERROR_CHECK();
}

template <typename T, typename L, typename M=amrex::EnableIf_t<std::is_integral<T>::value> >
void ParallelFor (Box const& box, T ncomp, L f, std::size_t shared_mem_bytes=0) noexcept
{
    For(box,ncomp,f,shared_mem_bytes);
}

}

#endif
