#include "FourPegHanoiGPU.h"
#include "HanoiTowersGPU.h"

template<int size, bool useSymmetry>
std::string FourPegHanoiGPU<size, useSymmetry>::Name() const {
    std::ostringstream stream;
    stream
        << "Four-Peg Hanoi Towers (GPU), size=" << m_SimplePuzzle.Size()
        << "; symmetry=" << m_SimplePuzzle.UseSymmetry();
    return stream.str();
}

template<int size, bool useSymmetry>
void FourPegHanoiGPU<size, useSymmetry>::ExpandGpu(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    uint64_t count,
    CuStream stream)
{
    GpuHanoiTowersExpand<size, useSymmetry>(
        gpuIndexes,
        gpuExpanded,
        count,
        stream);
}

template class FourPegHanoiGPU<10, false>;
template class FourPegHanoiGPU<11, false>;
template class FourPegHanoiGPU<12, false>;
template class FourPegHanoiGPU<13, false>;
template class FourPegHanoiGPU<14, false>;
template class FourPegHanoiGPU<15, false>;
template class FourPegHanoiGPU<16, false>;
template class FourPegHanoiGPU<17, false>;
template class FourPegHanoiGPU<18, false>;
template class FourPegHanoiGPU<19, false>;
template class FourPegHanoiGPU<20, false>;
template class FourPegHanoiGPU<21, false>;
template class FourPegHanoiGPU<22, false>;
template class FourPegHanoiGPU<23, false>;

template class FourPegHanoiGPU<10, true>;
template class FourPegHanoiGPU<11, true>;
template class FourPegHanoiGPU<12, true>;
template class FourPegHanoiGPU<13, true>;
template class FourPegHanoiGPU<14, true>;
template class FourPegHanoiGPU<15, true>;
template class FourPegHanoiGPU<16, true>;
template class FourPegHanoiGPU<17, true>;
template class FourPegHanoiGPU<18, true>;
template class FourPegHanoiGPU<19, true>;
template class FourPegHanoiGPU<20, true>;
template class FourPegHanoiGPU<21, true>;
template class FourPegHanoiGPU<22, true>;
template class FourPegHanoiGPU<23, true>;
