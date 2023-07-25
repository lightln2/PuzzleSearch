#include "Expander.h"

template<int size>
std::atomic<uint64_t> Expander<size>::m_StatExpandedNodes{ 0 };
template<int size>
std::atomic<uint64_t> Expander<size>::m_StatExpandedNanos{ 0 };
template<int size>
std::atomic<uint64_t> Expander<size>::m_StatExpandedTimes{ 0 };
template<int size>
std::atomic<uint64_t> Expander<size>::m_StatXExpandedNodes{ 0 };
template<int size>
std::atomic<uint64_t> Expander<size>::m_StatXExpandedNanos{ 0 };
template<int size>
std::atomic<uint64_t> Expander<size>::m_StatXExpandedTimes{ 0 };

template class Expander<14>;
template class Expander<15>;
template class Expander<16>;
template class Expander<17>;
template class Expander<18>;
template class Expander<19>;
template class Expander<20>;
template class Expander<21>;
template class Expander<22>;
template class Expander<23>;
template class Expander<24>;
