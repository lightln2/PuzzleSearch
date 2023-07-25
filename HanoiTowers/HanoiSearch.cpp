#include "HanoiSearch.h"
#include "HanoiTowers.h"

#include "../Common/BitArray.h"
#include "../Common/Buffer.h"
#include "../Common/CompressedFrontier.h"
#include "../Common/Multiplexor.h"
#include "../Common/SegmentReader.h"
#include "../Common/Store.h"
#include "../Common/StreamVInt.h"
#include "../Common/ThreadUtil.h"
#include "../Common/Util.h"

static void PrintResult(const std::vector<uint64_t>& result) {
    std::cerr << "Radius: " << result.size() - 1 << std::endl;
    std::cerr << "Result:";
    uint64_t sum = 0;
    for (auto w : result) {
        sum += w;
        std::cerr << " " << w;
    }
    std::cerr << "\nTotal: " << WithDecSep(sum) << std::endl;
}

template<int size>
class Expander {
public:
    static constexpr size_t MAX_INDEXES = 4 * 1024 * 1024;
public:
    Expander()
    {
        indexes.reserve(MAX_INDEXES);
        //crosssegChildren.reserve(MAX_INDEXES * 6);
        insegChildren.reserve(MAX_INDEXES * 6);
    }

    template<typename F>
    void AddCrossSegment(int segment, uint32_t index, F func) {
        indexes.push_back(index);
        if (indexes.size() >= MAX_INDEXES) {
            ExpandCrossSegment(segment, func);
        }
    }

    template<typename F>
    void FinishCrossSegment(int segment, F func) {
        if (indexes.size() > 0) {
            ExpandCrossSegment(segment, func);
        }
    }

    std::vector<uint32_t>& ExpandInSegment(int segment, size_t count, uint32_t* indexes) {
        Timer expandTimer;
        insegChildren.clear();
        HanoiTowers<size>::ExpandInSegment(segment, count, indexes, insegChildren);
        m_StatExpandedTimes++;
        m_StatExpandedNodes += count;
        m_StatExpandedNanos += expandTimer.Elapsed();
        return insegChildren;
    }

    static void PrintStats() {
        std::cerr
            << "Expand in-seg: " << WithDecSep(m_StatExpandedTimes) << " times, "
            << WithDecSep(m_StatExpandedNodes) << " nodes in "
            << WithTime(m_StatExpandedNanos)
            << std::endl;
        std::cerr
            << "Expand x-seg: " << WithDecSep(m_StatXExpandedTimes) << " times, "
            << WithDecSep(m_StatXExpandedNodes) << " nodes in "
            << WithTime(m_StatXExpandedNanos)
            << std::endl;
    }

private:
    template<typename F>
    void ExpandCrossSegment(int segment, F func) {
        Timer expandTimer;

        HanoiTowers<size>::ExpandCrossSegment(segment, indexes, crosssegChildren);

        m_StatXExpandedTimes++;
        m_StatXExpandedNodes += indexes.size();
        m_StatXExpandedNanos += expandTimer.Elapsed();

        for (auto child : crosssegChildren) {
            func(child);
        }

        indexes.clear();
        crosssegChildren.clear();
    }

private:
    std::vector<uint32_t> indexes;
    std::vector<uint64_t> crosssegChildren;
    std::vector<uint32_t> insegChildren;

private:
    static std::atomic<uint64_t> m_StatExpandedNodes;
    static std::atomic<uint64_t> m_StatExpandedNanos;
    static std::atomic<uint64_t> m_StatExpandedTimes;
    static std::atomic<uint64_t> m_StatXExpandedNodes;
    static std::atomic<uint64_t> m_StatXExpandedNanos;
    static std::atomic<uint64_t> m_StatXExpandedTimes;
};

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

template<int size>
class HanoiSolver {
public:
    HanoiSolver(
        int segments,
        Store& oldFrontierStore,
        Store& curFrontierStore,
        Store& newFrontierStore,
        Store& curCrossSegmentStore,
        Store& nextCrossSegmentStore)

        : OldFrontierReader(oldFrontierStore)
        , CurFrontierReader(curFrontierStore)
        , FrontierWriter(newFrontierStore)
        , CrossSegmentReader(curCrossSegmentStore)
        , Mult(nextCrossSegmentStore, segments)
        , NextArray(1ui64 << 32)
        , CurArray(1ui64 << 32)
    { }

    void SetInitialNode(const std::string& initialState) {
        auto initialIndex = HanoiTowers<size>::Parse(initialState);
        auto [seg, idx] = HanoiTowers<size>::SplitIndex(initialIndex);
        FrontierWriter.SetSegment(seg);
        FrontierWriter.Add(idx);
        FrontierWriter.Flush();

        auto fnExpandCrossSegment = [&](uint64_t child) {
            auto [s, idx] = HanoiTowers<size>::SplitIndex(child);
            if (s == seg) return;
            Mult.Add(s, idx);
        };

        Expander.AddCrossSegment(seg, idx, fnExpandCrossSegment);
        Expander.FinishCrossSegment(seg, fnExpandCrossSegment);
        Mult.FlushAllSegments();
    }

    uint64_t Expand(int segment) {
        bool hasData = false;

        CrossSegmentReader.SetSegment(segment);
        OldFrontierReader.SetSegment(segment);
        CurFrontierReader.SetSegment(segment);
        FrontierWriter.SetSegment(segment);

        while (true) {
            auto& vect = CrossSegmentReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect[i];
                NextArray.Set(idx);
            }
        }

        while (true) {
            auto& vect = CurFrontierReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                CurArray.Set(vect[i]);
            }
            auto& expandedVect = Expander.ExpandInSegment(segment, vect.Size(), &vect[0]);
            for (const auto child: expandedVect) {
                NextArray.Set(child);
            }
        }

        if (!hasData) return 0;

        while (true) {
            auto& vect = OldFrontierReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect[i];
                NextArray.Clear(idx);
            }
        }

        auto fnExpandCrossSegment = [&](uint64_t child) {
            auto [seg, idx] = HanoiTowers<size>::SplitIndex(child);
            if (seg == segment) return;
            Mult.Add(seg, idx);
        };

        uint64_t count = 0;

        uint64_t indexBase = (uint64_t)segment << 32;

        NextArray.ScanBitsAndClearWithExcl([&](uint64_t index) {
            count++;
            Expander.AddCrossSegment(segment, uint32_t(index), fnExpandCrossSegment);
            FrontierWriter.Add(uint32_t(index));
        }, CurArray);
        CurArray.Clear();
        Expander.FinishCrossSegment(segment, fnExpandCrossSegment);
        FrontierWriter.Flush();

        return count;
    }

    void FinishExpand() {
        Mult.FlushAllSegments();
    }

private:
    CompressedSegmentReader OldFrontierReader;
    CompressedSegmentReader CurFrontierReader;
    CompressedSegmentWriter FrontierWriter;
    SegmentReader CrossSegmentReader;
    SimpleMultiplexor Mult;
    Expander<size> Expander;
    BitArray CurArray;
    IndexedBitArray NextArray;
};

template<int size>
std::vector<uint64_t> HanoiSearch(std::string initialState, SearchOptions options) {
    Timer timer;
    std::cerr << "HanoiSearch (three-bit BFS)" << std::endl;

    int maxSegments = HanoiTowers<size>::MaxSegments();

    std::vector<uint64_t> result;

    StoreOptions sopts;
    sopts.directories = options.directories;
    sopts.filesPerPath = 1;
    Store oldFrontierStore = Store::CreateFileStore(maxSegments, "frontier1", sopts);
    Store curFrontierStore = Store::CreateFileStore(maxSegments, "frontier2", sopts);
    Store newFrontierStore = Store::CreateFileStore(maxSegments, "frontier3", sopts);
    Store curCrossSegmentStore = Store::CreateFileStore(maxSegments, "xseg1", sopts);
    Store nextCrossSegmentStore = Store::CreateFileStore(maxSegments, "xseg2", sopts);

    auto fnSwapStores = [&]() {
        oldFrontierStore.DeleteAll();
        std::swap(oldFrontierStore, curFrontierStore);
        std::swap(curFrontierStore, newFrontierStore);
        curCrossSegmentStore.DeleteAll();
        std::swap(curCrossSegmentStore, nextCrossSegmentStore);
    };

    std::vector<std::unique_ptr<HanoiSolver<size>>> solvers;
    for (int i = 0; i < options.threads; i++) {
        solvers.emplace_back(std::make_unique<HanoiSolver<size>>(
            maxSegments,
            oldFrontierStore,
            curFrontierStore,
            newFrontierStore,
            curCrossSegmentStore,
            nextCrossSegmentStore));
    }

    solvers[0]->SetInitialNode(initialState);
    fnSwapStores();
    result.push_back(1);

    uint64_t nanos_collect = 0;

    std::cerr << "Step: 0; Count: 1" << std::endl;

    while (result.size() <= options.maxSteps) {
        Timer stepTimer;

        std::atomic<uint64_t> totalCount{ 0 };

        ParallelExec(options.threads, maxSegments, [&](int thread, int segment) {
            totalCount += solvers[thread]->Expand(segment);
        });

        ParallelExec(options.threads, [&](int thread) {
            solvers[thread]->FinishExpand();
        });

        if (totalCount == 0) break;
        result.push_back(totalCount);
        fnSwapStores();
        std::cerr
            << "Step: " << result.size() - 1
            << "; count: " << WithDecSep(totalCount)
            << " in " << stepTimer
            << "; size: old frontier=" << WithSize(oldFrontierStore.TotalLength())
            << "; size: frontier=" << WithSize(curFrontierStore.TotalLength())
            << ", x-seg=" << WithSize(curCrossSegmentStore.TotalLength())
            << std::endl;
    }

    std::cerr << "Time: " << timer << std::endl;
    Store::PrintStats();
    Expander<size>::PrintStats();
    StreamVInt::PrintStats();
    PrintResult(result);
    return result;
}


template std::vector<uint64_t> HanoiSearch<17>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<18>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<19>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<20>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<21>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<22>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<23>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<24>(std::string initialState, SearchOptions options);
