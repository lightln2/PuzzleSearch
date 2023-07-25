#include "Expander.h"
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

        Expander.AddCrossSegment(seg, idx);
        auto& vect = Expander.ExpandCrossSegment(seg);
        for (const uint64_t child : vect) {
            auto [s, idx] = HanoiTowers<size>::SplitIndex(child);
            if (s != seg) {
                Mult.Add(s, idx);
            }
        }

        Mult.FlushAllSegments();
    }

    std::pair<uint64_t, uint64_t> Expand(int segment) {
        bool hasData = false;

        bool fullSymmetry = HanoiTowers<size>::PegsMult(segment) == 6;

        CrossSegmentReader.SetSegment(segment);
        OldFrontierReader.SetSegment(segment);
        CurFrontierReader.SetSegment(segment);
        FrontierWriter.SetSegment(segment);

        Timer timerLoadXSeg;

        while (true) {
            auto& vect = CrossSegmentReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect[i];
                NextArray.Set(idx);
            }
        }

        m_StatLoadXSegNanos += timerLoadXSeg.Elapsed();

        Timer timerExpandInSeg;

        while (true) {
            auto& vect = CurFrontierReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                CurArray.Set(vect[i]);
            }
            if (fullSymmetry) {
                for (size_t i = 0; i < vect.Size(); i++) {
                    uint32_t baseIndex = vect[i] & ~3;
                    NextArray.SetNextFourBits(baseIndex);
                }
                auto& expandedVect = Expander.ExpandInSegmentWithoutSmallest(segment, vect.Size(), &vect[0]);
                for (const auto child : expandedVect) {
                    NextArray.Set(child);
                }
            }
            else {
                auto& expandedVect = Expander.ExpandInSegment(segment, vect.Size(), &vect[0]);
                for (const auto child : expandedVect) {
                    NextArray.Set(child);
                }
            }
        }

        if (!hasData) return { 0, 0 };

        m_StatExpandInSegNanos += timerExpandInSeg.Elapsed();

        Timer timerLoadOldSeg;

        while (true) {
            auto& vect = OldFrontierReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect[i];
                NextArray.Clear(idx);
            }
        }

        m_StatLoadOldSegNanos += timerLoadOldSeg.Elapsed();

        Timer timerCollectSeg;

        uint64_t count = 0;
        uint64_t restoredCount = 0;
        uint64_t indexBase = (uint64_t)segment << 32;

        if (fullSymmetry) {
            NextArray.ScanBitsAndClearWithExcl([&](uint64_t index) {
                count++;
                Expander.AddCrossSegment(segment, uint32_t(index));
                FrontierWriter.Add(uint32_t(index));
            }, CurArray);
            restoredCount = count * 6;
        }
        else {
            NextArray.ScanBitsAndClearWithExcl([&](uint64_t index) {
                count++;
                restoredCount += HanoiTowers<size>::PegsMult(indexBase | index);
                Expander.AddCrossSegment(segment, uint32_t(index));
                FrontierWriter.Add(uint32_t(index));
            }, CurArray);
        }

        CurArray.Clear();

        const auto& expandedXSeg = Expander.ExpandCrossSegment(segment);
        for (const uint64_t child : expandedXSeg) {
            auto [seg, idx] = HanoiTowers<size>::SplitIndex(child);
            Mult.Add(seg, idx);
        }
        FrontierWriter.Flush();

        m_StatCollectNanos += timerCollectSeg.Elapsed();
        return { count, restoredCount };
    }

    void FinishExpand() {
        Mult.FlushAllSegments();
    }

    static void PrintStats() {
        std::cerr <<
            "LoadXSeg:    " << WithTime(m_StatLoadXSegNanos) << "\n" <<
            "ExpandInSeg: " << WithTime(m_StatExpandInSegNanos) << "\n" <<
            "LoadOldSeg:  " << WithTime(m_StatLoadOldSegNanos) << "\n" <<
            "Collect:     " << WithTime(m_StatCollectNanos) <<
            std::endl;
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

private:
    static std::atomic<uint64_t> m_StatLoadXSegNanos;
    static std::atomic<uint64_t> m_StatExpandInSegNanos;
    static std::atomic<uint64_t> m_StatLoadOldSegNanos;
    static std::atomic<uint64_t> m_StatCollectNanos;
};
template<int size>
std::atomic<uint64_t> HanoiSolver<size>::m_StatLoadXSegNanos{ 0 };
template<int size>
std::atomic<uint64_t> HanoiSolver<size>::m_StatExpandInSegNanos{ 0 };
template<int size>
std::atomic<uint64_t> HanoiSolver<size>::m_StatLoadOldSegNanos{ 0 };
template<int size>
std::atomic<uint64_t> HanoiSolver<size>::m_StatCollectNanos{ 0 };

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
        std::atomic<uint64_t> totalRestoredCount{ 0 };

        ParallelExec(options.threads, maxSegments, [&](int thread, int segment) {
            auto [count, restoredCount] = solvers[thread]->Expand(segment);
            totalCount += count;
            totalRestoredCount += restoredCount;
        });

        ParallelExec(options.threads, [&](int thread) {
            solvers[thread]->FinishExpand();
        });

        if (totalCount == 0) break;
        result.push_back(totalRestoredCount);
        fnSwapStores();
        std::cerr
            << "Step: " << result.size() - 1
            << "; count: " << WithDecSep(totalCount)
            << " -> " << WithDecSep(totalRestoredCount)
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
    HanoiSolver<size>::PrintStats();
    PrintResult(result);
    return result;
}


template std::vector<uint64_t> HanoiSearch<14>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<15>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<16>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<17>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<18>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<19>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<20>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<21>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<22>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<23>(std::string initialState, SearchOptions options);
template std::vector<uint64_t> HanoiSearch<24>(std::string initialState, SearchOptions options);
