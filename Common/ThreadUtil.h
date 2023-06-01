#pragma once

#include <atomic>
#include <thread>

template<typename F>
void ParallelExec(int threadsCount, F func) {
    auto fnProcess = [&](int threadIndex) {
        func(threadIndex);
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < threadsCount; i++) {
        threads.emplace_back(fnProcess, i);
    }
    for (auto& thread : threads) thread.join();
}


template<typename F>
void ParallelExec(int threadsCount, int segmentsCount, F func) {
    std::atomic<int> currentSegment{ 0 };

    auto fnProcess = [&](int threadIndex) {
        while (true) {
            int segment = currentSegment.fetch_add(1);
            if (segment >= segmentsCount) break;
            func(threadIndex, segment);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < threadsCount; i++) {
        threads.emplace_back(fnProcess, i);
    }
    for (auto& thread : threads) thread.join();
}

