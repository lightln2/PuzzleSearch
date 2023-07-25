#include "../HanoiTowers/HanoiSearch.h"
#include "../HanoiTowers/HanoiTowers.h"

void HanoiSearch() {
    HanoiTowers<22> towers;
    std::string initial = towers.ToString(0);

    SearchOptions opts;
    //opts.storeOptions.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ", "h:/PUZ2" };
    opts.directories = { "c:/PUZ", "d:/PUZ" };
    opts.threads = 6;
    //opts.maxSteps = 140;

    HanoiSearch<towers.Size>(initial, opts);
}
