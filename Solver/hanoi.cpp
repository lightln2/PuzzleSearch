#include "../HanoiTowers/HanoiSearch.h"
#include "../HanoiTowers/HanoiTowers.h"

void HanoiSearch() {
    HanoiTowers<17> towers;
    std::string initial = towers.ToString(0);

    SearchOptions opts;
    //opts.storeOptions.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ", "h:/PUZ2" };
    opts.directories = { "c:/PUZ", "d:/PUZ" };
    opts.threads = 1;
    //opts.maxSteps = 140;
    //opts.maxSteps = 140;

    HanoiSearch<towers.Size>(initial, opts);
}
