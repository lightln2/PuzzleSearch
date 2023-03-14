#include "../SlidingTilePuzzle/FrontierSearch.h"

int main() {

    SearchOptions opts;
    opts.MaxDepth = 29;
    FrontierSearch<4, 4>(opts);


}
