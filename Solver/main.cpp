
#include <iostream>

void HanoiSearch();
void TestSlidingTile();
void TestDiskBasedHanoi();
void TestPancake();


int main() {
    try {
        //TestSlidingTile();
        HanoiSearch();
        //TestDiskBasedHanoi();
        //TestPancake();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        throw;
    }
}
