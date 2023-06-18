#include "pch.h"

#include "../Common/Multiplexor.h"
#include "../Common/SectorMap.h"

#include <vector>

TEST(TestSectorMap, GetFreeSectors) {
    SectorMap map(10, 3);
    EXPECT_EQ(map.HasFreeSectors(), true);

    EXPECT_EQ(map.Next(0), 1);
    EXPECT_EQ(map.Next(1), 2);
    EXPECT_EQ(map.Next(2), -1);

    SectorMap::LinkedList list1;
    SectorMap::LinkedList list2;
    EXPECT_EQ(list1.head, -1);
    EXPECT_EQ(list1.tail, -1);
    map.GetFreeSector(list1);
    EXPECT_EQ(list1.head, 0);
    EXPECT_EQ(list1.tail, 0);
    EXPECT_EQ(map.Next(0), -1);
    EXPECT_EQ(map.HasFreeSectors(), true);

    map.GetFreeSector(list2);
    EXPECT_EQ(list2.head, 1);
    EXPECT_EQ(list2.tail, 1);
    EXPECT_EQ(map.Next(1), -1);
    EXPECT_EQ(map.HasFreeSectors(), true);

    map.GetFreeSector(list1);
    EXPECT_EQ(list1.head, 0);
    EXPECT_EQ(list1.tail, 2);
    EXPECT_EQ(map.Next(0), 2);
    EXPECT_EQ(map.HasFreeSectors(), false);

    map.Free(list1);
    EXPECT_EQ(map.Next(0), 2);
    EXPECT_EQ(map.Next(1), -1);
    EXPECT_EQ(map.Next(2), -1);
    EXPECT_EQ(map.HasFreeSectors(), true);
    EXPECT_EQ(list1.head, -1);
    EXPECT_EQ(list1.tail, -1);

    map.GetFreeSector(list2);
    EXPECT_EQ(list2.head, 1);
    EXPECT_EQ(list2.tail, 0);
    EXPECT_EQ(map.Next(0), -1);
    EXPECT_EQ(map.Next(1), 0);
    EXPECT_EQ(map.Next(2), -1);
    EXPECT_EQ(map.HasFreeSectors(), true);

    map.Free(list2);
    EXPECT_EQ(map.Next(0), -1);
    EXPECT_EQ(map.Next(1), 0);
    EXPECT_EQ(map.Next(2), 1);
    EXPECT_EQ(map.HasFreeSectors(), true);
}

TEST(TestSectorMap, Files) {
    SectorMap map(10, 3);
    SectorFile file(map);

    std::vector<uint8_t> r;
    for (size_t i = 0; i < 2500; i++) {
        r.push_back(i);
    }
    EXPECT_EQ(file.GetSectorsCount(), 0);
    EXPECT_EQ(file.TotalSize(), 0);
    EXPECT_EQ(file.CanWriteWithoutExpand(1000), false);
    file.Write(&r[0], 1000);
    EXPECT_EQ(file.GetSectorsCount(), 1);
    EXPECT_EQ(file.TotalSize(), 1000);
    EXPECT_EQ(file.CanWriteWithoutExpand(1000), false);
    file.Write(&r[1000], 1000);
    EXPECT_EQ(file.GetSectorsCount(), 2);
    EXPECT_EQ(file.TotalSize(), 2000);
    EXPECT_EQ(file.CanWriteWithoutExpand(400), false);
    file.Write(&r[2000], 400);
    EXPECT_EQ(file.GetSectorsCount(), 3);
    EXPECT_EQ(file.TotalSize(), 2400);
    EXPECT_EQ(file.CanWriteWithoutExpand(100), true);
    file.Write(&r[2400], 100);
    EXPECT_EQ(file.GetSectorsCount(), 3);
    EXPECT_EQ(file.TotalSize(), 2500);

    std::vector<uint8_t> r2(2500);
    auto readSize = file.Read(&r2[0], 2500);
    EXPECT_EQ(readSize, 2500);

    for (int i = 0; i < 2500; i++) {
        ensure(r2[i] == r[i]);
    }
    file.Clear();
    EXPECT_EQ(file.GetSectorsCount(), 0);
    EXPECT_EQ(map.HasFreeSectors(), true);
}

