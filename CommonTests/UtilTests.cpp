#include "pch.h"

#include "../Common/Util.h"

TEST(UtilTests, WithDecSep) {
  EXPECT_EQ(WithDecSep(1), "1");
  EXPECT_EQ(WithDecSep(12), "12");
  EXPECT_EQ(WithDecSep(123), "123");
  EXPECT_EQ(WithDecSep(1234), "1,234");
  EXPECT_EQ(WithDecSep(12345), "12,345");
  EXPECT_EQ(WithDecSep(123456), "123,456");
  EXPECT_EQ(WithDecSep(1234567), "1,234,567");
  EXPECT_EQ(WithDecSep(12345678), "12,345,678");
  EXPECT_EQ(WithDecSep(123456789), "123,456,789");
  EXPECT_EQ(WithDecSep(1234567890), "1,234,567,890");
  EXPECT_EQ(WithDecSep(12345678901ui64), "12,345,678,901");
  EXPECT_EQ(WithDecSep(123456789012ui64), "123,456,789,012");
  EXPECT_EQ(WithDecSep(1234567890123ui64), "1,234,567,890,123");
}

TEST(UtilTests, WithTime) {
    EXPECT_EQ(WithTime(1234 * 1000000ui64), "1.234");
    EXPECT_EQ(WithTime(12345 * 1000000ui64), "12.345");
    EXPECT_EQ(WithTime(123456 * 1000000ui64), "2:03.456");
    EXPECT_EQ(WithTime((16 * 3600000 + 48 * 60000 + 3 * 1000 + 15) * 1000000ui64), "16:48:03");
}

TEST(UtilTests, WithSize) {
    EXPECT_EQ(WithSize(123), "123 b");
    EXPECT_EQ(WithSize(123 * 1024 + 100), "123.1 kb");
    EXPECT_EQ(WithSize(123 * 1024 * 1024 + 100 * 1024), "123.1 mb");
    EXPECT_EQ(WithSize(123ui64 * 1024 * 1024 * 1024 + 100 * 1024 * 1024), "123.1 gb");
}
