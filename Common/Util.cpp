#include "Util.h"

std::string WithDecSep(uint64_t value) {
    char str[32];
    int pos = 0;
    if (value == 0) return "0";
    while (value != 0) {
        if (pos % 4 == 3) str[pos++] = ',';
        str[pos++] = '0' + value % 10;
        value /= 10;
    }
    std::reverse(str, str + pos);
    return std::string(str, str + pos);
}

std::string WithTime(uint64_t nanos) {
    std::ostringstream stream;
    auto millis = nanos / 1000000i64;
    auto seconds = millis / 1000;
    auto minutes = seconds / 60;
    auto hours = minutes / 60;
    if (minutes == 0) {
        stream
            << std::setfill('0')
            << seconds << '.'
            << std::setw(3) << millis % 1000;
        return stream.str();
    }
    if (hours == 0) {
        stream
            << std::setfill('0')
            << minutes << ':'
            << std::setw(2) << seconds % 60 << '.'
            << std::setw(3) << millis % 1000;
        return stream.str();
    }
    stream
        << std::setfill('0')
        << std::setw(2) << hours << ':'
        << std::setw(2) << minutes % 60 << ':'
        << std::setw(2) << seconds % 60;
    return stream.str();
}

std::string WithSize(uint64_t size) {
    std::ostringstream stream;

    auto sizeBytes = size, sizeKB = sizeBytes / 1024, sizeMB = sizeKB / 1024, sizeGB = sizeMB / 1024;

    if (sizeKB == 0) {
        stream << sizeBytes << " b";
        return stream.str();
    }
    if (sizeMB == 0) {
        stream << sizeKB << '.' << (sizeBytes % 1024) / 100 << " kb";
        return stream.str();
    }
    if (sizeGB == 0) {
        stream << sizeMB << '.' << (sizeKB % 1024) / 100 << " mb";
        return stream.str();
    }

    stream << sizeGB << '.' << (sizeMB % 1024) / 100 << " gb";
    return stream.str();
}

void PrintVecor(const std::string& title, const std::vector<std::string>& values) {
    std::cerr << title << ": [ ";
    for (const auto& val : values) std::cerr << val << ' ';
    std::cerr << "]" << std::endl;
}

std::vector<std::string> CreatePaths(const std::vector<std::string>& directories, const std::string& path) {
    std::vector<std::string> paths;
    for (const auto& dir : directories) {
        if (dir.ends_with('/'))
            paths.push_back(dir + path);
        else
            paths.push_back(dir + "/" + path);
    }
    return paths;
}

std::ostream& operator<<(std::ostream& os, const Timer& timer) {
    os << WithTime(timer.Elapsed());
    return os;
}
