#pragma once 

#include <vector>
#include <string>
#include <chrono>

namespace mininnet::utils
{
    struct Clock
    {
        using ClockT = std::chrono::high_resolution_clock;

        Clock()
        {
            _timer = ClockT::now();
        }

        int get()
        {
            int result = std::chrono::duration_cast<std::chrono::seconds>(ClockT::now() - _timer).count();
            _timer = ClockT::now();
            return result;
        }

    private:
        ClockT::time_point _timer;

    };

    std::vector<std::string> split(std::string s, const std::string& delimiter);
    std::string strip(std::string str);
}