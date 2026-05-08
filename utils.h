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
            return std::chrono::duration_cast<std::chrono::seconds>(ClockT::now() - _timer).count();
        }

    private:
        ClockT::time_point _timer;

    };

    std::vector<std::string> split(std::string s, const std::string& delimiter) {
        std::vector<std::string> tokens;
        size_t pos = 0;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            std::string token = s.substr(0, pos);
            tokens.push_back(token);
            s.erase(0, pos + delimiter.length());
        }
        tokens.push_back(s);
    
        return tokens;
    }
}