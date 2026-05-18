#include "utils.h"

#include <algorithm>

namespace mininnet::utils
{
    std::vector<std::string> split(std::string s, const std::string& delimiter)
    {
        std::vector<std::string> tokens;
        size_t pos = 0;
        while ((pos = s.find(delimiter)) != std::string::npos)
        {
            std::string token = s.substr(0, pos);
            tokens.push_back(token);
            s.erase(0, pos + delimiter.length());
        }
        tokens.push_back(s);
    
        return tokens;
    }

    std::string strip(std::string str)
    {
        str.erase(str.begin(), std::find_if_not(str.begin(), str.end(), isspace));
        str.erase((std::find_if_not(str.rbegin(), str.rend(), isspace)).base(), str.end());
        return str;
    }

    bool replace(std::string& str, const std::string& from, const std::string& to) 
    {
        size_t start_pos = str.find(from);
        if(start_pos == std::string::npos)
            return false;
        str.replace(start_pos, from.length(), to);
        return true;
    }

    void replaceAll(std::string& str, const std::string& from, const std::string& to) 
    {
        if(from.empty())
            return;
        size_t start_pos = 0;
        while((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
        }
    }
}
