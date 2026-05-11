#include "utils.h"

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
}