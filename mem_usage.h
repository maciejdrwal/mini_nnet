#ifndef MemUsage_h__
#define MemUsage_h__

#include <cstddef>
#include <string>

namespace mem_usage
{
    /**
     * Returns the peak (maxiumum so far) resident set size (physical
     * memory use) measured in bytes, or zero if the value cannot be
     * determined on this OS.
     */
    std::size_t get_peak_rss();

    /**
     * Returns the current resident set size (physical memory use) measured
     * in bytes, or zero if the value cannot be determined on this OS.
     */
    std::size_t get_current_rss();

    /**
     * Convenience function that returns a memory usage summary string.
     */
    std::string mem_usage_summary();

    const int KB_BYTES = 1024;
    const int MB_BYTES = 1024 * KB_BYTES;
    const int GB_BYTES = 1024 * MB_BYTES;
}

#endif
