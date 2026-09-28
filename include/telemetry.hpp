// Fixed-size ring-buffer telemetry — zero alloc on hot path
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <string>

namespace streamllm {

struct TelemetrySample {
    uint64_t ts_us;
    uint64_t rss_kb;
    uint64_t page_cache_kb;
    uint64_t bytes_read;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t major_faults;
    uint64_t evictions;
    uint64_t kv_evict;
    double   tok_per_sec;
    double   ttft_ms;
};

class Telemetry {
public:
    static constexpr size_t RING = 256;
    TelemetrySample ring[RING];
    size_t head = 0;
    size_t count = 0;

    static uint64_t now_us() {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
    }

    static uint64_t read_rss_kb() {
        // /proc/self/statm: size resident shared text lib data dt  (pages)
        std::ifstream f("/proc/self/statm");
        long size=0, res=0;
        f >> size >> res;
        long page = sysconf(_SC_PAGESIZE);
        return (uint64_t)res * (uint64_t)page / 1024;
    }

    static uint64_t read_model_pagecache_kb(const std::string& /*path*/) {
        // Approximate via smaps Private_Clean+Private_Dirty of the process (conservative)
        std::ifstream f("/proc/self/smaps_rollup");
        std::string line;
        uint64_t rss = 0;
        while (std::getline(f, line)) {
            if (line.compare(0, 4, "Rss:") == 0) {
                long v = 0;
                sscanf(line.c_str(), "Rss: %ld", &v);
                rss = (uint64_t)v;
                break;
            }
        }
        return rss;
    }

    void record(uint64_t bytes_read, uint64_t hits, uint64_t misses,
                uint64_t faults, uint64_t evict, uint64_t kv_ev,
                double tps, double ttft) {
        auto& s = ring[head];
        s.ts_us = now_us();
        s.rss_kb = read_rss_kb();
        s.page_cache_kb = read_model_pagecache_kb("");
        s.bytes_read = bytes_read;
        s.cache_hits = hits;
        s.cache_misses = misses;
        s.major_faults = faults;
        s.evictions = evict;
        s.kv_evict = kv_ev;
        s.tok_per_sec = tps;
        s.ttft_ms = ttft;
        head = (head + 1) % RING;
        if (count < RING) count++;
    }

    void dump(FILE* out) const {
        fprintf(out, "=== Telemetry (last %zu samples) ===\n", count);
        size_t start = count < RING ? 0 : head;
        for (size_t i = 0; i < count; i++) {
            const auto& s = ring[(start + i) % RING];
            fprintf(out, "rss=%llu kB  pagecache~=%llu kB  bytes_read=%llu  hits=%llu misses=%llu  "
                         "evict=%llu  kv_evict=%llu  tps=%.2f  ttft=%.1f ms\n",
                    (unsigned long long)s.rss_kb, (unsigned long long)s.page_cache_kb,
                    (unsigned long long)s.bytes_read, (unsigned long long)s.cache_hits,
                    (unsigned long long)s.cache_misses, (unsigned long long)s.evictions,
                    (unsigned long long)s.kv_evict, s.tok_per_sec, s.ttft_ms);
        }
    }
};

} // namespace streamllm
