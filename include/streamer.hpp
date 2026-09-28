// TensorStreamer v2 — ultra-low RSS, hard page-cache cap, dual mmap/direct
#pragma once
#include "gguf.hpp"
#include <atomic>
#include <mutex>
#include <vector>
#include <list>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

namespace streamllm {

enum class IOMode { MMAP, DIRECT, AUTO };

struct PageEntry {
    uint64_t file_off;
    size_t   len;
    int      refcount;
    uint64_t last_use;
    bool     is_expert;
};

class TensorStreamer {
public:
    static constexpr size_t PAGE = 4096;
    static constexpr size_t STAGING_POOL_BYTES = 4 * 1024 * 1024; // 4 MB

    GGUFFile* gguf = nullptr;
    IOMode mode = IOMode::AUTO;
    size_t page_cache_cap = 64 * 1024 * 1024;
    size_t page_cache_used = 0;

    uint8_t* staging = nullptr;
    size_t staging_head = 0;

    std::mutex mtx;
    std::list<PageEntry> lru;
    std::unordered_map<uint64_t, std::list<PageEntry>::iterator> page_map;
    uint64_t clock = 0;

    std::atomic<uint64_t> bytes_read{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> major_faults{0};
    std::atomic<uint64_t> evictions{0};

    bool init(GGUFFile* g, IOMode m = IOMode::AUTO, size_t cap_mb = 64) {
        gguf = g;
        mode = m;
        page_cache_cap = cap_mb * 1024 * 1024;
        staging_head = 0;
        if (!staging) {
            if (posix_memalign((void**)&staging, 4096, STAGING_POOL_BYTES) != 0)
                staging = (uint8_t*)malloc(STAGING_POOL_BYTES);
            if (!staging) return false;
            memset(staging, 0, STAGING_POOL_BYTES);
        }
        if (gguf && gguf->mmap_base) {
            madvise(gguf->mmap_base, gguf->file_size, MADV_RANDOM);
            madvise(gguf->mmap_base, gguf->file_size, MADV_DONTNEED);
        }
        return true;
    }

    ~TensorStreamer() {
        drop_all();
        if (staging) { free(staging); staging = nullptr; }
    }

    uint8_t* alloc_staging(size_t n) {
        n = (n + 63) & ~63ULL;
        if (n > STAGING_POOL_BYTES) return nullptr;
        if (staging_head + n > STAGING_POOL_BYTES) staging_head = 0;
        uint8_t* p = staging + staging_head;
        staging_head += n;
        return p;
    }

    bool read_tensor_slice(size_t tensor_id, uint64_t byte_off, size_t length, void* dst) {
        if (!gguf || tensor_id >= gguf->tensors.size()) return false;
        const auto& t = gguf->tensors[tensor_id];
        if (byte_off + length > t.nbytes) return false;
        uint64_t file_off = gguf->data_offset + t.offset + byte_off;

        bool is_expert = t.name.find("exps.") != std::string::npos ||
                         t.name.find("experts.") != std::string::npos ||
                         t.name.find(".ffn_gate_exps.") != std::string::npos ||
                         t.name.find(".ffn_up_exps.") != std::string::npos ||
                         t.name.find(".ffn_down_exps.") != std::string::npos;

        bool use_direct = (mode == IOMode::DIRECT) ||
                          (mode == IOMode::AUTO && is_expert) ||
                          !gguf->mmap_base;

        if (use_direct) {
            size_t done = 0;
            while (done < length) {
                ssize_t r = pread(gguf->fd, (char*)dst + done, length - done, (off_t)(file_off + done));
                if (r < 0) {
                    if (errno == EINTR) continue;
                    return false;
                }
                if (r == 0) return false;
                done += (size_t)r;
            }
            bytes_read.fetch_add(length, std::memory_order_relaxed);
            cache_misses.fetch_add(1, std::memory_order_relaxed);
            return true;
        }

        if (file_off + length > gguf->file_size) return false;
        memcpy(dst, gguf->mmap_base + file_off, length);
        bytes_read.fetch_add(length, std::memory_order_relaxed);
        cache_hits.fetch_add(1, std::memory_order_relaxed);
        track_and_maybe_evict(file_off, length, is_expert);
        return true;
    }

    bool stream_tensor_chunked(size_t tensor_id, size_t chunk = 256 * 1024) {
        if (!gguf || tensor_id >= gguf->tensors.size()) return false;
        const auto& t = gguf->tensors[tensor_id];
        uint64_t off = 0;
        while (off < t.nbytes) {
            size_t n = (size_t)std::min((uint64_t)chunk, t.nbytes - off);
            uint8_t* buf = alloc_staging(n);
            if (!buf) return false;
            if (!read_tensor_slice(tensor_id, off, n, buf)) return false;
            uint64_t foff = gguf->data_offset + t.offset + off;
            release_and_drop(foff, n);
            off += n;
        }
        return true;
    }

    void track_and_maybe_evict(uint64_t file_off, size_t length, bool is_expert) {
        std::lock_guard<std::mutex> lock(mtx);
        uint64_t start = file_off & ~(PAGE - 1);
        uint64_t end = (file_off + length + PAGE - 1) & ~(PAGE - 1);
        for (uint64_t p = start; p < end; p += PAGE) {
            auto it = page_map.find(p);
            if (it != page_map.end()) {
                it->second->last_use = ++clock;
                it->second->refcount++;
                lru.splice(lru.end(), lru, it->second);
            } else {
                PageEntry e{p, PAGE, 1, ++clock, is_expert};
                lru.push_back(e);
                page_map[p] = std::prev(lru.end());
                page_cache_used += PAGE;
            }
        }
        int spins = 0;
        while (page_cache_used > page_cache_cap && !lru.empty()) {
            auto& front = lru.front();
            if (front.refcount > 0) {
                lru.splice(lru.end(), lru, lru.begin());
                if (++spins > (int)lru.size()) break;
                continue;
            }
            if (gguf->mmap_base)
                madvise(gguf->mmap_base + front.file_off, front.len, MADV_DONTNEED);
            page_map.erase(front.file_off);
            page_cache_used -= front.len;
            lru.pop_front();
            evictions.fetch_add(1, std::memory_order_relaxed);
            spins = 0;
        }
    }

    void release_and_drop(uint64_t file_off, size_t length) {
        std::lock_guard<std::mutex> lock(mtx);
        uint64_t start = file_off & ~(PAGE - 1);
        uint64_t end = (file_off + length + PAGE - 1) & ~(PAGE - 1);
        for (uint64_t p = start; p < end; p += PAGE) {
            auto it = page_map.find(p);
            if (it != page_map.end()) {
                if (gguf->mmap_base)
                    madvise(gguf->mmap_base + p, PAGE, MADV_DONTNEED);
                page_cache_used = page_cache_used >= PAGE ? page_cache_used - PAGE : 0;
                lru.erase(it->second);
                page_map.erase(it);
                evictions.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    void release_pages(uint64_t file_off, size_t length) {
        std::lock_guard<std::mutex> lock(mtx);
        uint64_t start = file_off & ~(PAGE - 1);
        uint64_t end = (file_off + length + PAGE - 1) & ~(PAGE - 1);
        for (uint64_t p = start; p < end; p += PAGE) {
            auto it = page_map.find(p);
            if (it != page_map.end() && it->second->refcount > 0)
                it->second->refcount--;
        }
    }

    void drop_all() {
        std::lock_guard<std::mutex> lock(mtx);
        if (gguf && gguf->mmap_base && gguf->file_size)
            madvise(gguf->mmap_base, gguf->file_size, MADV_DONTNEED);
        page_map.clear();
        lru.clear();
        page_cache_used = 0;
    }
};

} // namespace streamllm
