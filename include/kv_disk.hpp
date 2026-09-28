// Disk-backed KV v2 — high context (up to 1M), windowed residency, sparse file
#pragma once
#include <cstdint>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <atomic>
#include <algorithm>

namespace streamllm {

class DiskKVCache {
public:
    int fd = -1;
    uint8_t* base = nullptr;
    size_t mapped_size = 0;
    std::string path;

    int n_layer = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int n_ctx = 0;
    size_t bytes_per_token = 0;
    size_t block_tokens = 512;
    size_t resident_window = 2048;

    int n_tokens = 0;
    std::atomic<uint64_t> evict_count{0};
    std::atomic<uint64_t> reload_count{0};
    std::atomic<uint64_t> bytes_written{0};

    bool init(const std::string& disk_path, int layers, int heads_kv, int dim, int ctx,
              size_t window = 2048) {
        path = disk_path;
        n_layer = layers;
        n_head_kv = heads_kv;
        head_dim = dim;
        n_ctx = ctx;
        resident_window = window;
        // K+V f16
        bytes_per_token = (size_t)2 * (size_t)n_layer * (size_t)n_head_kv * (size_t)head_dim * 2;
        if (bytes_per_token == 0) bytes_per_token = 1;
        mapped_size = bytes_per_token * (size_t)n_ctx;
        mapped_size = (mapped_size + 4095) & ~4095ULL;
        // Cap mapped size for safety on tiny hosts (still allows huge logical ctx via sparse)
        // Sparse file: physical disk only grows where written.
        fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd < 0) return false;
        if (ftruncate(fd, (off_t)mapped_size) != 0) {
            ::close(fd); fd = -1; return false;
        }
        base = (uint8_t*)mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            base = nullptr;
            ::close(fd); fd = -1; return false;
        }
        madvise(base, mapped_size, MADV_RANDOM);
        // Don't keep the whole sparse map resident
        madvise(base, mapped_size, MADV_DONTNEED);
        n_tokens = 0;
        return true;
    }

    void close() {
        if (base && base != MAP_FAILED) {
            munmap(base, mapped_size);
            base = nullptr;
        }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }

    ~DiskKVCache() { close(); }

    uint8_t* ptr(int layer, int type, int pos) {
        size_t off = (size_t)pos * bytes_per_token
                   + (size_t)layer * ((size_t)n_head_kv * head_dim * 2 * 2)
                   + (size_t)type * ((size_t)n_head_kv * head_dim * 2);
        if (off + ((size_t)n_head_kv * head_dim * 2) > mapped_size) return nullptr;
        return base + off;
    }

    uint8_t* append_slot() {
        if (n_tokens >= n_ctx) return nullptr;
        int pos = n_tokens++;
        // Touch one page to allocate sparse storage for this token
        uint8_t* p = ptr(0, 0, pos);
        if (p) {
            // write a single byte so the page is allocated (sparse hole filled)
            p[0] = (uint8_t)(pos & 0xff);
            bytes_written.fetch_add(1, std::memory_order_relaxed);
        }
        // Evict old window
        if ((size_t)n_tokens > resident_window) {
            size_t drop_end = (size_t)n_tokens - resident_window;
            size_t drop_start = drop_end > block_tokens ? drop_end - block_tokens : 0;
            size_t byte_start = drop_start * bytes_per_token;
            size_t byte_len = (drop_end - drop_start) * bytes_per_token;
            byte_start &= ~4095ULL;
            byte_len = (byte_len + 4095) & ~4095ULL;
            if (byte_start + byte_len <= mapped_size && base) {
                madvise(base + byte_start, byte_len, MADV_DONTNEED);
                evict_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return p;
    }

    // Append many slots (for long-context prefill simulation) without keeping them resident
    int append_many(int n) {
        int got = 0;
        for (int i = 0; i < n; i++) {
            if (!append_slot()) break;
            got++;
        }
        // Bulk drop everything older than window
        if (base && (size_t)n_tokens > resident_window) {
            size_t keep_start = (size_t)n_tokens - resident_window;
            size_t byte_end = keep_start * bytes_per_token;
            byte_end &= ~4095ULL;
            if (byte_end > 0)
                madvise(base, byte_end, MADV_DONTNEED);
        }
        return got;
    }

    void reset() {
        n_tokens = 0;
        if (base) madvise(base, mapped_size, MADV_DONTNEED);
    }

    size_t disk_bytes_logical() const { return mapped_size; }
    size_t disk_bytes_physical() const {
        // approximate via seeks — st_blocks * 512
        struct stat st{};
        if (fd >= 0 && fstat(fd, &st) == 0)
            return (size_t)st.st_blocks * 512;
        return 0;
    }
};

} // namespace streamllm
