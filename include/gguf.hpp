#include "termux_compat.hpp"
// GGUF parser — contracts derived from llama.cpp src/llama-model-loader.cpp
// Metadata-only: never reads tensor bytes during load.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace streamllm {

static constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
static constexpr uint32_t GGUF_VERSION = 3;

enum class GGUFType : uint32_t {
    UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5,
    FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12
};

enum class GGMLType : uint32_t {
    F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9,
    Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15,
    IQ2_XXS=16, IQ2_XS=17, IQ3_XXS=18, IQ1_S=19, IQ4_NL=20,
    IQ3_S=21, IQ2_S=22, IQ4_XS=23, I8=24, I16=25, I32=26, I64=27, F64=28,
    IQ1_M=29, BF16=30, COUNT
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    GGMLType dtype;
    uint64_t offset; // relative to data section
    uint64_t nbytes;
};

struct GGUFMeta {
    std::unordered_map<std::string, std::string> strs;
    std::unordered_map<std::string, int64_t> ints;
    std::unordered_map<std::string, double> floats;
    std::unordered_map<std::string, std::vector<int32_t>> arr_i32;
    std::unordered_map<std::string, std::vector<float>> arr_f32;
    std::unordered_map<std::string, std::vector<std::string>> arr_str;
};

inline size_t ggml_type_size(GGMLType t) {
    // bytes per block (see ggml-common.h)
    switch (t) {
        case GGMLType::F32: return 4;
        case GGMLType::F16: case GGMLType::BF16: return 2;
        case GGMLType::Q4_0: return 18;
        case GGMLType::Q4_1: return 20;
        case GGMLType::Q5_0: return 22;
        case GGMLType::Q5_1: return 24;
        case GGMLType::Q8_0: return 34;
        case GGMLType::Q8_1: return 36;
        case GGMLType::Q2_K: return 84;
        case GGMLType::Q3_K: return 110;
        case GGMLType::Q4_K: return 144;
        case GGMLType::Q5_K: return 176;
        case GGMLType::Q6_K: return 210;
        case GGMLType::Q8_K: return 292;
        case GGMLType::IQ2_XXS: return 64;   // 256 elems
        case GGMLType::IQ2_XS:  return 64;
        case GGMLType::IQ3_XXS: return 64;
        case GGMLType::IQ1_S:   return 32;
        case GGMLType::IQ4_NL:  return 18;
        case GGMLType::IQ3_S:   return 64;
        case GGMLType::IQ2_S:   return 64;
        case GGMLType::IQ4_XS:  return 34;
        case GGMLType::IQ1_M:   return 32;
        case GGMLType::I8: return 1;
        case GGMLType::I16: return 2;
        case GGMLType::I32: return 4;
        case GGMLType::I64: return 8;
        case GGMLType::F64: return 8;
        default: return 1;
    }
}

inline size_t ggml_blck_size(GGMLType t) {
    switch (t) {
        case GGMLType::Q4_0: case GGMLType::Q4_1: case GGMLType::Q5_0:
        case GGMLType::Q5_1: case GGMLType::Q8_0: case GGMLType::Q8_1:
        case GGMLType::IQ4_NL: return 32;
        case GGMLType::Q2_K: case GGMLType::Q3_K: case GGMLType::Q4_K:
        case GGMLType::Q5_K: case GGMLType::Q6_K: case GGMLType::Q8_K:
        case GGMLType::IQ2_XXS: case GGMLType::IQ2_XS: case GGMLType::IQ3_XXS:
        case GGMLType::IQ1_S: case GGMLType::IQ3_S: case GGMLType::IQ2_S:
        case GGMLType::IQ4_XS: case GGMLType::IQ1_M: return 256;
        default: return 1;
    }
}

class GGUFFile {
public:
    int fd = -1;
    size_t file_size = 0;
    uint8_t* mmap_base = nullptr;
    uint64_t data_offset = 0;
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    GGUFMeta meta;
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, size_t> tensor_index;

    ~GGUFFile() { close(); }

    void close() {
        if (mmap_base && mmap_base != MAP_FAILED) {
            munmap(mmap_base, file_size);
            mmap_base = nullptr;
        }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }

    // Metadata-only open: maps file but never faults tensor pages during parse
    bool open(const std::string& path, bool use_mmap = true) {
        close();
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) != 0) { close(); return false; }
        file_size = (size_t)st.st_size;
        if (file_size < 24) { close(); return false; }

        if (use_mmap) {
            mmap_base = (uint8_t*)mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
            if (mmap_base == MAP_FAILED) {
                mmap_base = nullptr;
                // fallback: will use pread
            } else {
                // Advise random access; never WILLNEED
                madvise(mmap_base, file_size, MADV_RANDOM);
            }
        }
        return parse_header();
    }

    bool parse_header() {
        auto rd = [&](void* dst, size_t n, uint64_t off) -> bool {
            if (mmap_base) {
                if (off + n > file_size) return false;
                memcpy(dst, mmap_base + off, n);
                return true;
            }
            ssize_t r = pread(fd, dst, n, (off_t)off);
            return r == (ssize_t)n;
        };

        uint32_t magic = 0;
        if (!rd(&magic, 4, 0) || magic != GGUF_MAGIC) { fprintf(stderr,"bad magic\n"); return false; }
        if (!rd(&version, 4, 4) || version < 2 || version > 3) { fprintf(stderr,"bad ver %u\n", version); return false; }
        if (!rd(&n_tensors, 8, 8)) return false;
        if (!rd(&n_kv, 8, 16)) return false;

        uint64_t off = 24;
        // Parse KV metadata
        for (uint64_t i = 0; i < n_kv; i++) {
            uint64_t klen = 0;
            if (!rd(&klen, 8, off)) return false; off += 8;
            std::string key(klen, '\0');
            if (!rd(key.data(), klen, off)) return false; off += klen;
            uint32_t typ = 0;
            if (!rd(&typ, 4, off)) return false; off += 4;
            if (!parse_value(rd, off, (GGUFType)typ, key)) { fprintf(stderr,"fail kv %s typ %u off %llu\n", key.c_str(), typ, (unsigned long long)off); return false; }
        }

        // Parse tensor infos
        tensors.resize(n_tensors);
        fprintf(stderr, "parsing %llu tensors from off %llu\n", (unsigned long long)n_tensors, (unsigned long long)off);
        for (uint64_t i = 0; i < n_tensors; i++) {
            uint64_t nlen = 0;
            if (!rd(&nlen, 8, off)) { fprintf(stderr,"tensor %llu nlen fail off %llu\n",(unsigned long long)i,(unsigned long long)off); return false; } off += 8;
            tensors[i].name.assign(nlen, '\0');
            if (!rd(tensors[i].name.data(), nlen, off)) { fprintf(stderr,"name fail i=%llu\n",(unsigned long long)i); return false; } off += nlen;
            uint32_t ndim = 0;
            if (!rd(&ndim, 4, off)) { fprintf(stderr,"ndim fail i=%llu\n",(unsigned long long)i); return false; } off += 4;
            tensors[i].shape.resize(ndim);
            for (uint32_t d = 0; d < ndim; d++) {
                if (!rd(&tensors[i].shape[d], 8, off)) { fprintf(stderr,"shape fail\n"); return false; } off += 8;
            }
            uint32_t dt = 0;
            if (!rd(&dt, 4, off)) { fprintf(stderr,"dtype fail i=%llu\n",(unsigned long long)i); return false; } off += 4;
            tensors[i].dtype = (GGMLType)dt;
            if (!rd(&tensors[i].offset, 8, off)) { fprintf(stderr,"toff fail i=%llu\n",(unsigned long long)i); return false; } off += 8;

            // compute nbytes
            uint64_t ne = 1;
            for (auto s : tensors[i].shape) ne *= s;
            size_t bs = ggml_blck_size(tensors[i].dtype);
            size_t ts = ggml_type_size(tensors[i].dtype);
            if (bs > 1) tensors[i].nbytes = (ne / bs) * ts;
            else tensors[i].nbytes = ne * ts;

            tensor_index[tensors[i].name] = i;
        }

        // Align data section
        data_offset = (off + 31) & ~31ULL;
        // Validate offsets
        for (auto& t : tensors) {
            if (data_offset + t.offset >= file_size) { fprintf(stderr,"tensor %s bad offset\n", t.name.c_str()); return false; }
        }
        return true;
    }

private:
    template<typename Rd>
    bool parse_value(Rd& rd, uint64_t& off, GGUFType typ, const std::string& key) {
        switch (typ) {
            case GGUFType::UINT8: case GGUFType::INT8: case GGUFType::BOOL: {
                uint8_t v; if (!rd(&v,1,off)) return false; off+=1;
                meta.ints[key] = v; break;
            }
            case GGUFType::UINT16: case GGUFType::INT16: {
                uint16_t v; if (!rd(&v,2,off)) return false; off+=2;
                meta.ints[key] = v; break;
            }
            case GGUFType::UINT32: case GGUFType::INT32: {
                int32_t v; if (!rd(&v,4,off)) return false; off+=4;
                meta.ints[key] = v; break;
            }
            case GGUFType::UINT64: case GGUFType::INT64: {
                int64_t v; if (!rd(&v,8,off)) return false; off+=8;
                meta.ints[key] = v; break;
            }
            case GGUFType::FLOAT32: {
                float v; if (!rd(&v,4,off)) return false; off+=4;
                meta.floats[key] = v; break;
            }
            case GGUFType::FLOAT64: {
                double v; if (!rd(&v,8,off)) return false; off+=8;
                meta.floats[key] = v; break;
            }
            case GGUFType::STRING: {
                uint64_t len; if (!rd(&len,8,off)) return false; off+=8;
                std::string s(len,'\0');
                if (!rd(s.data(),len,off)) return false; off+=len;
                meta.strs[key] = std::move(s); break;
            }
            case GGUFType::ARRAY: {
                uint32_t atyp; if (!rd(&atyp,4,off)) return false; off+=4;
                uint64_t alen; if (!rd(&alen,8,off)) return false; off+=8;
                if ((GGUFType)atyp == GGUFType::STRING) {
                    // Skip huge arrays (e.g. merges ~280k) to keep RSS tiny.
                    // Tokens (~128k) we still load — needed for decode.
                    const bool skip = (alen > 200000) || (key.find("merges") != std::string::npos);
                    if (skip) {
                        for (uint64_t i=0;i<alen;i++) {
                            uint64_t l; if (!rd(&l,8,off)) return false; off+=8;
                            off += l; // skip body
                        }
                        meta.ints[key + ".count"] = (int64_t)alen;
                    } else {
                        std::vector<std::string> arr;
                        arr.reserve(alen > 200000 ? 0 : alen);
                        for (uint64_t i=0;i<alen;i++) {
                            uint64_t l; if (!rd(&l,8,off)) return false; off+=8;
                            std::string s(l,'\0');
                            if (!rd(s.data(),l,off)) return false; off+=l;
                            arr.push_back(std::move(s));
                        }
                        meta.arr_str[key] = std::move(arr);
                    }
                } else if ((GGUFType)atyp == GGUFType::INT32) {
                    std::vector<int32_t> arr(alen);
                    if (alen) if (!rd(arr.data(), alen*4, off)) return false;
                    off += alen*4;
                    meta.arr_i32[key] = std::move(arr);
                } else if ((GGUFType)atyp == GGUFType::FLOAT32) {
                    std::vector<float> arr(alen);
                    if (alen) if (!rd(arr.data(), alen*4, off)) return false;
                    off += alen*4;
                    meta.arr_f32[key] = std::move(arr);
                } else {
                    // skip unknown array types by element size heuristics
                    size_t es = 1;
                    switch ((GGUFType)atyp) {
                        case GGUFType::UINT8: case GGUFType::INT8: case GGUFType::BOOL: es=1; break;
                        case GGUFType::UINT16: case GGUFType::INT16: es=2; break;
                        case GGUFType::UINT32: case GGUFType::INT32: case GGUFType::FLOAT32: es=4; break;
                        default: es=8; break;
                    }
                    off += alen * es;
                }
                break;
            }
            default: return false;
        }
        return true;
    }
};

} // namespace streamllm
