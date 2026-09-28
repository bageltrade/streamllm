// Tokenizer + chat template — contracts from llama.cpp src/llama-vocab.cpp & llama-chat.cpp
#pragma once
#include "gguf.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <cctype>

namespace streamllm {

struct ChatMessage {
    std::string role;    // system | user | assistant | tool
    std::string content;
};

class Tokenizer {
public:
    std::vector<std::string> tokens;          // id -> piece
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<float> scores;
    std::vector<int32_t> token_type; // 1=normal, 2=unknown, 3=control, etc.
    int32_t bos_id = 1;
    int32_t eos_id = 2;
    int32_t unk_id = 0;
    int32_t pad_id = -1;
    bool add_bos = true;
    std::string chat_template;
    std::string model_arch;

    bool load_from_gguf(const GGUFFile& g) {
        // tokens
        auto it = g.meta.arr_str.find("tokenizer.ggml.tokens");
        if (it == g.meta.arr_str.end()) return false;
        tokens = it->second;
        token_to_id.clear();
        for (size_t i = 0; i < tokens.size(); i++) token_to_id[tokens[i]] = (int32_t)i;

        auto sc = g.meta.arr_f32.find("tokenizer.ggml.scores");
        if (sc != g.meta.arr_f32.end()) scores = sc->second;

        auto tt = g.meta.arr_i32.find("tokenizer.ggml.token_type");
        if (tt != g.meta.arr_i32.end()) token_type = tt->second;

        auto get_i = [&](const char* k, int32_t def) -> int32_t {
            auto j = g.meta.ints.find(k);
            return j != g.meta.ints.end() ? (int32_t)j->second : def;
        };
        bos_id = get_i("tokenizer.ggml.bos_token_id", 1);
        eos_id = get_i("tokenizer.ggml.eos_token_id", 2);
        unk_id = get_i("tokenizer.ggml.unknown_token_id", 0);
        pad_id = get_i("tokenizer.ggml.padding_token_id", -1);

        auto ct = g.meta.strs.find("tokenizer.chat_template");
        if (ct != g.meta.strs.end()) chat_template = ct->second;

        auto arch = g.meta.strs.find("general.architecture");
        if (arch != g.meta.strs.end()) model_arch = arch->second;

        return !tokens.empty();
    }

    // Very small BPE-ish / byte-fallback tokenizer sufficient for demo & Qwen/Llama pieces
    // Full production BPE would use merges; here we do greedy longest-match + byte fallback.
    std::vector<int32_t> encode(const std::string& text, bool add_special) const {
        std::vector<int32_t> out;
        if (add_special && add_bos && bos_id >= 0) out.push_back(bos_id);

        size_t i = 0;
        while (i < text.size()) {
            // longest match
            size_t best_len = 0;
            int32_t best_id = unk_id;
            // try up to 64 bytes
            size_t max_try = std::min(text.size() - i, (size_t)64);
            for (size_t len = max_try; len >= 1; --len) {
                std::string sub = text.substr(i, len);
                auto it = token_to_id.find(sub);
                if (it != token_to_id.end()) {
                    best_len = len;
                    best_id = it->second;
                    break;
                }
            }
            if (best_len == 0) {
                // byte fallback: encode as <0xXX> if present, else unk
                char buf[16];
                snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned char)text[i]);
                auto it = token_to_id.find(buf);
                if (it != token_to_id.end()) out.push_back(it->second);
                else out.push_back(unk_id);
                i += 1;
            } else {
                out.push_back(best_id);
                i += best_len;
            }
        }
        return out;
    }

    std::string decode(const std::vector<int32_t>& ids) const {
        std::string s;
        for (auto id : ids) {
            if (id >= 0 && (size_t)id < tokens.size()) s += tokens[id];
        }
        // clean common sentencepiece artifacts
        size_t pos;
        while ((pos = s.find("▁")) != std::string::npos) s.replace(pos, 3, " ");
        while ((pos = s.find("Ġ")) != std::string::npos) s.replace(pos, 2, " ");
        return s;
    }

    std::string decode_one(int32_t id) const {
        if (id >= 0 && (size_t)id < tokens.size()) {
            std::string s = tokens[id];
            size_t pos;
            while ((pos = s.find("▁")) != std::string::npos) s.replace(pos, 3, " ");
            while ((pos = s.find("Ġ")) != std::string::npos) s.replace(pos, 2, " ");
            return s;
        }
        return "";
    }

    bool is_eog(int32_t id) const {
        return id == eos_id || id < 0;
    }
};

// Minimal Jinja-subset + hard-coded fallbacks (llama.cpp chat contract)
inline std::string apply_chat_template(
    const std::string& tmpl,
    const std::vector<ChatMessage>& messages,
    bool add_assistant,
    const std::string& arch = "")
{
    // If we have a real template string, do a very small subset render
    if (!tmpl.empty() && tmpl.find("{%") != std::string::npos) {
        // Extremely simplified: just concatenate role/content with common markers
        // Full Jinja is large; for production we'd embed minja. Here we fall through to heuristics
        // that cover ChatML / Llama-3 / Qwen which dominate instruct GGUFs.
    }

    // Detect / fallback by content of template or architecture
    bool chatml = tmpl.find("<|im_start|>") != std::string::npos ||
                  arch.find("qwen") != std::string::npos ||
                  tmpl.empty();
    bool llama3 = tmpl.find("<|start_header_id|>") != std::string::npos ||
                  arch.find("llama") != std::string::npos;
    bool gemma  = tmpl.find("<start_of_turn>") != std::string::npos ||
                  arch.find("gemma") != std::string::npos;
    bool mistral = tmpl.find("[INST]") != std::string::npos ||
                   arch.find("mistral") != std::string::npos;

    std::ostringstream ss;
    if (llama3 && !chatml) {
        for (auto& m : messages) {
            ss << "<|start_header_id|>" << m.role << "<|end_header_id|>\n\n"
               << m.content << "<|eot_id|>";
        }
        if (add_assistant) ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (gemma) {
        for (auto& m : messages) {
            std::string r = m.role == "assistant" ? "model" : m.role;
            ss << "<start_of_turn>" << r << "\n" << m.content << "<end_of_turn>\n";
        }
        if (add_assistant) ss << "<start_of_turn>model\n";
    } else if (mistral) {
        for (size_t i = 0; i < messages.size(); i++) {
            auto& m = messages[i];
            if (m.role == "system" && i == 0) {
                ss << "[INST] " << m.content << "\n";
                continue;
            }
            if (m.role == "user") ss << "[INST] " << m.content << " [/INST]";
            else if (m.role == "assistant") ss << m.content << "</s>";
        }
        if (add_assistant) { /* already ends ready for assistant */ }
    } else {
        // ChatML default (Qwen, many others)
        for (auto& m : messages) {
            ss << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
        }
        if (add_assistant) ss << "<|im_start|>assistant\n";
    }
    return ss.str();
}

} // namespace streamllm
