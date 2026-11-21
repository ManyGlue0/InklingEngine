// refdump: run the reference llama.cpp on a prompt and dump named intermediate tensors as .npy files.
//
//   refdump -m model.gguf -o outdir [-p "text" | -f prompt.txt] [-r regex] [-c n_ctx] [--kv f16|q8_0] [--special]
//
// Writes:
//   outdir/tokens.npy           int32 [n_tokens]
//   outdir/logits.npy           f32   [n_tokens, n_vocab]   (every position)
//   outdir/<tensor-name>.npy    f32   ggml shape reversed (numpy row-major), for every graph tensor matching regex
//   outdir/manifest.tsv         name, ggml type, ne0..ne3
// The whole prompt goes in one ubatch, so every tensor name (e.g. "l_out-17") appears exactly once.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

#if defined(_MSC_VER)
#pragma warning(disable : 4996)
#endif
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace fs = std::filesystem;

static void write_npy(const fs::path & path, const char * descr, const void * data, size_t elem_size,
                      const std::vector<int64_t> & shape) {
    std::ostringstream hdr;
    hdr << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) hdr << shape[i] << (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
    hdr << "), }";
    std::string h = hdr.str();
    size_t total = 10 + h.size() + 1;
    h.append((64 - total % 64) % 64, ' ');
    h.push_back('\n');
    std::ofstream f(path, std::ios::binary);
    const char magic[] = "\x93NUMPY\x01\x00";
    f.write(magic, 8);
    uint16_t hl = (uint16_t) h.size();
    f.write((const char *) &hl, 2);
    f.write(h.data(), h.size());
    size_t n = elem_size;
    for (auto s : shape) n *= (size_t) s;
    f.write((const char *) data, n);
}

struct dump_state {
    fs::path    dir;
    std::regex  re;
    std::ofstream manifest;
    int         n_dumped = 0;
};

static bool eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (dump_state *) ud;
    const char * name = ggml_get_name(t);
    if (!name || !name[0]) return ask ? false : true;
    const bool want = std::regex_match(name, st->re);
    if (ask) return want;
    if (!want) return true;

    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (!ggml_is_contiguous(t)) {
        fprintf(stderr, "refdump: skipping non-contiguous %s\n", name);
        return true;
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    switch (t->type) {
        case GGML_TYPE_F32: memcpy(out.data(), raw.data(), n * sizeof(float)); break;
        case GGML_TYPE_F16:
            for (int64_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(((ggml_fp16_t *) raw.data())[i]);
            break;
        case GGML_TYPE_BF16:
            for (int64_t i = 0; i < n; ++i) out[i] = ggml_bf16_to_fp32(((ggml_bf16_t *) raw.data())[i]);
            break;
        case GGML_TYPE_I32:
            for (int64_t i = 0; i < n; ++i) out[i] = (float) ((int32_t *) raw.data())[i];
            break;
        default:
            fprintf(stderr, "refdump: skipping %s of type %s\n", name, ggml_type_name(t->type));
            return true;
    }
    std::vector<int64_t> shape;
    int nd = ggml_n_dims(t);
    for (int i = nd - 1; i >= 0; --i) shape.push_back(t->ne[i]);
    std::string fname = name;
    for (auto & c : fname) if (c == '/' || c == '\\' || c == ':' || c == ' ') c = '_';
    write_npy(st->dir / (fname + ".npy"), "<f4", out.data(), sizeof(float), shape);
    st->manifest << name << '\t' << ggml_type_name(t->type) << '\t' << t->ne[0] << '\t' << t->ne[1] << '\t' << t->ne[2]
                 << '\t' << t->ne[3] << '\t' << ggml_op_desc(t) << '\n';
    st->n_dumped++;
    return true;
}

int main(int argc, char ** argv) {
    std::string model, outdir = "refdump_out", prompt = "The quick brown fox jumps over the lazy dog.", regex = ".*";
    int n_ctx = 4096;
    std::string kv = "f16";
    bool special = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model = next();
        else if (a == "-o") outdir = next();
        else if (a == "-p") prompt = next();
        else if (a == "-f") { std::ifstream f(next(), std::ios::binary); prompt.assign(std::istreambuf_iterator<char>(f), {}); }
        else if (a == "-r") regex = next();
        else if (a == "-c") n_ctx = std::stoi(next());
        else if (a == "--kv") kv = next();
        else if (a == "--special") special = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model.empty()) { fprintf(stderr, "usage: refdump -m model.gguf -o outdir [-p text|-f file] [-r regex]\n"); return 1; }
    fs::create_directories(outdir);

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * mdl = llama_model_load_from_file(model.c_str(), mp);
    if (!mdl) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(mdl);

    std::vector<llama_token> toks(prompt.size() + 16);
    int nt = llama_tokenize(vocab, prompt.data(), (int) prompt.size(), toks.data(), (int) toks.size(), true, special);
    if (nt < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(nt);
    fprintf(stderr, "refdump: %d tokens\n", nt);

    dump_state st;
    st.dir = outdir;
    st.re = std::regex(regex);
    st.manifest.open(fs::path(outdir) / "manifest.tsv");

    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = cp.n_ubatch = std::max(nt, 32);
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = kv == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(mdl, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    llama_batch b = llama_batch_init(nt, 0, 1);
    for (int i = 0; i < nt; ++i) {
        b.token[i] = toks[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = nt;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<float> logits((size_t) nt * n_vocab);
    for (int i = 0; i < nt; ++i) memcpy(&logits[(size_t) i * n_vocab], llama_get_logits_ith(ctx, i), n_vocab * sizeof(float));
    write_npy(fs::path(outdir) / "logits.npy", "<f4", logits.data(), sizeof(float), {nt, n_vocab});
    std::vector<int32_t> ti(toks.begin(), toks.end());
    write_npy(fs::path(outdir) / "tokens.npy", "<i4", ti.data(), sizeof(int32_t), {nt});
    fprintf(stderr, "refdump: wrote %d tensors + logits to %s\n", st.n_dumped, outdir.c_str());

    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(mdl);
    return 0;
}
