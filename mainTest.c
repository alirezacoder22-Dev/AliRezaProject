#define _POSIX_C_SOURCE 199309L
#define RAI_IMPLEMENTATION
#include "rriftt_ai.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>

static FILE *out = NULL;
static volatile sig_atomic_t stop = 0;
static void on_signal(int sig) { (void)sig; stop = 1; }

#define LOG(...) do { \
    fprintf(stdout, __VA_ARGS__); fflush(stdout); \
    if (out) { fprintf(out, __VA_ARGS__); fflush(out); } \
} while(0)

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double g_t0;
static double elapsed(void) { return now_sec() - g_t0; }

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    out = fopen("re_output.txt", "w");
    if (out) setvbuf(out, NULL, _IONBF, 0);

    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);
    signal(SIGHUP, on_signal);

    g_t0 = now_sec();
    LOG("=== rriftt_ai env test ===\n");
    LOG("[t=%.3f] start\n", elapsed());

    // ---------- 1. system ----------
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    long avail = sysconf(_SC_AVPHYS_PAGES);
    double total_ram = (double)pages * psize / 1048576.0;
    double avail_ram = (double)avail * psize / 1048576.0;
    LOG("[t=%.3f] cpu=%ld  ram=%.0fMB  avail=%.0fMB\n",
        elapsed(), ncpu, total_ram, avail_ram);

    // ---------- 2. matmul bench (small, fast) ----------
    LOG("[t=%.3f] matmul bench (64x64 x3)...\n", elapsed());
    double gflops = 0;
    {
        size_t N = 64;
        float *a = malloc(N*N*4), *b = malloc(N*N*4), *c = malloc(N*N*4);
        for (size_t i=0; i<N*N; i++) {
            a[i] = (float)(i%17)/17.f;
            b[i] = (float)(i%13)/13.f;
        }
        // warmup
        for (size_t i=0; i<N; i++) for (size_t j=0; j<N; j++) {
            float s=0; for (size_t k=0; k<N; k++) s+=a[i*N+k]*b[k*N+j];
            c[i*N+j]=s;
        }
        int iters = 3;
        double t0 = now_sec();
        for (int it=0; it<iters; it++) {
            for (size_t i=0; i<N; i++) for (size_t j=0; j<N; j++) {
                float s=0; for (size_t k=0; k<N; k++) s+=a[i*N+k]*b[k*N+j];
                c[i*N+j]=s;
            }
        }
        double dt = now_sec() - t0;
        gflops = 2.0 * N*N*N * iters / dt / 1e9;
        volatile float sink = c[0]; (void)sink;
        free(a); free(b); free(c);
        LOG("[t=%.3f] matmul: %.3fs  %.2f GFLOPS\n", elapsed(), dt, gflops);
    }

    if (stop) { LOG("[t=%.3f] interrupted\n", elapsed()); goto done; }
    if (elapsed() > 1.0) { LOG("[t=%.3f] budget exceeded, bail early\n", elapsed()); goto done; }

    // ---------- 3. arenas + engine ----------
    LOG("[t=%.3f] alloc arenas (32MB each)...\n", elapsed());
    RaiArena s_acts  = rai_arena_create(32 * 1024 * 1024);
    RaiArena s_grads = rai_arena_create(32 * 1024 * 1024);
    LOG("[t=%.3f] arenas OK\n", elapsed());

    RaiModelConfig mcfg = {
        .vocab_size = 256, .embedding_length = 64,
        .attention_head_count = 4, .feed_forward_length = 256,
        .block_count = 2, .attention_layer_norm_rms_epsilon = 1e-5f,
        .rope_freq_base = 10000.0f
    };
    RaiAdamWConfig acfg = { .lr=3e-3f, .beta1=0.9f, .beta2=0.999f, .eps=1e-8f, .weight_decay=0.01f };

    LOG("[t=%.3f] create engine...\n", elapsed());
    RaiEngine engine = rai_engine_create(32 * 1024 * 1024, mcfg, acfg);
    int engine_ok = 1;
    LOG("[t=%.3f] engine OK\n", elapsed());

    if (stop) goto cleanup;
    if (elapsed() > 1.2) { LOG("[t=%.3f] budget exceeded\n", elapsed()); goto cleanup; }

    // ---------- 4. ONE forward ----------
    static const char *TEXT =
        "Hi, I am Ali.\nHi Ali! How are you?\nI am fine, thank you.\nGoodbye Ali!\nGoodbye!\n";
    size_t text_len = strlen(TEXT);
    if (text_len > 512) text_len = 512;
    int tokens[512];
    for (size_t i=0; i<text_len; i++) tokens[i] = (unsigned char)TEXT[i];
    size_t sl = text_len - 1;
    int *inputs = tokens;
    int *targets = tokens + 1;

    LOG("[t=%.3f] one forward (seq=%zu)...\n", elapsed(), sl);
    rai_arena_clear(&s_acts);
    double t0 = now_sec();
    RaiModelActs acts = rai_engine_forward(&s_acts, engine, 1, sl, inputs, 0);
    double fwd_time = now_sec() - t0;
    float loss = rai_cross_entropy(acts.output, 1, sl, targets);
    LOG("[t=%.3f] forward: %.4fs loss=%.4f\n", elapsed(), fwd_time, loss);

    // ---------- 5. ONE train step (only if forward was fast enough) ----------
    double train_time = -1;
    if (fwd_time < 0.3 && !stop) {
        LOG("[t=%.3f] one train step...\n", elapsed());
        rai_arena_clear(&s_acts);
        rai_arena_clear(&s_grads);
        t0 = now_sec();
        acts = rai_engine_forward(&s_acts, engine, 1, sl, inputs, 0);
        loss = rai_cross_entropy(acts.output, 1, sl, targets);
        RaiTensor d_logits = rai_cross_entropy_grad(&s_acts, acts.output, 1, sl, targets);
        RaiModel grads = rai_engine_backward(&s_acts, &s_grads, engine, acts, d_logits, 1, sl, inputs, 0);
        rai_engine_step(&engine, grads);
        train_time = now_sec() - t0;
        LOG("[t=%.3f] train step: %.4fs\n", elapsed(), train_time);
    } else {
        LOG("[t=%.3f] skipping train step (fwd=%.3fs)\n", elapsed(), fwd_time);
    }

    // ---------- 6. estimates ----------
    size_t params = 256*64 + 2*(12*64*64 + 3*64*256) + 256*64;
    double scale_10m = 10e6 / (double)params;
    double scale_90m = 90e6 / (double)params;
    LOG("\n=== estimates ===\n");
    LOG("params=%zu\n", params);
    if (train_time > 0) {
        LOG("3000 steps (this model): %.1f min\n", train_time * 3000 / 60);
        LOG("10M model, 3000 steps:   %.1f min\n", train_time * 3000 * scale_10m / 60);
        LOG("90M model, 3000 steps:   %.1f min\n", train_time * 3000 * scale_90m / 60);
    }
    LOG("90M model RAM: ~%.0f MB\n", 90e6 * 4.0 * 3.0 / 1048576.0);

    // ---------- 7. verdict ----------
    LOG("\n=== verdict ===\n");
    if (fwd_time < 0.01) LOG("fwd: 🚀 FAST\n");
    else if (fwd_time < 0.1) LOG("fwd: ✅ OK\n");
    else if (fwd_time < 0.3) LOG("fwd: ⚠️  SLOW\n");
    else LOG("fwd: ❌ TOO SLOW\n");

    if (total_ram < 500) LOG("ram: ❌ too low for big models\n");
    else if (total_ram < 1500) LOG("ram: ⚠️  OK for ~10M, not 90M\n");
    else LOG("ram: ✅ OK for 90M+\n");

cleanup:
    if (engine_ok) rai_engine_destroy(&engine);
    rai_arena_destroy(&s_acts);
    rai_arena_destroy(&s_grads);
done:
    LOG("[t=%.3f] done\n", elapsed());
    if (out) fclose(out);
    return 0;
}