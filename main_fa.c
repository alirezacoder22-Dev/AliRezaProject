#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdint.h>

#define RAI_IMPLEMENTATION
#include "rriftt_ai.h"

#define MAX_LEN      512
#define MAX_FILE     (16 * 1024 * 1024)
#define BOS          0x02
#define EOS          0x03
#define VOCAB        512
#define WORD_BASE    0x100   // 256

// ============ word codes (بالای 256، بدون conflict با UTF-8) ============
typedef struct { const char *word; size_t len; int code; } WordCode;
static WordCode g_wc[] = {
    // names
    {"علیرضا",0,0x100},{"محمد",0,0x101},{"فاطمه",0,0x102},{"مریم",0,0x103},
    {"سارا",0,0x104},{"زهرا",0,0x105},{"حسین",0,0x106},{"رضا",0,0x107},
    {"علی",0,0x108},{"مهدی",0,0x109},{"حسن",0,0x10A},{"یاسین",0,0x10B},
    {"امید",0,0x10C},{"محسن",0,0x10D},{"آرش",0,0x10E},{"بابک",0,0x10F},
    // common
    {"سلام",0,0x110},{"خداحافظ",0,0x111},
    {"هستم",0,0x112},{"هستی",0,0x113},{"هست",0,0x114},
    {"من",0,0x115},{"تو",0,0x116},{"او",0,0x117},
    {"ما",0,0x118},{"شما",0,0x119},{"آنها",0,0x11A},
    {"خوبم",0,0x11B},{"خوبی",0,0x11C},{"خوب",0,0x11D},{"ممنون",0,0x11E},
    // verbs
    {"می‌روم",0,0x120},{"می‌روی",0,0x121},{"می‌رود",0,0x122},
    {"رفتم",0,0x123},{"خواهم",0,0x124},{"رفت",0,0x125},
    {"است",0,0x126},{"بود",0,0x127},{"نیست",0,0x128},
    {"می‌کنم",0,0x129},{"می‌کنی",0,0x12A},{"می‌کند",0,0x12B},
    {"دارم",0,0x12C},{"داری",0,0x12D},{"دارد",0,0x12E},
    // questions
    {"چطوری",0,0x130},{"چرا",0,0x131},{"کجا",0,0x132},{"چه",0,0x133},{"کی",0,0x134},
    {"چقدر",0,0x135},{"کدام",0,0x136},{"آیا",0,0x137},{"بله",0,0x138},{"نه",0,0x139},
    // objects
    {"کتاب",0,0x140},{"خانه",0,0x141},{"سیب",0,0x142},{"آب",0,0x143},
    {"نان",0,0x144},{"گربه",0,0x145},{"سگ",0,0x146},{"پرنده",0,0x147},
    {"داستان",0,0x148},{"سفر",0,0x149},{"کار",0,0x14A},
    {"می‌خونم",0,0x14B},{"می‌خونی",0,0x14C},{"می‌خونه",0,0x14D},
    {"دوستش",0,0x14E},{"بیشتر",0,0x14F},
    // adjectives
    {"جالب",0,0x150},{"زیبا",0,0x151},{"بزرگ",0,0x152},{"کوچک",0,0x153},
    {"دوست",0,0x154},{"بله",0,0x155},{"که",0,0x156},{"و",0,0x157},
};
#define NWC (sizeof(g_wc)/sizeof(g_wc[0]))

static int g_color = 0;
#define CR  "\033[0m"
#define CG  "\033[32m"
#define CY  "\033[33m"
#define CC  "\033[36m"
#define CRD "\033[31m"
#define CD  "\033[2m"
static void C(const char*s){ if(g_color) fputs(s,stdout); }

static double wall(void){ struct timeval tv; gettimeofday(&tv,0); return tv.tv_sec+tv.tv_usec*1e-6; }
static void init_wc(void){ for(size_t i=0;i<NWC;i++) g_wc[i].len=strlen(g_wc[i].word); }

// ============ encode: خروجی int، نه byte ============
static size_t encode(const char *in, size_t n, int *out, size_t cap) {
    if (cap < 4) return 0;
    size_t o = 0, i = 0;
    out[o++] = BOS;
    while (i < n && o + 8 < cap) {
        if (in[i] == '\n') {
            out[o++] = EOS;
            out[o++] = BOS;
            i++;
            continue;
        }
        int hit = 0;
        for (size_t w = 0; w < NWC; w++) {
            if (i + g_wc[w].len <= n && memcmp(in+i, g_wc[w].word, g_wc[w].len) == 0) {
                out[o++] = g_wc[w].code;
                i += g_wc[w].len;
                hit = 1;
                break;
            }
        }
        if (!hit) out[o++] = (int)(unsigned char)in[i++];
    }
    if (o + 1 < cap) out[o++] = EOS;
    return o;
}

static void print_tok(int t) {
    if (t == BOS || t == EOS || t == 0) return;
    if (t >= WORD_BASE) {
        for (size_t w = 0; w < NWC; w++)
            if (g_wc[w].code == t) { printf("%s", g_wc[w].word); return; }
        printf("[?%d]", t);
        return;
    }
    if (t > 0 && t < 256) putchar((char)t);
}

// ============ sampling ============
static int sample_topk(float *logits, size_t V, float temp, int k) {
    if (k > 32) k = 32;
    if (k < 1) k = 1;
    if (V < (size_t)k) k = (int)V;
    
    int ti[32]; float tv[32];
    for (int i = 0; i < k; i++) { ti[i] = 0; tv[i] = -INFINITY; }
    for (size_t i = 0; i < V; i++) {
        float v = logits[i];
        if (v > tv[k-1]) {
            int p = k-1;
            while (p > 0 && v > tv[p-1]) { tv[p] = tv[p-1]; ti[p] = ti[p-1]; p--; }
            tv[p] = v;
            ti[p] = (int)i;
        }
    }
    if (tv[0] == -INFINITY) return 0;
    
    float mx = tv[0], pr[32], sum = 0;
    for (int i = 0; i < k; i++) { pr[i] = expf((tv[i]-mx)/temp); sum += pr[i]; }
    if (sum <= 0) return ti[0];
    float r = ((float)rand()/RAND_MAX) * sum, acc = 0;
    for (int i = 0; i < k; i++) { acc += pr[i]; if (r < acc) return ti[i]; }
    return ti[0];
}

static int argmax_f(float *l, size_t V) {
    int b = 0; float bv = l[0];
    for (size_t i = 1; i < V; i++) if (l[i] > bv) { bv = l[i]; b = (int)i; }
    return b;
}

// ============ generate ============
static void gen(RaiEngine *e, const char *seed, int n_new, float temp, int topk) {
    int buf[MAX_LEN];
    int len = 0;
    buf[len++] = BOS;

    int tmp[MAX_LEN];
    size_t seed_enc = encode(seed, strlen(seed), tmp, MAX_LEN);
    for (size_t i = 0; i < seed_enc && len < MAX_LEN-1; i++) {
        if (tmp[i] == BOS || tmp[i] == EOS) continue;
        buf[len++] = tmp[i];
    }

    C(CD); printf("[seed] "); C(CR);
    printf("%s\n", seed);
    C(CC); printf("[out ] "); C(CR);
    for (int i = 0; i < len; i++) print_tok(buf[i]);

    for (int i = 0; i < n_new; i++) {
        if (len + 1 >= MAX_LEN) break;
        RaiArena s = rai_arena_create(8 * 1024 * 1024);
        RaiModelActs a = rai_engine_forward(&s, *e, 1, len, buf, 0);
        size_t V = e->model_config.vocab_size;
        float *lg = a.output.data + (len-1) * V;
        int t = (temp <= 0.01f) ? argmax_f(lg, V) : sample_topk(lg, V, temp, topk);
        rai_arena_destroy(&s);
        if (t == EOS) { printf(" [EOS]"); break; }
        if (t == BOS || t == 0) continue;
        print_tok(t);
        buf[len++] = t;
    }
    printf("\n\n");
}

// ============ checkpoint (weights + AdamW) ============
#define CKPT_MAGIC "RFAICK01"
#define CKPT_VER   1

static void write_model(FILE *f, RaiModel *m) {
    size_t n = RAI_MODEL_TENSOR_COUNT;
    fwrite(&n, sizeof(size_t), 1, f);
    for (size_t i = 0; i < n; i++) {
        RaiTensor t = m->array[i];
        fwrite(&t.rank, sizeof(size_t), 1, f);
        fwrite(t.dims, sizeof(size_t), t.rank, f);
        fwrite(t.data, sizeof(float), t.count, f);
    }
}
static int read_model(FILE *f, RaiModel *m) {
    size_t n;
    if (fread(&n, sizeof(size_t), 1, f) != 1 || n != RAI_MODEL_TENSOR_COUNT) return 0;
    for (size_t i = 0; i < n; i++) {
        RaiTensor t = m->array[i];
        size_t rank;
        if (fread(&rank, sizeof(size_t), 1, f) != 1 || rank != t.rank) return 0;
        if (fread(t.dims, sizeof(size_t), rank, f) != rank) return 0;
        if (fread(t.data, sizeof(float), t.count, f) != t.count) return 0;
    }
    return 1;
}

static int ckpt_save(RaiEngine *e, const char *path, size_t step, float loss) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite(CKPT_MAGIC, 1, 8, f);
    uint32_t ver = CKPT_VER;
    fwrite(&ver, sizeof(uint32_t), 1, f);
    fwrite(&e->model_config, sizeof(RaiModelConfig), 1, f);
    fwrite(&e->adamw_config, sizeof(RaiAdamWConfig), 1, f);
    fwrite(&e->adamw.time, sizeof(size_t), 1, f);
    fwrite(&step, sizeof(size_t), 1, f);
    fwrite(&loss, sizeof(float), 1, f);
    write_model(f, &e->weights);
    write_model(f, &e->adamw.means);
    write_model(f, &e->adamw.uvars);
    fclose(f);
    return 1;
}
static int ckpt_load(RaiEngine *e, const char *path, size_t *step, float *loss) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, CKPT_MAGIC, 8) != 0) { fclose(f); return 0; }
    uint32_t ver;
    if (fread(&ver, sizeof(uint32_t), 1, f) != 1 || ver != CKPT_VER) { fclose(f); return 0; }
    RaiModelConfig mc; RaiAdamWConfig ac;
    if (fread(&mc, sizeof(RaiModelConfig), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&ac, sizeof(RaiAdamWConfig), 1, f) != 1) { fclose(f); return 0; }
    if (memcmp(&mc, &e->model_config, sizeof(RaiModelConfig)) != 0) { fclose(f); return 0; }
    if (fread(&e->adamw.time, sizeof(size_t), 1, f) != 1) { fclose(f); return 0; }
    if (fread(step, sizeof(size_t), 1, f) != 1) { fclose(f); return 0; }
    if (fread(loss, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (!read_model(f, &e->weights)) { fclose(f); return 0; }
    if (!read_model(f, &e->adamw.means)) { fclose(f); return 0; }
    if (!read_model(f, &e->adamw.uvars)) { fclose(f); return 0; }
    fclose(f);
    return 1;
}

// ============ file ============
static char* read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > MAX_FILE) { fclose(f); return NULL; }
    char *b = malloc(sz + 1);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    b[sz] = 0; fclose(f); *n = sz; return b;
}

// ============ main ============
int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int train = 0, resume = 0, steps = 800, batch_show = 10, window = 128, ckpt_every = 300;
    const char *path = "data_fa.txt";
    const char *ckpt_last = "ckpt_last.bin";
    const char *ckpt_best = "ckpt_best.bin";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-train")) train = 1;
        else if (!strcmp(argv[i], "-resume")) resume = 1;
        else if (!strcmp(argv[i], "-color")) g_color = 1;
        else if (!strcmp(argv[i], "-nocolor")) g_color = 0;
        else if (!strcmp(argv[i], "-steps") && i+1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-window") && i+1 < argc) window = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-show") && i+1 < argc) batch_show = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ckpt-every") && i+1 < argc) ckpt_every = atoi(argv[++i]);
        else if (argv[i][0] != '-') {
            int n = atoi(argv[i]);
            if (n > 100) steps = n; else path = argv[i];
        }
    }
    if (steps < 1) steps = 1;
    if (window < 8) window = 8;
    if (batch_show < 1) batch_show = 1;
    if (ckpt_every < 10) ckpt_every = 10;

    RAI_SRAND(42); srand(42); init_wc();

    RaiModelConfig mcfg = {
        .vocab_size = VOCAB, .embedding_length = 48, .attention_head_count = 4,
        .feed_forward_length = 192, .block_count = 1,
        .attention_layer_norm_rms_epsilon = 1e-5f, .rope_freq_base = 10000.0f
    };
    RaiAdamWConfig acfg = { .lr = 5e-3f, .beta1 = 0.9f, .beta2 = 0.999f,
                            .eps = 1e-8f, .weight_decay = 0.01f };
    RaiArena s_acts = rai_arena_create(16 * 1024 * 1024);
    RaiArena s_grads = rai_arena_create(16 * 1024 * 1024);
    RaiEngine engine = rai_engine_create(16 * 1024 * 1024, mcfg, acfg);

    // ============ inference ============
    if (!train) {
        size_t st; float ls;
        if (!ckpt_load(&engine, ckpt_best, &st, &ls)) {
            if (!ckpt_load(&engine, ckpt_last, &st, &ls)) {
                fprintf(stderr, "no checkpoint found\nrun -train first\n");
                return 1;
            }
        }
        C(CG); printf("loaded checkpoint (step=%zu, loss=%.4f)\n\n", st, ls); C(CR);
        printf("=== sampling ===\n\n");
        C(CD); printf("greedy:\n"); C(CR);
        gen(&engine, "من ", 40, 0, 1);
        gen(&engine, "سلام ", 40, 0, 1);
        C(CD); printf("temp=0.7 top-k=10:\n"); C(CR);
        gen(&engine, "من ", 40, 0.7f, 10);
        gen(&engine, "سلام ", 40, 0.7f, 10);
        C(CD); printf("unseen name (بابک):\n"); C(CR);
        gen(&engine, "من ", 40, 0.7f, 10);
        gen(&engine, "سلام بابک. ", 40, 0.7f, 10);
        rai_engine_destroy(&engine);
        rai_arena_destroy(&s_acts);
        rai_arena_destroy(&s_grads);
        return 0;
    }

    // ============ training ============
    size_t raw_len = 0;
    char *raw = read_file(path, &raw_len);
    if (!raw) { fprintf(stderr, "cannot read %s\n", path); return 1; }

    size_t cap = raw_len + 1024;
    int *enc = malloc(cap * sizeof(int));
    if (!enc) { fprintf(stderr, "oom\n"); return 1; }
    size_t enc_len = encode(raw, raw_len, enc, cap);
    free(raw);
    if (enc_len < 4) { fprintf(stderr, "data too small\n"); free(enc); return 1; }

    printf("=== data ===\n");
    printf("  file:    %s\n", path);
    printf("  raw:     %zu bytes\n", raw_len);
    printf("  encoded: %zu tokens (%.1f%% saved)\n",
           enc_len, 100.0 * (1.0 - (double)enc_len / raw_len));

    if (enc_len < (size_t)window + 2) {
        window = (int)enc_len - 2;
        if (window < 8) window = 8;
        printf("  window:  shrunk to %d\n", window);
    }

    // resume
    int start_step = 0;
    float best_loss = 1e9f;
    if (resume) {
        size_t st; float ls;
        if (ckpt_load(&engine, ckpt_last, &st, &ls)) {
            start_step = (int)st;
            printf("  resumed: step=%zu loss=%.4f\n", st, ls);
        }
        size_t st2; float ls2;
        if (ckpt_load(&engine, ckpt_best, &st2, &ls2)) {
            // load best into a separate engine check is complex; just track best
            best_loss = ls2;
            // re-load last (this may have overwritten)
            if (ckpt_load(&engine, ckpt_last, &st, &ls)) {
                start_step = (int)st;
            }
        }
    }

    printf("  steps:   %d (from %d to %d)\n", steps, start_step, start_step + steps);
    printf("  window:  %d\n\n", window);

    size_t max_start = (enc_len > (size_t)window + 1) ? (enc_len - window - 1) : 0;

    printf("=== training ===\n\n");
    double t0 = wall();
    int stop_cnt = 0;
    int stopped_early = 0;
    float last_loss = 0;

    for (int i = 0; i < steps; i++) {
        int step = start_step + i + 1;
        size_t start = (max_start > 0) ? (size_t)(rand() % max_start) : 0;
        int *in = enc + start;
        int *tg = enc + start + 1;

        rai_arena_clear(&s_acts);
        rai_arena_clear(&s_grads);
        RaiModelActs a = rai_engine_forward(&s_acts, engine, 1, window, in, 0);
        float loss = rai_cross_entropy(a.output, 1, window, tg);
        RaiTensor dl = rai_cross_entropy_grad(&s_acts, a.output, 1, window, tg);
        RaiModel g = rai_engine_backward(&s_acts, &s_grads, engine, a, dl, 1, window, in, 0);
        rai_engine_step(&engine, g);
        last_loss = loss;

        if (loss < best_loss) best_loss = loss;

        if (loss < 0.05f) stop_cnt++; else stop_cnt = 0;

        if ((i + 1) % batch_show == 0 || i == steps - 1) {
            double el = wall() - t0;
            double ps = el / (i + 1);
            double eta = ps * (steps - i - 1);
            printf("\r  [%4d/%d] loss=%.4f  %.0fms/step  elapsed=%.1fs  eta=%02d:%02d",
                   step, start_step + steps, loss, ps * 1000, el,
                   (int)(eta / 60), (int)eta % 60);
        }

        // checkpoint دوره‌ای
        if ((i + 1) % ckpt_every == 0) {
            ckpt_save(&engine, ckpt_last, step, loss);
        }

        // اگه رکورد زد → best هم ذخیره
        if (loss <= best_loss) {
            ckpt_save(&engine, ckpt_best, step, loss);
        }

        if (stop_cnt >= 50) { stopped_early = 1; break; }
    }

    double total = wall() - t0;
    printf("\n\n");
    C(CG); printf("  done: "); C(CR);
    printf("%.1fs  best loss=%.4f  last=%.4f", total, best_loss, last_loss);
    if (stopped_early) { C(CY); printf("  (early stop)"); C(CR); }
    printf("\n\n");

    // ذخیره نهایی
    ckpt_save(&engine, ckpt_last, start_step + steps, last_loss);

    printf("=== samples ===\n\n");
    C(CD); printf("greedy:\n"); C(CR);
    gen(&engine, "من ", 40, 0, 1);
    gen(&engine, "سلام ", 40, 0, 1);
    C(CD); printf("temp=0.7 top-k=10:\n"); C(CR);
    gen(&engine, "من ", 40, 0.7f, 10);
    gen(&engine, "سلام ", 40, 0.7f, 10);
    C(CD); printf("unseen (بابک):\n"); C(CR);
    gen(&engine, "من ", 40, 0.7f, 10);
    gen(&engine, "سلام بابک. ", 40, 0.7f, 10);

    rai_engine_destroy(&engine);
    rai_arena_destroy(&s_acts);
    rai_arena_destroy(&s_grads);
    free(enc);
    return 0;
}