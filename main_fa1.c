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

#define MAX_LEN      1024
#define MAX_FILE     (16 * 1024 * 1024)
#define MAX_QA       4000
#define BOS          0x02
#define EOS          0x03
#define QSEP         0x04
#define VOCAB        512
#define WORD_BASE    0x100

// ============ word codes ============
typedef struct { const char *word; size_t len; int code; } WordCode;
static WordCode g_wc[] = {
    {"سلام",0,0x100},{"من",0,0x101},{"هستم",0,0x102},{"خداحافظ",0,0x103},
    {"خوبم",0,0x104},{"چطوری",0,0x105},{"چه",0,0x106},{"کار",0,0x107},
    {"می‌کنی",0,0x108},{"دارم",0,0x109},{"کتاب",0,0x10A},{"می‌خونم",0,0x10B},
    {"کتابی",0,0x10C},{"یک",0,0x10D},{"داستان",0,0x10E},{"درباره‌ی",0,0x10F},
    {"اسم",0,0x1C0},{"شما",0,0x1C1},{"چیه",0,0x1C2},{"آمدید",0,0x1C3},
    {"خوش",0,0x1C4},{"آمدم",0,0x1C5},
    // names
    {"علی",0,0x110},{"سارا",0,0x111},{"محمد",0,0x112},{"مریم",0,0x113},
    {"رضا",0,0x114},{"مهدی",0,0x115},{"زهرا",0,0x116},{"حسین",0,0x117},
    {"فاطمه",0,0x118},{"امیر",0,0x119},{"نرگس",0,0x11A},{"بهرام",0,0x11B},
    {"لیلا",0,0x11C},{"کاوه",0,0x11D},{"شیرین",0,0x11E},{"آرش",0,0x11F},
    {"پریسا",0,0x120},{"بابک",0,0x121},{"نازنین",0,0x122},{"سعید",0,0x123},
    {"الهام",0,0x124},{"کیان",0,0x125},{"مینا",0,0x126},{"هومن",0,0x127},
    {"شقایق",0,0x128},{"فرهاد",0,0x129},{"سما",0,0x12A},{"پارسا",0,0x12B},
    {"یاسمن",0,0x12C},{"رامین",0,0x12D},{"هستی",0,0x12E},{"نوید",0,0x12F},
    {"ترانه",0,0x130},{"بهنام",0,0x131},{"رویا",0,0x132},{"کامران",0,0x133},
    {"سحر",0,0x134},{"امین",0,0x135},{"نگار",0,0x136},{"پویا",0,0x137},
    {"دنیا",0,0x138},{"سینا",0,0x139},{"مهسا",0,0x13A},{"عرفان",0,0x13B},
    {"تینا",0,0x13C},{"بردیا",0,0x13D},{"آیدا",0,0x13E},{"کیانا",0,0x13F},
    {"شایان",0,0x140},{"الناز",0,0x141},{"آرمان",0,0x142},{"پگاه",0,0x143},
    {"سامان",0,0x144},{"ندا",0,0x145},{"پوریا",0,0x146},{"ملیکا",0,0x147},
    {"آبتین",0,0x148},{"آرزو",0,0x149},{"آرمین",0,0x14A},{"آزاده",0,0x14B},
    {"آسا",0,0x14C},{"آشا",0,0x14D},{"آفرین",0,0x14E},{"آوا",0,0x14F},
    {"اتابک",0,0x150},{"ارسلان",0,0x151},{"اردلان",0,0x152},{"ارمیا",0,0x153},
    {"اسد",0,0x154},{"اشکان",0,0x155},{"اصغر",0,0x156},{"افسانه",0,0x157},
    {"اکرم",0,0x158},{"الیا",0,0x159},{"امید",0,0x15A},{"انوشا",0,0x15B},
    {"اهورا",0,0x15C},{"ایران",0,0x15D},{"ایلیا",0,0x15E},{"باران",0,0x15F},
    {"بهار",0,0x160},{"بیژن",0,0x161},{"پرستو",0,0x162},{"پریا",0,0x163},
    {"پیام",0,0x164},{"تارا",0,0x165},{"تورج",0,0x166},{"ثنا",0,0x167},
    {"جاوید",0,0x168},{"جمشید",0,0x169},{"جوانه",0,0x16A},{"حامی",0,0x16B},
    {"حورا",0,0x16C},{"حنانه",0,0x16D},{"خسرو",0,0x16E},{"خورشید",0,0x16F},
    {"داریوش",0,0x170},{"دانش",0,0x171},{"درسا",0,0x172},{"دلارام",0,0x173},
    {"رها",0,0x174},{"رهام",0,0x175},{"روزبه",0,0x176},{"رایان",0,0x177},
    {"زری",0,0x178},{"ژاله",0,0x179},{"سام",0,0x17A},{"سبا",0,0x17B},
    {"سپیده",0,0x17C},{"ستایش",0,0x17D},{"سروش",0,0x17E},{"سعادت",0,0x17F},
    {"سمانه",0,0x180},{"سوگند",0,0x181},{"سها",0,0x182},{"شادی",0,0x183},
    {"شاهد",0,0x184},{"شمیم",0,0x185},{"شهاب",0,0x186},{"شهرام",0,0x187},
    {"شهریار",0,0x188},{"شیوا",0,0x189},{"صبا",0,0x18A},{"صدرا",0,0x18B},
    {"صنم",0,0x18C},{"طاهره",0,0x18D},{"طلا",0,0x18E},{"عاطفه",0,0x18F},
    {"عدنان",0,0x190},{"علیا",0,0x191},{"غزل",0,0x192},{"فرانک",0,0x193},
    {"فرشاد",0,0x194},{"فرزانه",0,0x195},{"فربد",0,0x196},{"فرشته",0,0x197},
    {"فریبا",0,0x198},{"فیروز",0,0x199},{"قاسم",0,0x19A},{"کامیار",0,0x19B},
    {"کتایون",0,0x19C},{"کیانوش",0,0x19D},{"گلنار",0,0x19E},{"گلاره",0,0x19F},
    {"مانی",0,0x1A0},{"ماهور",0,0x1A1},{"مازیار",0,0x1A2},{"مرجان",0,0x1A3},
    {"مروا",0,0x1A4},{"مسیحا",0,0x1A5},{"مظاهر",0,0x1A6},{"معصومه",0,0x1A7},
    {"مهران",0,0x1A8},{"مهرداد",0,0x1A9},{"مهرناز",0,0x1AA},{"میثم",0,0x1AB},
    {"میرا",0,0x1AC},{"میلاد",0,0x1AD},{"نازی",0,0x1AE},{"ناهید",0,0x1AF},
    {"نسیم",0,0x1B0},{"نیایش",0,0x1B1},{"نیما",0,0x1B2},{"هلیا",0,0x1B3},
    {"همتا",0,0x1B4},{"هیوا",0,0x1B5},{"وانا",0,0x1B6},{"وحید",0,0x1B7},
    {"هانیه",0,0x1B8},{"یلدا",0,0x1B9},{"یوسف",0,0x1BA},
    {"محسن",0,0x1BB},{"علیرضا",0,0x1BC},
};
#define NWC (sizeof(g_wc)/sizeof(g_wc[0]))

// ============ Q&A pair ============
typedef struct {
    int q_tokens[MAX_LEN];
    int q_len;
    int a_tokens[MAX_LEN];
    int a_len;
    char q_text[256];
    char a_text[256];
} QAPair;

static QAPair g_qa[MAX_QA];
static int g_qa_count = 0;

static int g_color = 0;
#define CR  "\033[0m"
#define CG  "\033[32m"
#define CY  "\033[33m"
#define CC  "\033[36m"
#define CRD "\033[31m"
#define CD  "\033[2m"
#define CB  "\033[34m"
static void C(const char*s){ if(g_color) fputs(s,stdout); }

static double wall(void){ struct timeval tv; gettimeofday(&tv,0); return tv.tv_sec+tv.tv_usec*1e-6; }
static void init_wc(void){ for(size_t i=0;i<NWC;i++) g_wc[i].len=strlen(g_wc[i].word); }

// ============ encode ============
static size_t encode(const char *in, size_t n, int *out, size_t cap) {
    if (cap < 4) return 0;
    size_t o = 0, i = 0;
    out[o++] = BOS;
    while (i < n && o + 8 < cap) {
        if (in[i] == '\n') { out[o++] = EOS; out[o++] = BOS; i++; continue; }
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
    if (t == BOS || t == EOS || t == 0 || t == QSEP) return;
    if (t >= WORD_BASE) {
        for (size_t w = 0; w < NWC; w++)
            if (g_wc[w].code == t) { printf("%s", g_wc[w].word); return; }
        printf("[?%d]", t);
        return;
    }
    if (t > 0 && t < 256) putchar((char)t);
}

// ============ parse ============
static int parse_qa(const char *raw, size_t raw_len) {
    g_qa_count = 0;
    size_t i = 0;
    while (i < raw_len && g_qa_count < MAX_QA) {
        while (i < raw_len && (raw[i]=='\n' || raw[i]=='\r' || raw[i]==' ' || raw[i]=='\t')) i++;
        if (i + 3 < raw_len && raw[i]=='#' && raw[i+1]=='#' && raw[i+2]=='#' && raw[i+3]=='#') { i += 4; continue; }
        if (i + 3 < raw_len && raw[i]=='=' && raw[i+1]=='=' && raw[i+2]=='=' && raw[i+3]=='=') { i += 4; continue; }
        if (i >= raw_len) break;
        
        size_t s1 = i;
        while (i < raw_len && raw[i] != '\n') i++;
        size_t len1 = i - s1;
        if (i < raw_len) i++;
        while (len1 > 0 && (raw[s1+len1-1]=='\r' || raw[s1+len1-1]==' ')) len1--;
        if (len1 == 0) continue;
        
        while (i < raw_len && (raw[i]=='\n' || raw[i]=='\r' || raw[i]==' ' || raw[i]=='\t')) i++;
        if (i + 3 < raw_len && raw[i]=='#' && raw[i+1]=='#' && raw[i+2]=='#' && raw[i+3]=='#') { i += 4; continue; }
        if (i + 3 < raw_len && raw[i]=='=' && raw[i+1]=='=' && raw[i+2]=='=' && raw[i+3]=='=') { i += 4; continue; }
        if (i >= raw_len) break;
        
        size_t s2 = i;
        while (i < raw_len && raw[i] != '\n') i++;
        size_t len2 = i - s2;
        if (i < raw_len) i++;
        while (len2 > 0 && (raw[s2+len2-1]=='\r' || raw[s2+len2-1]==' ')) len2--;
        if (len2 == 0) continue;
        
        QAPair *qa = &g_qa[g_qa_count];
        qa->q_len = (int)encode(raw + s1, len1, qa->q_tokens, MAX_LEN);
        qa->a_len = (int)encode(raw + s2, len2, qa->a_tokens, MAX_LEN);
        
        size_t c1 = len1 < 255 ? len1 : 255;
        memcpy(qa->q_text, raw + s1, c1); qa->q_text[c1] = 0;
        size_t c2 = len2 < 255 ? len2 : 255;
        memcpy(qa->a_text, raw + s2, c2); qa->a_text[c2] = 0;
        
        g_qa_count++;
    }
    return g_qa_count;
}

// ============ helpers ============
static int argmax_f(float *l, size_t V) {
    int b = 0; float bv = l[0];
    for (size_t i = 1; i < V; i++) if (l[i] > bv) { bv = l[i]; b = (int)i; }
    return b;
}

// ============ DEBUG one ============
static void debug_one(RaiEngine *e, int idx, int *ok_count) {
    QAPair *qa = &g_qa[idx];
    
    printf("  ");
    C(CB); printf("Q: "); C(CR);
    printf("%s\n", qa->q_text);
    printf("  ");
    C(CG); printf("A(واقعی): "); C(CR);
    printf("%s\n", qa->a_text);
    printf("  ");
    C(CC); printf("A(مدل):   "); C(CR);
    
    // Build prompt = Q tokens + QSEP + BOS
    int buf[MAX_LEN];
    int len = 0;
    for (int i = 0; i < qa->q_len; i++) buf[len++] = qa->q_tokens[i];
    buf[len++] = QSEP;
    buf[len++] = BOS;
    
    int correct = 0, total = 0;
    for (int i = 0; i < qa->a_len; i++) {
        int expected = qa->a_tokens[i];
        if (expected == BOS) continue;
        
        RaiArena s = rai_arena_create(8*1024*1024);
        RaiModelActs a = rai_engine_forward(&s, *e, 1, len, buf, 0);
        size_t V = e->model_config.vocab_size;
        float *lg = a.output.data + (len-1)*V;
        int pred = argmax_f(lg, V);
        rai_arena_destroy(&s);
        
        total++;
        
        if (expected == EOS) {
            if (pred == EOS) { correct++; C(CG); printf("."); C(CR); }
            else { C(CRD); printf("[?EOS]"); C(CR); }
            break;
        }
        
        if (pred == expected) {
            correct++;
            C(CG); print_tok(pred); C(CR);
        } else {
            C(CRD); printf("("); print_tok(pred); printf("→"); print_tok(expected); printf(")"); C(CR);
        }
        buf[len++] = expected;
    }
    
    printf("  [%d/%d", correct, total);
    if (correct == total) { C(CG); printf(" ✓"); C(CR); }
    else if (correct * 2 >= total) { C(CY); printf(" ⚠"); C(CR); }
    else { C(CRD); printf(" ✗"); C(CR); }
    printf("]\n\n");
    
    if (ok_count) *ok_count += (correct == total);
}

static void debug_report(RaiEngine *e, int step, float loss) {
    C(CC);
    printf("\n  ╔══════════════════════════════════════╗\n");
    printf("  ║  DEBUG @ step %-6d  loss=%.4f    ║\n", step, loss);
    printf("  ╚══════════════════════════════════════╝\n");
    C(CR);
    printf("\n");
    
    int n_show = g_qa_count < 5 ? g_qa_count : 5;
    int ok_count = 0;
    for (int i = 0; i < n_show; i++) debug_one(e, i, &ok_count);
    
    C(CC);
    printf("  ────────────────────────────────────\n");
    printf("  نتیجه: %d/%d مکالمه کامل درست", ok_count, n_show);
    if (ok_count == n_show) { C(CG); printf(" 🏆"); C(CR); }
    else if (ok_count * 2 >= n_show) { C(CY); printf(" ⚠️"); C(CR); }
    else { C(CRD); printf(" ❌"); C(CR); }
    C(CC);
    printf("\n  ────────────────────────────────────\n\n");
    C(CR);
}

// ============ generate ============
static void gen(RaiEngine *e, const char *seed, int n_new) {
    int buf[MAX_LEN];
    int len = 0;
    buf[len++] = BOS;
    int tmp[MAX_LEN];
    size_t seed_enc = encode(seed, strlen(seed), tmp, MAX_LEN);
    for (size_t i = 0; i < seed_enc && len < MAX_LEN-1; i++) {
        if (tmp[i] == BOS || tmp[i] == EOS) continue;
        buf[len++] = tmp[i];
    }
    buf[len++] = QSEP;
    buf[len++] = BOS;
    C(CD); printf("[seed] "); C(CR);
    printf("%s\n", seed);
    C(CC); printf("[out ] "); C(CR);
    for (int i = 0; i < n_new; i++) {
        if (len + 1 >= MAX_LEN) break;
        RaiArena s = rai_arena_create(8 * 1024 * 1024);
        RaiModelActs a = rai_engine_forward(&s, *e, 1, len, buf, 0);
        size_t V = e->model_config.vocab_size;
        float *lg = a.output.data + (len-1) * V;
        int t = argmax_f(lg, V);
        rai_arena_destroy(&s);
        if (t == EOS) { printf(" [EOS]"); break; }
        if (t == BOS || t == QSEP || t == 0) continue;
        print_tok(t);
        buf[len++] = t;
    }
    printf("\n\n");
}

// ============ checkpoint ============
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
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    fwrite(CKPT_MAGIC, 1, 8, f);
    uint32_t ver = CKPT_VER; fwrite(&ver, sizeof(uint32_t), 1, f);
    fwrite(&e->model_config, sizeof(RaiModelConfig), 1, f);
    fwrite(&e->adamw_config, sizeof(RaiAdamWConfig), 1, f);
    fwrite(&e->adamw.time, sizeof(size_t), 1, f);
    fwrite(&step, sizeof(size_t), 1, f);
    fwrite(&loss, sizeof(float), 1, f);
    write_model(f, &e->weights);
    write_model(f, &e->adamw.means);
    write_model(f, &e->adamw.uvars);
    fclose(f); return 1;
}
static int ckpt_load(RaiEngine *e, const char *path, size_t *step, float *loss) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
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
    fclose(f); return 1;
}

// ============ file ============
static char* read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > MAX_FILE) { fclose(f); return NULL; }
    char *b = malloc(sz + 1); if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    b[sz] = 0; fclose(f); *n = sz; return b;
}

// ============ main ============
int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int train = 0, resume = 0, debug = 0;
    int steps = 800, batch_show = 10, ckpt_every = 200;
    const char *path = "data_fa.txt";
    const char *ckpt_last = "ckpt_last.bin";
    const char *ckpt_best = "ckpt_best.bin";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-train")) train = 1;
        else if (!strcmp(argv[i], "-resume")) resume = 1;
        else if (!strcmp(argv[i], "-debug")) debug = 1;
        else if (!strcmp(argv[i], "-color")) g_color = 1;
        else if (!strcmp(argv[i], "-steps") && i+1<argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-show") && i+1<argc) batch_show = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ckpt-every") && i+1<argc) ckpt_every = atoi(argv[++i]);
        else if (argv[i][0] != '-') {
            int n = atoi(argv[i]);
            if (n > 100) steps = n; else path = argv[i];
        }
    }
    
    RAI_SRAND(42); srand(42); init_wc();
    
    RaiModelConfig mcfg = {
        .vocab_size = VOCAB, .embedding_length = 64, .attention_head_count = 4,
        .feed_forward_length = 256, .block_count = 2,
        .attention_layer_norm_rms_epsilon = 1e-5f, .rope_freq_base = 10000.0f
    };
    RaiAdamWConfig acfg = { .lr = 2e-3f, .beta1 = 0.9f, .beta2 = 0.999f,
                            .eps = 1e-8f, .weight_decay = 0.0f };
    RaiArena s_acts = rai_arena_create(64 * 1024 * 1024);
    RaiArena s_grads = rai_arena_create(64 * 1024 * 1024);
    RaiEngine engine = rai_engine_create(64 * 1024 * 1024, mcfg, acfg);
    
    if (!train) {
        size_t st; float ls;
        if (!ckpt_load(&engine, ckpt_best, &st, &ls)) {
            if (!ckpt_load(&engine, ckpt_last, &st, &ls)) {
                fprintf(stderr, "no checkpoint\n"); return 1;
            }
        }
        size_t raw_len; char *raw = read_file(path, &raw_len);
        if (raw) { parse_qa(raw, raw_len); free(raw); }
        C(CG); printf("loaded (step=%zu, loss=%.4f)\n", st, ls); C(CR);
        if (g_qa_count > 0) debug_report(&engine, (int)st, ls);
        printf("=== samples ===\n\n");
        gen(&engine, "سلام من ", 20);
        rai_engine_destroy(&engine);
        rai_arena_destroy(&s_acts); rai_arena_destroy(&s_grads);
        return 0;
    }
    
    size_t raw_len = 0;
    char *raw = read_file(path, &raw_len);
    if (!raw) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    int n_qa = parse_qa(raw, raw_len);
    free(raw);
    if (n_qa == 0) { fprintf(stderr, "no Q&A parsed\n"); return 1; }
    
    printf("=== data ===\n");
    printf("  file:    %s\n", path);
    printf("  raw:     %zu bytes\n", raw_len);
    printf("  Q&A:     %d جفت\n\n", n_qa);
    
    if (debug) {
        C(CY); printf("=== قبل از آموزش ===\n"); C(CR);
        debug_report(&engine, 0, 0.0f);
    }
    
    int start_step = 0;
    if (resume) {
        size_t st; float ls;
        if (ckpt_load(&engine, ckpt_last, &st, &ls)) {
            start_step = (int)st;
            printf("  resumed: step=%zu loss=%.4f\n\n", st, ls);
        }
    }
    
    printf("=== training ===\n\n");
    double t0 = wall();
    float best_loss = 1e9f;
    float last_loss = 0;
    
    int win_tokens[MAX_LEN];
    
    for (int i = 0; i < steps; i++) {
        int step = start_step + i + 1;
        
        // Build window from COMPLETE Q&A pairs, with QSEP
        int qa_start = (start_step + i) % n_qa;
        int wlen = 0;
        int n_added = 0;
        for (int k = 0; k < n_qa && n_added < 4; k++) {
            int qi = (qa_start + k) % n_qa;
            int total = g_qa[qi].q_len + 1 + g_qa[qi].a_len;  // +1 for QSEP
            if (wlen + total > MAX_LEN - 2) break;
            memcpy(win_tokens + wlen, g_qa[qi].q_tokens, g_qa[qi].q_len * sizeof(int));
            wlen += g_qa[qi].q_len;
            win_tokens[wlen++] = QSEP;
            memcpy(win_tokens + wlen, g_qa[qi].a_tokens, g_qa[qi].a_len * sizeof(int));
            wlen += g_qa[qi].a_len;
            n_added++;
        }
        if (wlen < 8) continue;
        int sl = wlen - 1;
        
        rai_arena_clear(&s_acts);
        rai_arena_clear(&s_grads);
        RaiModelActs a = rai_engine_forward(&s_acts, engine, 1, sl, win_tokens, 0);
        float loss = rai_cross_entropy(a.output, 1, sl, win_tokens + 1);
        RaiTensor dl = rai_cross_entropy_grad(&s_acts, a.output, 1, sl, win_tokens + 1);
        RaiModel g = rai_engine_backward(&s_acts, &s_grads, engine, a, dl, 1, sl, win_tokens, 0);
        rai_engine_step(&engine, g);
        last_loss = loss;
        if (loss < best_loss) best_loss = loss;
        
        if ((i+1) % batch_show == 0 || i == steps-1) {
            double el = wall() - t0;
            double ps = el / (i+1);
            double eta = ps * (steps - i - 1);
            printf("\r  [%4d/%d] loss=%.4f  %.0fms/step  eta=%02d:%02d",
                   step, start_step+steps, loss, ps*1000,
                   (int)(eta/60), (int)eta%60);
        }
        
        if ((i+1) % ckpt_every == 0) {
            ckpt_save(&engine, ckpt_last, step, loss);
        }
        
        if (debug && (step % 100) == 0) {
            printf("\n");
            debug_report(&engine, step, loss);
        }
    }
    
    double total = wall() - t0;
    printf("\n\n");
    C(CG); printf("  done: "); C(CR);
    printf("%.1fs  best=%.4f  last=%.4f\n\n", total, best_loss, last_loss);
    ckpt_save(&engine, ckpt_last, start_step+steps, last_loss);
    
    if (debug) {
        C(CY); printf("=== بعد از آموزش ===\n"); C(CR);
        debug_report(&engine, start_step + steps, last_loss);
    }
    
    rai_engine_destroy(&engine);
    rai_arena_destroy(&s_acts);
    rai_arena_destroy(&s_grads);
    return 0;
}