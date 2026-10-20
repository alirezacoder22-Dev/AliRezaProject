#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdint.h>
#include <signal.h>

#define RAI_IMPLEMENTATION
#include "rriftt_ai.h"

#define MAX_LEN      2048
#define MAX_FILE     (16 * 1024 * 1024)
#define MAX_QA       4000
#define BOS          0x02
#define EOS          0x03
#define QSEP         0x04
#define VOCAB        512
#define WORD_BASE    0x100

// ============ 1.3M params ============
#define D_MODEL      128
#define N_HEADS      8
#define N_BLOCKS     3
#define FFN_LEN      512

// ============ word codes ============
typedef struct { const char *word; size_t len; int code; } WordCode;
static WordCode g_wc[] = {
    {"سلام",0,0x100},{"من",0,0x101},{"هستم",0,0x102},{"خداحافظ",0,0x103},
    {"خوبم",0,0x104},{"چطوری",0,0x105},{"چه",0,0x106},{"کار",0,0x107},
    {"می‌کنی",0,0x108},{"دارم",0,0x109},{"کتاب",0,0x10A},{"می‌خونم",0,0x10B},
    {"کتابی",0,0x10C},{"یک",0,0x10D},{"داستان",0,0x10E},{"درباره‌ی",0,0x10F},
    {"اسم",0,0x1C0},{"شما",0,0x1C1},{"چیه",0,0x1C2},{"آمدید",0,0x1C3},
    {"خوش",0,0x1C4},{"آمدم",0,0x1C5},
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

typedef struct {
    int q_tokens[MAX_LEN]; int q_len;
    int a_tokens[MAX_LEN]; int a_len;
    char q_text[256]; char a_text[256];
} QAPair;

static QAPair g_qa[MAX_QA];
static int g_qa_count = 0;
static volatile sig_atomic_t g_stop = 0;

static int g_color = 0;
#define CR  "\033[0m"
#define CG  "\033[32m"
#define CY  "\033[33m"
#define CC  "\033[36m"
#define CRD "\033[31m"
#define CD  "\033[2m"
#define CB  "\033[34m"
#define CBOLD "\033[1m"
static void C(const char*s){ if(g_color) fputs(s,stdout); }

static double wall(void){ struct timeval tv; gettimeofday(&tv,0); return tv.tv_sec+tv.tv_usec*1e-6; }
static void init_wc(void){ for(size_t i=0;i<NWC;i++) g_wc[i].len=strlen(g_wc[i].word); }
static void on_sig(int s){ (void)s; g_stop = 1; }

// ============ encode / print ============
static size_t encode(const char *in, size_t n, int *out, size_t cap) {
    if (cap < 4) return 0;
    size_t o = 0, i = 0;
    out[o++] = BOS;
    while (i < n && o + 8 < cap) {
        if (in[i] == '\n') { out[o++] = EOS; out[o++] = BOS; i++; continue; }
        int hit = 0;
        for (size_t w = 0; w < NWC; w++) {
            if (i + g_wc[w].len <= n && memcmp(in+i, g_wc[w].word, g_wc[w].len) == 0) {
                out[o++] = g_wc[w].code; i += g_wc[w].len; hit = 1; break;
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
        printf("[?%d]", t); return;
    }
    if (t > 0 && t < 256) putchar((char)t);
}

// ============ parse ============
static int parse_qa(const char *raw, size_t raw_len) {
    g_qa_count = 0;
    size_t i = 0;
    while (i < raw_len && g_qa_count < MAX_QA) {
        while (i < raw_len && (raw[i]=='\n'||raw[i]=='\r'||raw[i]==' '||raw[i]=='\t')) i++;
        if (i+3 < raw_len && raw[i]=='#' && raw[i+1]=='#' && raw[i+2]=='#' && raw[i+3]=='#') { i+=4; continue; }
        if (i+3 < raw_len && raw[i]=='=' && raw[i+1]=='=' && raw[i+2]=='=' && raw[i+3]=='=') { i+=4; continue; }
        if (i >= raw_len) break;
        size_t s1 = i;
        while (i < raw_len && raw[i] != '\n') i++;
        size_t l1 = i - s1;
        if (i < raw_len) i++;
        while (l1 > 0 && (raw[s1+l1-1]=='\r'||raw[s1+l1-1]==' ')) l1--;
        if (l1 == 0) continue;
        while (i < raw_len && (raw[i]=='\n'||raw[i]=='\r'||raw[i]==' '||raw[i]=='\t')) i++;
        if (i+3 < raw_len && raw[i]=='#' && raw[i+1]=='#' && raw[i+2]=='#' && raw[i+3]=='#') { i+=4; continue; }
        if (i+3 < raw_len && raw[i]=='=' && raw[i+1]=='=' && raw[i+2]=='=' && raw[i+3]=='=') { i+=4; continue; }
        if (i >= raw_len) break;
        size_t s2 = i;
        while (i < raw_len && raw[i] != '\n') i++;
        size_t l2 = i - s2;
        if (i < raw_len) i++;
        while (l2 > 0 && (raw[s2+l2-1]=='\r'||raw[s2+l2-1]==' ')) l2--;
        if (l2 == 0) continue;
        QAPair *qa = &g_qa[g_qa_count];
        qa->q_len = (int)encode(raw + s1, l1, qa->q_tokens, MAX_LEN);
        qa->a_len = (int)encode(raw + s2, l2, qa->a_tokens, MAX_LEN);
        size_t c1 = l1<255?l1:255; memcpy(qa->q_text, raw+s1, c1); qa->q_text[c1]=0;
        size_t c2 = l2<255?l2:255; memcpy(qa->a_text, raw+s2, c2); qa->a_text[c2]=0;
        g_qa_count++;
    }
    return g_qa_count;
}

static int argmax_f(float *l, size_t V) {
    int b = 0; float bv = l[0];
    for (size_t i = 1; i < V; i++) if (l[i] > bv) { bv = l[i]; b = (int)i; }
    return b;
}

// ============ BENCHMARK (پنجمارک) ============
typedef struct {
    int name_correct, name_total;
    int exact_match;
    int token_correct, token_total;
} EvalResult;

static EvalResult evaluate(RaiEngine *e) {
    EvalResult r = {0};
    for (int idx = 0; idx < g_qa_count; idx++) {
        QAPair *qa = &g_qa[idx];
        int buf[MAX_LEN]; int len = 0;
        for (int i = 0; i < qa->q_len; i++) buf[len++] = qa->q_tokens[i];
        buf[len++] = QSEP; buf[len++] = BOS;
        int all_ok = 1, name_found = 0;
        for (int i = 0; i < qa->a_len; i++) {
            int expected = qa->a_tokens[i];
            if (expected == BOS) continue;
            RaiArena s = rai_arena_create(16*1024*1024);
            RaiModelActs a = rai_engine_forward(&s, *e, 1, len, buf, 0);
            size_t V = e->model_config.vocab_size;
            float *lg = a.output.data + (len-1)*V;
            int pred = argmax_f(lg, V);
            rai_arena_destroy(&s);
            if (expected == EOS) {
                r.token_total++;
                if (pred == EOS) r.token_correct++; else all_ok = 0;
                break;
            }
            r.token_total++;
            if (pred == expected) r.token_correct++; else all_ok = 0;
            if (expected >= 0x110 && expected <= 0x1BC && !name_found) {
                name_found = 1; r.name_total++;
                if (pred == expected) r.name_correct++;
            }
            buf[len++] = expected;
        }
        if (all_ok) r.exact_match++;
    }
    return r;
}

static void eval_print(EvalResult r, const char *title, FILE *out) {
    FILE *f = out ? out : stdout;
    if (g_color && !out) { fputs(CY, f); }
    fprintf(f, "\n  ╔══ %s ══╗\n", title);
    if (g_color && !out) { fputs(CR, f); }
    fprintf(f, "  │ اسم درست:   %d/%d (%.1f%%)\n",
            r.name_correct, r.name_total, 100.0*r.name_correct/(r.name_total?r.name_total:1));
    fprintf(f, "  │ توکن درست:  %d/%d (%.1f%%)\n",
            r.token_correct, r.token_total, 100.0*r.token_correct/(r.token_total?r.token_total:1));
    fprintf(f, "  │ جمله کامل:  %d/%d (%.1f%%)\n",
            r.exact_match, g_qa_count, 100.0*r.exact_match/(g_qa_count?g_qa_count:1));
    if (g_color && !out) { fputs(CY, f); }
    fprintf(f, "  ╚══════════════════════════════╝\n\n");
    if (g_color && !out) { fputs(CR, f); }
}

// ============ DEBUG one ============
static void debug_one(RaiEngine *e, int idx, int *ok_count) {
    QAPair *qa = &g_qa[idx];
    printf("  "); C(CB); printf("Q: "); C(CR); printf("%s\n", qa->q_text);
    printf("  "); C(CG); printf("A: "); C(CR); printf("%s\n", qa->a_text);
    printf("  "); C(CC); printf("M: "); C(CR);
    int buf[MAX_LEN]; int len = 0;
    for (int i = 0; i < qa->q_len; i++) buf[len++] = qa->q_tokens[i];
    buf[len++] = QSEP; buf[len++] = BOS;
    int correct = 0, total = 0;
    for (int i = 0; i < qa->a_len; i++) {
        int expected = qa->a_tokens[i];
        if (expected == BOS) continue;
        RaiArena s = rai_arena_create(16*1024*1024);
        RaiModelActs a = rai_engine_forward(&s, *e, 1, len, buf, 0);
        size_t V = e->model_config.vocab_size;
        float *lg = a.output.data + (len-1)*V;
        int pred = argmax_f(lg, V);
        rai_arena_destroy(&s);
        total++;
        if (expected == EOS) {
            if (pred == EOS) { correct++; C(CG); printf("."); C(CR); }
            break;
        }
        if (pred == expected) { correct++; C(CG); print_tok(pred); C(CR); }
        else { C(CRD); printf("("); print_tok(pred); printf("→"); print_tok(expected); printf(")"); C(CR); }
        buf[len++] = expected;
    }
    printf("  [%d/%d", correct, total);
    if (correct == total) { C(CG); printf(" ✓"); C(CR); }
    else if (correct*2 >= total) { C(CY); printf(" ⚠"); C(CR); }
    else { C(CRD); printf(" ✗"); C(CR); }
    printf("]\n\n");
    if (ok_count) *ok_count += (correct == total);
}

static void debug_report(RaiEngine *e, int step, float loss) {
    C(CC);
    printf("\n  ╔══ DEBUG @ step %d  loss=%.4f ══╗\n", step, loss);
    C(CR); printf("\n");
    int n_show = g_qa_count < 5 ? g_qa_count : 5;
    for (int i = 0; i < n_show; i++) debug_one(e, i, NULL);
}

// ============ checkpoint ============
#define CKPT_MAGIC "RFBIG001"
#define CKPT_VER   1

static void wm(FILE *f, RaiModel *m) {
    size_t n = RAI_MODEL_TENSOR_COUNT;
    fwrite(&n, sizeof(size_t), 1, f);
    for (size_t i = 0; i < n; i++) {
        RaiTensor t = m->array[i];
        fwrite(&t.rank, sizeof(size_t), 1, f);
        fwrite(t.dims, sizeof(size_t), t.rank, f);
        fwrite(t.data, sizeof(float), t.count, f);
    }
}
static int rm(FILE *f, RaiModel *m) {
    size_t n;
    if (fread(&n, sizeof(size_t), 1, f) != 1 || n != RAI_MODEL_TENSOR_COUNT) return 0;
    for (size_t i = 0; i < n; i++) {
        RaiTensor t = m->array[i]; size_t rank;
        if (fread(&rank, sizeof(size_t), 1, f) != 1 || rank != t.rank) return 0;
        if (fread(t.dims, sizeof(size_t), rank, f) != rank) return 0;
        if (fread(t.data, sizeof(float), t.count, f) != t.count) return 0;
    }
    return 1;
}
static int ckpt_save(RaiEngine *e, const char *path, size_t step, float loss) {
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    fwrite(CKPT_MAGIC, 1, 8, f);
    uint32_t v = CKPT_VER; fwrite(&v, sizeof(uint32_t), 1, f);
    fwrite(&e->model_config, sizeof(RaiModelConfig), 1, f);
    fwrite(&e->adamw_config, sizeof(RaiAdamWConfig), 1, f);
    fwrite(&e->adamw.time, sizeof(size_t), 1, f);
    fwrite(&step, sizeof(size_t), 1, f);
    fwrite(&loss, sizeof(float), 1, f);
    wm(f, &e->weights); wm(f, &e->adamw.means); wm(f, &e->adamw.uvars);
    fclose(f); return 1;
}
static int ckpt_load(RaiEngine *e, const char *path, size_t *step, float *loss) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    char m[8];
    if (fread(m, 1, 8, f) != 8 || memcmp(m, CKPT_MAGIC, 8) != 0) { fclose(f); return 0; }
    uint32_t v;
    if (fread(&v, sizeof(uint32_t), 1, f) != 1 || v != CKPT_VER) { fclose(f); return 0; }
    RaiModelConfig mc; RaiAdamWConfig ac;
    if (fread(&mc, sizeof(RaiModelConfig), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&ac, sizeof(RaiAdamWConfig), 1, f) != 1) { fclose(f); return 0; }
    if (memcmp(&mc, &e->model_config, sizeof(RaiModelConfig)) != 0) { fclose(f); return 0; }
    if (fread(&e->adamw.time, sizeof(size_t), 1, f) != 1) { fclose(f); return 0; }
    if (fread(step, sizeof(size_t), 1, f) != 1) { fclose(f); return 0; }
    if (fread(loss, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (!rm(f, &e->weights)) { fclose(f); return 0; }
    if (!rm(f, &e->adamw.means)) { fclose(f); return 0; }
    if (!rm(f, &e->adamw.uvars)) { fclose(f); return 0; }
    fclose(f); return 1;
}

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
    int steps = 1500, batch_show = 10, ckpt_every = 200;
    int cpu_pct = 100, log_sec = 300;
    const char *path = "data_fa.txt";
    const char *ckpt_last = "ckpt_last.bin";
    const char *log_path = "train.log";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-train")) train = 1;
        else if (!strcmp(argv[i], "-resume")) resume = 1;
        else if (!strcmp(argv[i], "-debug")) debug = 1;
        else if (!strcmp(argv[i], "-color")) g_color = 1;
        else if (!strcmp(argv[i], "-steps") && i+1<argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-show") && i+1<argc) batch_show = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ckpt-every") && i+1<argc) ckpt_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-cpu") && i+1<argc) cpu_pct = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-log") && i+1<argc) log_path = argv[++i];
        else if (!strcmp(argv[i], "-log-sec") && i+1<argc) log_sec = atoi(argv[++i]);
        else if (argv[i][0] != '-') {
            int n = atoi(argv[i]);
            if (n > 100) steps = n; else path = argv[i];
        }
    }
    if (cpu_pct < 10) cpu_pct = 10;
    if (cpu_pct > 100) cpu_pct = 100;

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    RAI_SRAND(42); srand(42); init_wc();

    RaiModelConfig mcfg = {
        .vocab_size = VOCAB,
        .embedding_length = D_MODEL,
        .attention_head_count = N_HEADS,
        .feed_forward_length = FFN_LEN,
        .block_count = N_BLOCKS,
        .attention_layer_norm_rms_epsilon = 1e-5f,
        .rope_freq_base = 10000.0f
    };
    RaiAdamWConfig acfg = { .lr = 1e-3f, .beta1 = 0.9f, .beta2 = 0.999f,
                            .eps = 1e-8f, .weight_decay = 0.0f };

    size_t params = VOCAB*D_MODEL + N_BLOCKS*(12ULL*D_MODEL*D_MODEL + 3ULL*D_MODEL*FFN_LEN) + VOCAB*D_MODEL;
    C(CBOLD); C(CC);
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  BIG BUILD — %s                  ║\n", resume ? "RESUME" : "FRESH");
    printf("╚══════════════════════════════════════════╝\n\n");
    C(CR);
    printf("  Parameters:  %zu (~%.2fM)\n", params, params/1e6);
    printf("  Config:      d=%d, heads=%d, blocks=%d, ffn=%d\n",
           D_MODEL, N_HEADS, N_BLOCKS, FFN_LEN);
    printf("  CPU limit:   %d%%\n", cpu_pct);
    printf("  Log every:   %d sec → %s\n\n", log_sec, log_path);
    fflush(stdout);

    RaiArena s_acts = rai_arena_create(128 * 1024 * 1024);
    RaiArena s_grads = rai_arena_create(128 * 1024 * 1024);
    RaiEngine engine = rai_engine_create(128 * 1024 * 1024, mcfg, acfg);

    size_t raw_len = 0;
    char *raw = read_file(path, &raw_len);
    if (!raw) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    int n_qa = parse_qa(raw, raw_len);
    free(raw);
    if (n_qa == 0) { fprintf(stderr, "no Q&A\n"); return 1; }

    printf("  Data: %d Q&A pairs\n\n", n_qa);
    fflush(stdout);

    int start_step = 0;
    float best_loss = 1e9f;
    if (resume) {
        size_t st; float ls;
        if (ckpt_load(&engine, ckpt_last, &st, &ls)) {
            start_step = (int)st; best_loss = ls;
            printf("  Resumed: step=%zu loss=%.4f\n\n", st, ls);
        }
    }

    // ===== initial benchmark =====
    C(CY); printf("=== پنجمارک اولیه ===\n"); C(CR);
    EvalResult init_r = evaluate(&engine);
    eval_print(init_r, "step 0", NULL);

    int win_tokens[MAX_LEN];
    double t0 = wall();
    double t_last_log = t0;
    float last_loss = 0;
    int64_t total_sleep_us = 0;
    double total_work_s = 0;

    for (int i = 0; i < steps && !g_stop; i++) {
        int step = start_step + i + 1;
        double t_step_start = wall();

        // window build
        int qa_start = (start_step + i) % n_qa;
        int wlen = 0, n_added = 0;
        for (int k = 0; k < n_qa && n_added < 6; k++) {
            int qi = (qa_start + k) % n_qa;
            int total = g_qa[qi].q_len + 1 + g_qa[qi].a_len;
            if (wlen + total > MAX_LEN - 4) break;
            memcpy(win_tokens + wlen, g_qa[qi].q_tokens, g_qa[qi].q_len * sizeof(int));
            wlen += g_qa[qi].q_len;
            win_tokens[wlen++] = QSEP;
            memcpy(win_tokens + wlen, g_qa[qi].a_tokens, g_qa[qi].a_len * sizeof(int));
            wlen += g_qa[qi].a_len;
            n_added++;
        }
        if (wlen < 8) continue;
        int sl = wlen - 1;

        rai_arena_clear(&s_acts); rai_arena_clear(&s_grads);
        RaiModelActs a = rai_engine_forward(&s_acts, engine, 1, sl, win_tokens, 0);
        float loss = rai_cross_entropy(a.output, 1, sl, win_tokens + 1);
        RaiTensor dl = rai_cross_entropy_grad(&s_acts, a.output, 1, sl, win_tokens + 1);
        RaiModel g = rai_engine_backward(&s_acts, &s_grads, engine, a, dl, 1, sl, win_tokens, 0);
        rai_engine_step(&engine, g);
        last_loss = loss;
        if (loss < best_loss) best_loss = loss;

        double t_step = wall() - t_step_start;
        total_work_s += t_step;

        // CPU limiting
        if (cpu_pct < 100) {
            double sleep_s = t_step * (100.0 - cpu_pct) / cpu_pct;
            if (sleep_s > 0.001) {
                usleep((useconds_t)(sleep_s * 1e6));
                total_sleep_us += (int64_t)(sleep_s * 1e6);
            }
        }

        // progress
        if ((i+1) % batch_show == 0 || i == steps-1) {
            double el = wall() - t0;
            double ps = el / (i+1);
            double eta = ps * (steps - i - 1);
            printf("\r  [%4d/%d] loss=%.4f  %.0fms/step  eta=%02d:%02d   ",
                   step, start_step+steps, loss, ps*1000,
                   (int)(eta/60), (int)eta%60);
            fflush(stdout);
        }

        // checkpoint
        if ((i+1) % ckpt_every == 0) {
            ckpt_save(&engine, ckpt_last, step, loss);
        }

        // debug console
        if (debug && (step % 100) == 0) {
            printf("\n"); debug_report(&engine, step, loss);
        }

        // 5-minute log
        if (wall() - t_last_log >= log_sec || i == steps-1) {
            printf("\n");
            FILE *lf = fopen(log_path, "a");
            if (lf) {
                time_t now = time(NULL);
                char tbuf[64];
                strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
                fprintf(lf, "\n═══════════════════════════════════════════\n");
                fprintf(lf, " [%s]  step=%d/%d  loss=%.4f  best=%.4f\n",
                        tbuf, step, start_step+steps, loss, best_loss);
                fprintf(lf, " elapsed=%.0fs  work=%.0fs  sleep=%.0fs  avg=%.0fms/step\n",
                        wall()-t0, total_work_s, total_sleep_us/1e6,
                        (wall()-t0)*1000/(i+1));
                fprintf(lf, "═══════════════════════════════════════════\n");
                fflush(lf);
                // benchmark in log
                EvalResult r = evaluate(&engine);
                eval_print(r, "پنجمارک", lf);
                // samples in log
                for (int s = 0; s < 3 && s < n_qa; s++) {
                    QAPair *qa = &g_qa[s];
                    fprintf(lf, " Q: %s\n A: %s\n M: ", qa->q_text, qa->a_text);
                    int buf[MAX_LEN]; int len = 0;
                    for (int j = 0; j < qa->q_len; j++) buf[len++] = qa->q_tokens[j];
                    buf[len++] = QSEP; buf[len++] = BOS;
                    for (int j = 0; j < qa->a_len; j++) {
                        int exp = qa->a_tokens[j];
                        if (exp == BOS) continue;
                        if (exp == EOS) break;
                        RaiArena ss = rai_arena_create(16*1024*1024);
                        RaiModelActs aa = rai_engine_forward(&ss, engine, 1, len, buf, 0);
                        size_t V = engine.model_config.vocab_size;
                        float *lg = aa.output.data + (len-1)*V;
                        int pr = argmax_f(lg, V);
                        rai_arena_destroy(&ss);
                        // write token to log
                        if (pr >= WORD_BASE) {
                            for (size_t w = 0; w < NWC; w++)
                                if (g_wc[w].code == pr) { fprintf(lf, "%s", g_wc[w].word); break; }
                        } else if (pr > 0 && pr < 256) {
                            fputc((char)pr, lf);
                        }
                        if (pr != exp) fprintf(lf, "{%s}",
                            pr >= WORD_BASE ? "?" : "X");
                        buf[len++] = exp;
                    }
                    fprintf(lf, "\n\n");
                }
                fclose(lf);
                C(CD); printf("  [log saved: %s]\n", log_path); C(CR);
            }
            t_last_log = wall();
        }
    }

    printf("\n\n");
    if (g_stop) {
        C(CY); printf("  ⚠ interrupted, saving...\n"); C(CR);
        ckpt_save(&engine, ckpt_last, start_step + steps, last_loss);
    }

    double total = wall() - t0;
    C(CG);
    printf("\n  ✅ done: %.0fs  best=%.4f  last=%.4f\n", total, best_loss, last_loss);
    C(CR); printf("\n");

    // final benchmark
    C(CY); printf("=== پنجمارک نهایی ===\n"); C(CR);
    EvalResult fr = evaluate(&engine);
    eval_print(fr, "final", NULL);

    ckpt_save(&engine, ckpt_last, start_step + steps, last_loss);

    if (debug) {
        debug_report(&engine, start_step + steps, last_loss);
    }

    rai_engine_destroy(&engine);
    rai_arena_destroy(&s_acts);
    rai_arena_destroy(&s_grads);
    return 0;
}