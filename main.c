#define RAI_IMPLEMENTATION
#include "rriftt_ai.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_CTX 512

static const char *TRAIN_TEXT =
"Hi, I am Ali.\n"
"Hi Ali! How are you?\n"
"I am fine, thank you.\n"
"Goodbye Ali!\n"
"Goodbye!\n";

static int argmax(float *logits, size_t V) {
    int best = 0;
    for (size_t i = 1; i < V; i++) if (logits[i] > logits[best]) best = (int)i;
    return best;
}

static int sample_next(RaiEngine *eng, int *ctx, int n, int greedy) {
    RaiArena scratch = rai_arena_create(16 * 1024 * 1024);
    RaiModelActs acts = rai_engine_forward(&scratch, *eng, 1, n, ctx, 0);
    size_t V = eng->model_config.vocab_size;
    float *logits = acts.output.data + (n - 1) * V;
    int tok;
    if (greedy) {
        tok = argmax(logits, V);
    } else {
        float mx = logits[0];
        for (size_t i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
        float sum = 0;
        for (size_t i = 0; i < V; i++) { logits[i] = expf((logits[i]-mx)/0.3f); sum += logits[i]; }
        for (size_t i = 0; i < V; i++) logits[i] /= sum;
        float r = (float)rand() / (float)RAND_MAX, acc = 0;
        tok = (int)V - 1;
        for (size_t i = 0; i < V; i++) { acc += logits[i]; if (r < acc) { tok = (int)i; break; } }
    }
    rai_arena_destroy(&scratch);
    return tok;
}

static void generate(RaiEngine *eng, int seed, int n_new, int greedy) {
    int buf[MAX_CTX]; int len = 1; buf[0] = seed;
    putchar((char)seed);
    for (int i = 0; i < n_new; i++) {
        int tok = sample_next(eng, buf, len, greedy);
        putchar((char)tok);
        if (len < MAX_CTX) buf[len++] = tok;
        else { memmove(buf, buf+1, (MAX_CTX-1)*sizeof(int)); buf[MAX_CTX-1] = tok; }
    }
    putchar('\n');
}

static void save_model(RaiEngine *eng, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { printf("cannot open %s\n", path); return; }
    size_t n = RAI_MODEL_TENSOR_COUNT;
    fwrite(&n, sizeof(size_t), 1, f);
    for (size_t i = 0; i < n; i++) {
        RaiTensor t = eng->weights.array[i];
        fwrite(&t.rank, sizeof(size_t), 1, f);
        fwrite(t.dims, sizeof(size_t), t.rank, f);
        fwrite(t.data, sizeof(float), t.count, f);
    }
    fclose(f);
    printf("saved %s\n", path);
}

int main(void) {
    RAI_SRAND(42);
    //srand(42);
    srand((unsigned int)time(NULL));  // به جای srand(42)
    size_t text_len = strlen(TRAIN_TEXT);
    printf("train chars: %zu\n", text_len);

    int *tokens = malloc(text_len * sizeof(int));
    for (size_t i = 0; i < text_len; i++) tokens[i] = (unsigned char)TRAIN_TEXT[i];

    size_t sl = text_len - 1;
    if (sl > MAX_CTX) sl = MAX_CTX;
    printf("seq_len: %zu\n", sl);

    int *inputs  = tokens;
    int *targets = tokens + 1;

    RaiModelConfig mcfg = {
        .vocab_size = 256,
        .embedding_length = 64,
        .attention_head_count = 4,
        .feed_forward_length = 256,
        .block_count = 2,
        .attention_layer_norm_rms_epsilon = 1e-5f,
        .rope_freq_base = 10000.0f
    };
    RaiAdamWConfig acfg = { .lr=3e-3f, .beta1=0.9f, .beta2=0.999f, .eps=1e-8f, .weight_decay=0.01f };

    RaiArena s_acts  = rai_arena_create(64 * 1024 * 1024);
    RaiArena s_grads = rai_arena_create(64 * 1024 * 1024);
    RaiEngine engine = rai_engine_create(64 * 1024 * 1024, mcfg, acfg);

    int total = 3000;
    for (int step = 0; step < total; step++) {
        rai_arena_clear(&s_acts);
        rai_arena_clear(&s_grads);
        RaiModelActs acts = rai_engine_forward(&s_acts, engine, 1, sl, inputs, 0);
        float loss = rai_cross_entropy(acts.output, 1, sl, targets);
        RaiTensor d_logits = rai_cross_entropy_grad(&s_acts, acts.output, 1, sl, targets);
        RaiModel grads = rai_engine_backward(&s_acts, &s_grads, engine, acts, d_logits, 1, sl, inputs, 0);
        rai_engine_step(&engine, grads);
        if ((step+1) % 200 == 0) printf("step %4d  loss %.4f\n", step+1, loss);
    }

    rai_arena_clear(&s_acts);
    RaiModelActs acts = rai_engine_forward(&s_acts, engine, 1, sl, inputs, 0);
    int correct = 0;
    for (size_t i = 0; i < sl; i++)
        if (argmax(acts.output.data + i * 256, 256) == targets[i]) correct++;
    printf("\n>>> training accuracy: %d/%zu (%.1f%%)\n\n", correct, sl, 100.0*correct/sl);

    printf("--- greedy ---\n");
    generate(&engine, TRAIN_TEXT[0], 120, 1);

    save_model(&engine, "model.bin");

    rai_engine_destroy(&engine);
    rai_arena_destroy(&s_acts);
    rai_arena_destroy(&s_grads);
    free(tokens);
    return 0;
}