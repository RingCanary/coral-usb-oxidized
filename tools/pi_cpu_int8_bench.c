#include <arm_neon.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *mode;
    size_t m;
    size_t k;
    size_t n;
    int warmup;
    int runs;
} BenchConfig;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int cmp_f64(const void *lhs, const void *rhs) {
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static double percentile_sorted(const double *values, int count, double p) {
    if (count <= 0) {
        return 0.0;
    }
    int idx = (int)(p * count + 0.999999);
    if (idx < 1) {
        idx = 1;
    }
    if (idx > count) {
        idx = count;
    }
    return values[idx - 1];
}

static void fill_i8(int8_t *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        data[i] = (int8_t)((int)(i % 13) - 6);
    }
}

static void *aligned_calloc_i8(size_t len) {
    const size_t alignment = 64;
    const size_t rounded = ((len + alignment - 1) / alignment) * alignment;
    void *ptr = aligned_alloc(alignment, rounded);
    if (!ptr) {
        return NULL;
    }
    memset(ptr, 0, rounded);
    return ptr;
}

static int64_t gemv_kernel(const int8_t *a, const int8_t *x, size_t m, size_t k) {
    int64_t checksum = 0;
#pragma omp parallel for reduction(+ : checksum) schedule(static)
    for (size_t row = 0; row < m; ++row) {
        const int8_t *a_row = a + row * k;
        int32x4_t acc = vdupq_n_s32(0);
        size_t col = 0;
        for (; col + 15 < k; col += 16) {
            const int8x16_t av = vld1q_s8(a_row + col);
            const int8x16_t xv = vld1q_s8(x + col);
            acc = vdotq_s32(acc, av, xv);
        }
        int32_t sum = vaddvq_s32(acc);
        for (; col < k; ++col) {
            sum += (int32_t)a_row[col] * (int32_t)x[col];
        }
        checksum += sum;
    }
    return checksum;
}

static int64_t gemm_kernel(const int8_t *a, const int8_t *bt, size_t m, size_t k, size_t n) {
    int64_t checksum = 0;
#pragma omp parallel for reduction(+ : checksum) schedule(static)
    for (size_t row = 0; row < m; ++row) {
        const int8_t *a_row = a + row * k;
        size_t col = 0;
        for (; col + 3 < n; col += 4) {
            const int8_t *b0 = bt + (col + 0) * k;
            const int8_t *b1 = bt + (col + 1) * k;
            const int8_t *b2 = bt + (col + 2) * k;
            const int8_t *b3 = bt + (col + 3) * k;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            size_t depth = 0;
            for (; depth + 15 < k; depth += 16) {
                const int8x16_t av = vld1q_s8(a_row + depth);
                acc0 = vdotq_s32(acc0, av, vld1q_s8(b0 + depth));
                acc1 = vdotq_s32(acc1, av, vld1q_s8(b1 + depth));
                acc2 = vdotq_s32(acc2, av, vld1q_s8(b2 + depth));
                acc3 = vdotq_s32(acc3, av, vld1q_s8(b3 + depth));
            }
            int32_t sum0 = vaddvq_s32(acc0);
            int32_t sum1 = vaddvq_s32(acc1);
            int32_t sum2 = vaddvq_s32(acc2);
            int32_t sum3 = vaddvq_s32(acc3);
            for (; depth < k; ++depth) {
                const int32_t av = (int32_t)a_row[depth];
                sum0 += av * (int32_t)b0[depth];
                sum1 += av * (int32_t)b1[depth];
                sum2 += av * (int32_t)b2[depth];
                sum3 += av * (int32_t)b3[depth];
            }
            checksum += (int64_t)sum0 + (int64_t)sum1 + (int64_t)sum2 + (int64_t)sum3;
        }
        for (; col < n; ++col) {
            const int8_t *b_row = bt + col * k;
            int32x4_t acc = vdupq_n_s32(0);
            size_t depth = 0;
            for (; depth + 15 < k; depth += 16) {
                acc = vdotq_s32(
                    acc,
                    vld1q_s8(a_row + depth),
                    vld1q_s8(b_row + depth)
                );
            }
            int32_t sum = vaddvq_s32(acc);
            for (; depth < k; ++depth) {
                sum += (int32_t)a_row[depth] * (int32_t)b_row[depth];
            }
            checksum += sum;
        }
    }
    return checksum;
}

static void transpose_b(const int8_t *b, int8_t *bt, size_t k, size_t n) {
    for (size_t row = 0; row < k; ++row) {
        for (size_t col = 0; col < n; ++col) {
            bt[col * k + row] = b[row * n + col];
        }
    }
}

static void usage(const char *program) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s gemv <m> <k> [warmup] [runs]\n", program);
    fprintf(stderr, "  %s gemm <m> <k> <n> [warmup] [runs]\n", program);
}

static int parse_size(const char *text, size_t *value) {
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (!end || *end != '\0') {
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}

static int parse_int(const char *text, int *value) {
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (!end || *end != '\0') {
        return 0;
    }
    *value = (int)parsed;
    return 1;
}

static int parse_args(int argc, char **argv, BenchConfig *cfg) {
    if (argc < 4) {
        return 0;
    }

    memset(cfg, 0, sizeof(*cfg));
    cfg->mode = argv[1];
    cfg->warmup = 2;
    cfg->runs = 7;

    if (strcmp(cfg->mode, "gemv") == 0) {
        if (!parse_size(argv[2], &cfg->m) || !parse_size(argv[3], &cfg->k)) {
            return 0;
        }
        cfg->n = 1;
        if (argc >= 5 && !parse_int(argv[4], &cfg->warmup)) {
            return 0;
        }
        if (argc >= 6 && !parse_int(argv[5], &cfg->runs)) {
            return 0;
        }
        return 1;
    }

    if (strcmp(cfg->mode, "gemm") == 0) {
        if (argc < 5) {
            return 0;
        }
        if (!parse_size(argv[2], &cfg->m) || !parse_size(argv[3], &cfg->k) ||
            !parse_size(argv[4], &cfg->n)) {
            return 0;
        }
        if (argc >= 6 && !parse_int(argv[5], &cfg->warmup)) {
            return 0;
        }
        if (argc >= 7 && !parse_int(argv[6], &cfg->runs)) {
            return 0;
        }
        return 1;
    }

    return 0;
}

int main(int argc, char **argv) {
    BenchConfig cfg;
    if (!parse_args(argc, argv, &cfg) || cfg.m == 0 || cfg.k == 0 || cfg.n == 0 ||
        cfg.warmup < 0 || cfg.runs <= 0) {
        usage(argv[0]);
        return 1;
    }

    const size_t a_len = cfg.m * cfg.k;
    int8_t *a = aligned_calloc_i8(a_len);
    if (!a) {
        perror("aligned_alloc(a)");
        return 1;
    }
    fill_i8(a, a_len);

    int8_t *x = NULL;
    int8_t *b = NULL;
    int8_t *bt = NULL;
    if (strcmp(cfg.mode, "gemv") == 0) {
        x = aligned_calloc_i8(cfg.k);
        if (!x) {
            perror("aligned_alloc(x)");
            free(a);
            return 1;
        }
        fill_i8(x, cfg.k);
    } else {
        const size_t b_len = cfg.k * cfg.n;
        b = aligned_calloc_i8(b_len);
        bt = aligned_calloc_i8(b_len);
        if (!b || !bt) {
            perror("aligned_alloc(b/bt)");
            free(a);
            free(b);
            free(bt);
            return 1;
        }
        fill_i8(b, b_len);
        transpose_b(b, bt, cfg.k, cfg.n);
    }

    double *samples_ms = calloc((size_t)cfg.runs, sizeof(double));
    if (!samples_ms) {
        perror("calloc(samples_ms)");
        free(a);
        free(x);
        free(b);
        free(bt);
        return 1;
    }

    int64_t checksum = 0;
    for (int iter = 0; iter < cfg.warmup + cfg.runs; ++iter) {
        const double start_ms = now_ms();
        checksum = (strcmp(cfg.mode, "gemv") == 0)
            ? gemv_kernel(a, x, cfg.m, cfg.k)
            : gemm_kernel(a, bt, cfg.m, cfg.k, cfg.n);
        const double elapsed_ms = now_ms() - start_ms;
        if (iter >= cfg.warmup) {
            samples_ms[iter - cfg.warmup] = elapsed_ms;
        }
    }

    qsort(samples_ms, (size_t)cfg.runs, sizeof(double), cmp_f64);
    double sum_ms = 0.0;
    for (int i = 0; i < cfg.runs; ++i) {
        sum_ms += samples_ms[i];
    }
    const double mean_ms = sum_ms / (double)cfg.runs;
    const double median_ms = percentile_sorted(samples_ms, cfg.runs, 0.5);
    const double p95_ms = percentile_sorted(samples_ms, cfg.runs, 0.95);
    const double min_ms = samples_ms[0];
    const double max_ms = samples_ms[cfg.runs - 1];
    const double macs = (double)cfg.m * (double)cfg.k * (double)cfg.n;
    const double gmac_per_s_mean = macs / (mean_ms * 1e6);
    const double gmac_per_s_median = macs / (median_ms * 1e6);

    printf(
        "mode=%s m=%zu k=%zu n=%zu warmup=%d runs=%d threads=%d "
        "min_ms=%.3f median_ms=%.3f p95_ms=%.3f max_ms=%.3f mean_ms=%.3f "
        "gmac_per_s_median=%.3f gmac_per_s_mean=%.3f checksum=%lld\n",
        cfg.mode,
        cfg.m,
        cfg.k,
        cfg.n,
        cfg.warmup,
        cfg.runs,
        omp_get_max_threads(),
        min_ms,
        median_ms,
        p95_ms,
        max_ms,
        mean_ms,
        gmac_per_s_median,
        gmac_per_s_mean,
        (long long)checksum
    );
    printf("samples_ms=");
    for (int i = 0; i < cfg.runs; ++i) {
        printf(i == 0 ? "%.3f" : ",%.3f", samples_ms[i]);
    }
    printf("\n");

    free(samples_ms);
    free(a);
    free(x);
    free(b);
    free(bt);
    return 0;
}
