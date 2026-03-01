#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <libusb-1.0/libusb.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/*
 * Minimal SHA-256 implementation adapted from public-domain style references.
 * Purpose: deterministic payload fingerprints for libusb OUT submissions.
 */
typedef struct {
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
} sha256_ctx;

#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT((x), 2) ^ ROTRIGHT((x), 13) ^ ROTRIGHT((x), 22))
#define EP1(x) (ROTRIGHT((x), 6) ^ ROTRIGHT((x), 11) ^ ROTRIGHT((x), 25))
#define SIG0(x) (ROTRIGHT((x), 7) ^ ROTRIGHT((x), 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT((x), 17) ^ ROTRIGHT((x), 19) ^ ((x) >> 10))

static const uint32_t k_sha256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static void sha256_transform(sha256_ctx *ctx, const uint8_t data[]) {
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    for (i = 0, j = 0; i < 16; ++i, j += 4) {
        m[i] = ((uint32_t)data[j] << 24) | ((uint32_t)data[j + 1] << 16) |
               ((uint32_t)data[j + 2] << 8) | ((uint32_t)data[j + 3]);
    }
    for (; i < 64; ++i) {
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + k_sha256[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void sha256_init(sha256_ctx *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

static void sha256_update(sha256_ctx *ctx, const uint8_t data[], size_t len) {
    uint32_t i;
    for (i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

static void sha256_final(sha256_ctx *ctx, uint8_t hash[]) {
    uint32_t i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) {
            ctx->data[i++] = 0x00;
        }
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) {
            ctx->data[i++] = 0x00;
        }
        sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;
    ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16;
    ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32;
    ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48;
    ctx->data[56] = ctx->bitlen >> 56;
    sha256_transform(ctx, ctx->data);

    for (i = 0; i < 4; ++i) {
        hash[i] = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 4] = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 8] = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
    }
}

static void sha256_hex(const unsigned char *data, size_t len, char out_hex[65]) {
    static const char *k_hex = "0123456789abcdef";
    uint8_t digest[32];
    sha256_ctx ctx;
    sha256_init(&ctx);
    if (data != NULL && len > 0) {
        sha256_update(&ctx, data, len);
    }
    sha256_final(&ctx, digest);
    for (int i = 0; i < 32; ++i) {
        out_hex[2 * i] = k_hex[(digest[i] >> 4) & 0xF];
        out_hex[2 * i + 1] = k_hex[digest[i] & 0xF];
    }
    out_hex[64] = '\0';
}

static void bytes_hex_prefix(const unsigned char *data, size_t len, size_t take, char *out, size_t out_sz) {
    static const char *k_hex = "0123456789abcdef";
    if (out_sz == 0) {
        return;
    }
    if (data == NULL || len == 0) {
        out[0] = '\0';
        return;
    }
    size_t n = take < len ? take : len;
    size_t need = n * 2 + 1;
    if (need > out_sz) {
        n = (out_sz - 1) / 2;
    }
    for (size_t i = 0; i < n; ++i) {
        out[2 * i] = k_hex[(data[i] >> 4) & 0xF];
        out[2 * i + 1] = k_hex[data[i] & 0xF];
    }
    out[2 * n] = '\0';
}

static void bytes_hex_suffix(const unsigned char *data, size_t len, size_t take, char *out, size_t out_sz) {
    if (data == NULL || len == 0) {
        if (out_sz > 0) {
            out[0] = '\0';
        }
        return;
    }
    size_t n = take < len ? take : len;
    const unsigned char *start = data + (len - n);
    bytes_hex_prefix(start, n, n, out, out_sz);
}

static inline uint32_t read_u32_le(const unsigned char *p) {
    return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

typedef int (*fn_libusb_submit_transfer)(struct libusb_transfer *transfer);
typedef int (*fn_libusb_bulk_transfer)(libusb_device_handle *dev_handle,
                                       unsigned char endpoint,
                                       unsigned char *data,
                                       int length,
                                       int *transferred,
                                       unsigned int timeout);

static fn_libusb_submit_transfer real_submit_transfer = NULL;
static fn_libusb_bulk_transfer real_bulk_transfer = NULL;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_log = NULL;
static int g_out_only = 1;
static unsigned long long g_seq = 0;

#define MAX_DUMP_LENS 32
static int g_dump_enabled = 0;
static int g_dump_once_per_len = 1;
static char g_dump_dir[512];
static int g_dump_lens[MAX_DUMP_LENS];
static int g_dump_seen[MAX_DUMP_LENS];
static size_t g_dump_lens_count = 0;

struct stream_state {
    int valid;
    uint32_t tag;
    uint32_t remaining;
};

static struct stream_state g_stream = {0, 0, 0};

static void parse_dump_lens(const char *raw) {
    if (raw == NULL || raw[0] == '\0') {
        return;
    }
    char *tmp = strdup(raw);
    if (tmp == NULL) {
        return;
    }
    char *saveptr = NULL;
    char *tok = strtok_r(tmp, ",", &saveptr);
    while (tok != NULL && g_dump_lens_count < MAX_DUMP_LENS) {
        while (*tok == ' ' || *tok == '\t') {
            tok++;
        }
        if (*tok != '\0') {
            char *endptr = NULL;
            long v = strtol(tok, &endptr, 10);
            if (endptr != tok && v > 0 && v <= INT32_MAX) {
                g_dump_lens[g_dump_lens_count] = (int)v;
                g_dump_seen[g_dump_lens_count] = 0;
                g_dump_lens_count++;
            }
        }
        tok = strtok_r(NULL, ",", &saveptr);
    }
    free(tmp);
}

static int find_dump_len_index(int len) {
    for (size_t i = 0; i < g_dump_lens_count; ++i) {
        if (g_dump_lens[i] == len) {
            return (int)i;
        }
    }
    return -1;
}

static void maybe_dump_payload(const char *api,
                               int len,
                               const unsigned char *data,
                               int is_in,
                               unsigned long long seq) {
    if (!g_dump_enabled || is_in || data == NULL || len <= 0) {
        return;
    }
    int idx = find_dump_len_index(len);
    if (idx < 0) {
        return;
    }
    if (g_dump_once_per_len && g_dump_seen[idx]) {
        return;
    }

    char out_path[1024];
    snprintf(out_path, sizeof(out_path), "%s/%s_len%d_seq%llu.bin", g_dump_dir, api, len, seq);
    FILE *f = fopen(out_path, "wb");
    if (f == NULL) {
        return;
    }
    size_t written = fwrite(data, 1, (size_t)len, f);
    fclose(f);
    if (written == (size_t)len && g_dump_once_per_len) {
        g_dump_seen[idx] = 1;
    }
}

static void resolve_symbols(void) {
    if (real_submit_transfer == NULL) {
        real_submit_transfer = (fn_libusb_submit_transfer)dlsym(RTLD_NEXT, "libusb_submit_transfer");
    }
    if (real_bulk_transfer == NULL) {
        real_bulk_transfer = (fn_libusb_bulk_transfer)dlsym(RTLD_NEXT, "libusb_bulk_transfer");
    }
}

static unsigned long long now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (unsigned long long)ts.tv_sec * 1000000ULL + (unsigned long long)(ts.tv_nsec / 1000ULL);
}

static long get_tid(void) {
    return (long)syscall(SYS_gettid);
}

static void init_logger_once(void) {
    if (g_log != NULL) {
        return;
    }
    const char *path = getenv("LIBUSB_TRACE_LOG");
    if (path == NULL || path[0] == '\0') {
        path = "/tmp/libusb_trace.tsv";
    }

    const char *out_only_env = getenv("LIBUSB_TRACE_OUT_ONLY");
    if (out_only_env != NULL && strcmp(out_only_env, "0") == 0) {
        g_out_only = 0;
    }

    const char *dump_dir = getenv("LIBUSB_TRACE_DUMP_DIR");
    const char *dump_lens = getenv("LIBUSB_TRACE_DUMP_LENS");
    const char *dump_once = getenv("LIBUSB_TRACE_DUMP_ONCE_PER_LEN");
    if (dump_once != NULL && strcmp(dump_once, "0") == 0) {
        g_dump_once_per_len = 0;
    }
    if (dump_dir != NULL && dump_dir[0] != '\0' && dump_lens != NULL && dump_lens[0] != '\0') {
        snprintf(g_dump_dir, sizeof(g_dump_dir), "%s", dump_dir);
        parse_dump_lens(dump_lens);
        if (g_dump_lens_count > 0) {
            g_dump_enabled = 1;
        }
    }

    g_log = fopen(path, "a");
    if (g_log == NULL) {
        return;
    }

    static int header_written = 0;
    if (!header_written) {
        fprintf(g_log,
                "#ts_us\tseq\tpid\ttid\tapi\tep\tdir\tlen\tsha256\tfirst16\tlast16\tis_header\thdr_len\thdr_tag\tstream_tag\tstream_rem_before\tstream_rem_after\tret\ttransferred\ttimeout_ms\n");
        fflush(g_log);
        header_written = 1;
    }
}

static void classify_out_stream(const unsigned char *data,
                                int len,
                                int *is_header,
                                uint32_t *hdr_len,
                                uint32_t *hdr_tag,
                                int *stream_tag,
                                uint32_t *rem_before,
                                uint32_t *rem_after) {
    *is_header = 0;
    *hdr_len = 0;
    *hdr_tag = 0;
    *stream_tag = -1;
    *rem_before = g_stream.remaining;
    *rem_after = g_stream.remaining;

    if (len == 8 && data != NULL) {
        *is_header = 1;
        *hdr_len = read_u32_le(data);
        *hdr_tag = read_u32_le(data + 4);
        g_stream.valid = 1;
        g_stream.tag = *hdr_tag;
        g_stream.remaining = *hdr_len;
        *stream_tag = (int)g_stream.tag;
        *rem_before = g_stream.remaining;
        *rem_after = g_stream.remaining;
        return;
    }

    if (!g_stream.valid) {
        return;
    }

    *stream_tag = (int)g_stream.tag;
    *rem_before = g_stream.remaining;

    uint32_t delta = (len > 0) ? (uint32_t)len : 0;
    if (delta >= g_stream.remaining) {
        g_stream.remaining = 0;
        g_stream.valid = 0;
    } else {
        g_stream.remaining -= delta;
    }
    *rem_after = g_stream.remaining;
}

static void log_transfer(const char *api,
                         unsigned char endpoint,
                         const unsigned char *data,
                         int len,
                         int ret,
                         int transferred,
                         unsigned int timeout_ms) {
    if (len < 0) {
        len = 0;
    }
    int is_in = (endpoint & LIBUSB_ENDPOINT_IN) != 0;
    if (g_out_only && is_in) {
        return;
    }

    init_logger_once();
    if (g_log == NULL) {
        return;
    }

    char sha[65];
    char first16[33];
    char last16[33];
    sha[0] = '\0';
    first16[0] = '\0';
    last16[0] = '\0';

    if (data != NULL && len > 0) {
        sha256_hex(data, (size_t)len, sha);
        bytes_hex_prefix(data, (size_t)len, 16, first16, sizeof(first16));
        bytes_hex_suffix(data, (size_t)len, 16, last16, sizeof(last16));
    }

    int is_header = 0;
    uint32_t hdr_len = 0;
    uint32_t hdr_tag = 0;
    int stream_tag = -1;
    uint32_t rem_before = g_stream.remaining;
    uint32_t rem_after = g_stream.remaining;

    if (!is_in) {
        classify_out_stream(data, len, &is_header, &hdr_len, &hdr_tag, &stream_tag, &rem_before,
                            &rem_after);
    }

    unsigned long long seq = ++g_seq;
    maybe_dump_payload(api, len, data, is_in, seq);
    fprintf(g_log,
            "%llu\t%llu\t%d\t%ld\t%s\t0x%02x\t%c\t%d\t%s\t%s\t%s\t%d\t%u\t%u\t%d\t%u\t%u\t%d\t%d\t%u\n",
            now_us(), seq, getpid(), get_tid(), api, endpoint, is_in ? 'I' : 'O', len,
            sha, first16, last16, is_header, hdr_len, hdr_tag, stream_tag, rem_before, rem_after,
            ret, transferred, timeout_ms);
    fflush(g_log);
}

int libusb_submit_transfer(struct libusb_transfer *transfer) {
    resolve_symbols();
    if (real_submit_transfer == NULL) {
        errno = ENOSYS;
        return LIBUSB_ERROR_NOT_SUPPORTED;
    }

    int ret = real_submit_transfer(transfer);

    pthread_mutex_lock(&g_lock);
    if (transfer != NULL) {
        log_transfer("submit", transfer->endpoint, transfer->buffer, transfer->length, ret, -1,
                     transfer->timeout);
    } else {
        log_transfer("submit", 0, NULL, 0, ret, -1, 0);
    }
    pthread_mutex_unlock(&g_lock);

    return ret;
}

int libusb_bulk_transfer(libusb_device_handle *dev_handle,
                         unsigned char endpoint,
                         unsigned char *data,
                         int length,
                         int *transferred,
                         unsigned int timeout) {
    resolve_symbols();
    if (real_bulk_transfer == NULL) {
        errno = ENOSYS;
        return LIBUSB_ERROR_NOT_SUPPORTED;
    }

    int ret = real_bulk_transfer(dev_handle, endpoint, data, length, transferred, timeout);

    int actual = (transferred != NULL) ? *transferred : -1;
    pthread_mutex_lock(&g_lock);
    log_transfer("bulk", endpoint, data, length, ret, actual, timeout);
    pthread_mutex_unlock(&g_lock);

    return ret;
}
