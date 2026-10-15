/* balle.c : bmshj2018_factorized complet en C (float32, NEON) + codage entropique rANS.
   Compresse puis decompresse des images 64x64 RGB, mesure taille, PSNR et temps.
   Usage : ./balle weights.bin images.bin N sortie.csv [check.bin]
   Compilation carte : $CC -O2 -std=gnu11 balle.c -o balle -lm   (voir README) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error "NEON requis (ou utiliser le shim x86 de test)"
#endif

#define K 5
#define IMG 64
#define PROB_BITS 16
#define RANS_L (1u << 23)

typedef struct { float *w, *b; int cin, cout; } layer_t;
typedef struct { float *gt, *beta; int c; } gdn_t;

static int N, M, CDFL;
static layer_t ga[4], gs[4];
static gdn_t gda[3], gds[3];
static float *medians;
static int32_t *offs, *cdflen, *cdf;

static void *xmalloc(size_t n) { void *p = malloc(n); if (!p) { fprintf(stderr, "memoire insuffisante\n"); exit(1); } return p; }
static float *readf(FILE *f, size_t n) { float *p = xmalloc(n * 4); if (fread(p, 4, n, f) != n) { fprintf(stderr, "fichier de poids tronque\n"); exit(1); } return p; }
static int32_t *readi(FILE *f, size_t n) { int32_t *p = xmalloc(n * 4); if (fread(p, 4, n, f) != n) { fprintf(stderr, "fichier de poids tronque\n"); exit(1); } return p; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static void load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "impossible d'ouvrir %s\n", path); exit(1); }
    char mg[4]; int32_t h[3];
    if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "BAL1", 4) || fread(h, 4, 3, f) != 3) { fprintf(stderr, "fichier de poids invalide\n"); exit(1); }
    N = h[0]; M = h[1]; CDFL = h[2];
    int cia[4] = {3, N, N, N}, coa[4] = {N, N, N, M};
    int cis[4] = {M, N, N, N}, cos_[4] = {N, N, N, 8};
    for (int i = 0; i < 4; i++) {
        ga[i].cin = cia[i]; ga[i].cout = coa[i];
        ga[i].w = readf(f, (size_t)K * K * cia[i] * coa[i]); ga[i].b = readf(f, coa[i]);
        if (i < 3) { gda[i].c = N; gda[i].gt = readf(f, (size_t)N * N); gda[i].beta = readf(f, N); }
    }
    for (int i = 0; i < 4; i++) {
        gs[i].cin = cis[i]; gs[i].cout = cos_[i];
        gs[i].w = readf(f, (size_t)K * K * cis[i] * cos_[i]); gs[i].b = readf(f, cos_[i]);
        if (i < 3) { gds[i].c = N; gds[i].gt = readf(f, (size_t)N * N); gds[i].beta = readf(f, N); }
    }
    medians = readf(f, M); offs = readi(f, M); cdflen = readi(f, M); cdf = readi(f, (size_t)M * CDFL);
    fclose(f);
}

/* ---- convolution 5x5 stride 2 padding 2 (entree NHWC deja entouree de 2 pixels de zeros) ---- */
static void conv_s2(const float *inp, const layer_t *L, int hin, int win, float *out) {
    const int cin = L->cin, cout = L->cout, wp = win + 4, oh = hin / 2, ow = win / 2;
    for (int cb = 0; cb < cout; cb += 8) {
        const float32x4_t bz0 = vld1q_f32(L->b + cb), bz1 = vld1q_f32(L->b + cb + 4);
        for (int y = 0; y < oh; y++)
        for (int x0 = 0; x0 < ow; x0 += 4) {
            float32x4_t a00 = bz0, a01 = bz1, a10 = bz0, a11 = bz1, a20 = bz0, a21 = bz1, a30 = bz0, a31 = bz1;
            for (int ky = 0; ky < K; ky++) for (int kx = 0; kx < K; kx++) {
                const float *w = L->w + (size_t)(ky * K + kx) * cin * cout + cb;
                const float *i0 = inp + (size_t)((2 * y + ky) * wp + 2 * x0 + kx) * cin;
                const float *i1 = i0 + 2 * cin, *i2 = i1 + 2 * cin, *i3 = i2 + 2 * cin;
                for (int ci = 0; ci < cin; ci++) {
                    float32x4_t w0 = vld1q_f32(w), w1 = vld1q_f32(w + 4);
                    w += cout;
                    float s = i0[ci]; a00 = vmlaq_n_f32(a00, w0, s); a01 = vmlaq_n_f32(a01, w1, s);
                    s = i1[ci];       a10 = vmlaq_n_f32(a10, w0, s); a11 = vmlaq_n_f32(a11, w1, s);
                    s = i2[ci];       a20 = vmlaq_n_f32(a20, w0, s); a21 = vmlaq_n_f32(a21, w1, s);
                    s = i3[ci];       a30 = vmlaq_n_f32(a30, w0, s); a31 = vmlaq_n_f32(a31, w1, s);
                }
            }
            float *o = out + (size_t)(y * ow + x0) * cout + cb;
            vst1q_f32(o, a00); vst1q_f32(o + 4, a01); o += cout;
            vst1q_f32(o, a10); vst1q_f32(o + 4, a11); o += cout;
            vst1q_f32(o, a20); vst1q_f32(o + 4, a21); o += cout;
            vst1q_f32(o, a30); vst1q_f32(o + 4, a31);
        }
    }
}

/* ---- deconvolution 5x5 stride 2 padding 2 output_padding 1 : sortie 2h x 2w ---- */
/* forme "gather" : un pixel de sortie (oy,ox) recoit les taps ky = oy mod 2 (pas 2), kx = ox mod 2 (pas 2).
   entree NHWC entouree de 1 pixel de zeros. 4 pixels de sortie de meme parite en x par bloc. */
static void deconv_s2(const float *inp, const layer_t *L, int hin, int win, float *out) {
    const int cin = L->cin, cout = L->cout, wp = win + 2, oh = 2 * hin, ow = 2 * win;
    for (int cb = 0; cb < cout; cb += 8) {
        const float32x4_t bz0 = vld1q_f32(L->b + cb), bz1 = vld1q_f32(L->b + cb + 4);
        for (int oy = 0; oy < oh; oy++)
        for (int px = 0; px < 2; px++)
        for (int x0 = 0; x0 < win; x0 += 4) {
            float32x4_t a00 = bz0, a01 = bz1, a10 = bz0, a11 = bz1, a20 = bz0, a21 = bz1, a30 = bz0, a31 = bz1;
            for (int ky = oy & 1; ky < K; ky += 2) {
                const int iy = (oy + 2 - ky) / 2;
                for (int kx = px; kx < K; kx += 2) {
                    const int sx = (px + 2 - kx) / 2;
                    const float *w = L->w + (size_t)(ky * K + kx) * cin * cout + cb;
                    const float *i0 = inp + (size_t)((iy + 1) * wp + (x0 + sx + 1)) * cin;
                    const float *i1 = i0 + cin, *i2 = i1 + cin, *i3 = i2 + cin;
                    for (int ci = 0; ci < cin; ci++) {
                        float32x4_t w0 = vld1q_f32(w), w1 = vld1q_f32(w + 4);
                        w += cout;
                        float s = i0[ci]; a00 = vmlaq_n_f32(a00, w0, s); a01 = vmlaq_n_f32(a01, w1, s);
                        s = i1[ci];       a10 = vmlaq_n_f32(a10, w0, s); a11 = vmlaq_n_f32(a11, w1, s);
                        s = i2[ci];       a20 = vmlaq_n_f32(a20, w0, s); a21 = vmlaq_n_f32(a21, w1, s);
                        s = i3[ci];       a30 = vmlaq_n_f32(a30, w0, s); a31 = vmlaq_n_f32(a31, w1, s);
                    }
                }
            }
            float *o = out + (size_t)(oy * ow + px + 2 * x0) * cout + cb;
            const size_t d = (size_t)2 * cout;   /* pixels de sortie voisins du bloc : un pixel sur deux */
            vst1q_f32(o, a00); vst1q_f32(o + 4, a01); o += d;
            vst1q_f32(o, a10); vst1q_f32(o + 4, a11); o += d;
            vst1q_f32(o, a20); vst1q_f32(o + 4, a21); o += d;
            vst1q_f32(o, a30); vst1q_f32(o + 4, a31);
        }
    }
}

/* ---- GDN (inverse=0) / IGDN (inverse=1), sur place, NHWC, C = N canaux ---- */
static void gdn(float *x, const gdn_t *g, int npix, int inverse) {
    const int C = g->c;
    float x2[256];
    for (int p = 0; p < npix; p++) {
        float *px = x + (size_t)p * C;
        for (int i = 0; i < C; i++) x2[i] = px[i] * px[i];
        for (int cb = 0; cb < C; cb += 16) {
            float32x4_t a0 = vld1q_f32(g->beta + cb), a1 = vld1q_f32(g->beta + cb + 4),
                        a2 = vld1q_f32(g->beta + cb + 8), a3 = vld1q_f32(g->beta + cb + 12);
            const float *w = g->gt + cb;
            for (int ci = 0; ci < C; ci++, w += C) {
                float s = x2[ci];
                a0 = vmlaq_n_f32(a0, vld1q_f32(w), s);      a1 = vmlaq_n_f32(a1, vld1q_f32(w + 4), s);
                a2 = vmlaq_n_f32(a2, vld1q_f32(w + 8), s);  a3 = vmlaq_n_f32(a3, vld1q_f32(w + 12), s);
            }
            float t[16];
            vst1q_f32(t, a0); vst1q_f32(t + 4, a1); vst1q_f32(t + 8, a2); vst1q_f32(t + 12, a3);
            for (int k = 0; k < 16; k++) {
                float n = sqrtf(t[k]);
                px[cb + k] = inverse ? px[cb + k] * n : px[cb + k] / n;
            }
        }
    }
}

static void pad_copy(const float *src, int h, int w, int c, int p, float *dst) {
    const int wp = w + 2 * p;
    memset(dst, 0, (size_t)(h + 2 * p) * wp * c * 4);
    for (int y = 0; y < h; y++) memcpy(dst + ((size_t)(y + p) * wp + p) * c, src + (size_t)y * w * c, (size_t)w * c * 4);
}

/* ---- codage entropique rANS (probabilites 16 bits, tables du modele CompressAI) ---- */
typedef struct { uint8_t *ptr; uint32_t x; } enc_t;
static void rput(enc_t *e, uint32_t start, uint32_t freq) {
    uint32_t x = e->x, xmax = ((RANS_L >> PROB_BITS) << 8) * freq;
    while (x >= xmax) { *--e->ptr = (uint8_t)x; x >>= 8; }
    e->x = ((x / freq) << PROB_BITS) + (x % freq) + start;
}

static int entropy_encode(const int32_t *sym, uint8_t *out, int cap) {
    const int NS = M * 16;
    uint8_t *tmp = xmalloc(cap), *end = tmp + cap;
    enc_t e = { end, RANS_L };
    for (int i = NS - 1; i >= 0; i--) {
        const int c = i / 16;
        const int32_t *cd = cdf + (size_t)c * CDFL;
        const int maxv = cdflen[c] - 2;
        int v = sym[i] - offs[c]; uint32_t raw = 0; int esc = 0;
        if (v < 0) { raw = (uint32_t)(-2 * v - 1); v = maxv; esc = 1; }
        else if (v >= maxv) { raw = (uint32_t)(2 * (v - maxv)); v = maxv; esc = 1; }
        if (esc) { if (raw > 65535) { fprintf(stderr, "symbole hors plage\n"); raw = 65535; } rput(&e, raw, 1); }
        rput(&e, (uint32_t)cd[v], (uint32_t)(cd[v + 1] - cd[v]));
    }
    e.ptr -= 4; e.ptr[0] = e.x; e.ptr[1] = e.x >> 8; e.ptr[2] = e.x >> 16; e.ptr[3] = e.x >> 24;
    int n = (int)(end - e.ptr);
    memcpy(out, e.ptr, n); free(tmp);
    return n;
}

static void entropy_decode(const uint8_t *buf, int32_t *sym) {
    const int NS = M * 16;
    const uint8_t *p = buf;
    uint32_t x = p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); p += 4;
    for (int i = 0; i < NS; i++) {
        const int c = i / 16;
        const int32_t *cd = cdf + (size_t)c * CDFL;
        const int maxv = cdflen[c] - 2;
        uint32_t cum = x & ((1u << PROB_BITS) - 1);
        int v = 0; while (v < maxv && (uint32_t)cd[v + 1] <= cum) v++;
        uint32_t start = cd[v], freq = cd[v + 1] - cd[v];
        x = freq * (x >> PROB_BITS) + cum - start;
        while (x < RANS_L) x = (x << 8) | *p++;
        int val;
        if (v == maxv) {
            uint32_t raw = x & ((1u << PROB_BITS) - 1);
            x = x >> PROB_BITS;
            while (x < RANS_L) x = (x << 8) | *p++;
            val = (raw & 1) ? -(int)((raw + 1) / 2) : maxv + (int)(raw / 2);
        } else val = v;
        sym[i] = val + offs[c];
    }
}

/* ---- buffers de travail ---- */
static float *pad0, *a1, *p1, *a2, *p2, *a3, *p3, *yq;
static float *q0, *d1, *q1, *d2, *q2, *d3, *q3, *xhat;
static void alloc_buffers(void) {
    pad0 = xmalloc((size_t)68 * 68 * 3 * 4);
    a1 = xmalloc((size_t)32 * 32 * N * 4); p1 = xmalloc((size_t)36 * 36 * N * 4);
    a2 = xmalloc((size_t)16 * 16 * N * 4); p2 = xmalloc((size_t)20 * 20 * N * 4);
    a3 = xmalloc((size_t)8 * 8 * N * 4);   p3 = xmalloc((size_t)12 * 12 * N * 4);
    yq = xmalloc((size_t)16 * M * 4);
    q0 = xmalloc((size_t)6 * 6 * M * 4);
    d1 = xmalloc((size_t)8 * 8 * N * 4);   q1 = xmalloc((size_t)10 * 10 * N * 4);
    d2 = xmalloc((size_t)16 * 16 * N * 4); q2 = xmalloc((size_t)18 * 18 * N * 4);
    d3 = xmalloc((size_t)32 * 32 * N * 4); q3 = xmalloc((size_t)34 * 34 * N * 4);
    xhat = xmalloc((size_t)64 * 64 * 8 * 4);
}

static int encode_image(const uint8_t *img, uint8_t *out, int32_t *sym) {
    memset(pad0, 0, (size_t)68 * 68 * 3 * 4);
    for (int y = 0; y < IMG; y++) for (int x = 0; x < IMG; x++) for (int c = 0; c < 3; c++)
        pad0[((size_t)(y + 2) * 68 + x + 2) * 3 + c] = img[((size_t)y * IMG + x) * 3 + c] / 255.0f;
    conv_s2(pad0, &ga[0], 64, 64, a1); gdn(a1, &gda[0], 32 * 32, 0); pad_copy(a1, 32, 32, N, 2, p1);
    conv_s2(p1, &ga[1], 32, 32, a2);   gdn(a2, &gda[1], 16 * 16, 0); pad_copy(a2, 16, 16, N, 2, p2);
    conv_s2(p2, &ga[2], 16, 16, a3);   gdn(a3, &gda[2], 8 * 8, 0);   pad_copy(a3, 8, 8, N, 2, p3);
    conv_s2(p3, &ga[3], 8, 8, yq);
    for (int c = 0; c < M; c++) for (int p = 0; p < 16; p++)
        sym[c * 16 + p] = (int32_t)nearbyintf(yq[(size_t)p * M + c] - medians[c]);
    return entropy_encode(sym, out, 1 << 16);
}

static void decode_image(const uint8_t *buf, int32_t *sym) {
    entropy_decode(buf, sym);
    memset(q0, 0, (size_t)6 * 6 * M * 4);
    for (int p = 0; p < 16; p++) for (int c = 0; c < M; c++)
        q0[((size_t)((p / 4) + 1) * 6 + (p % 4) + 1) * M + c] = (float)sym[c * 16 + p] + medians[c];
    deconv_s2(q0, &gs[0], 4, 4, d1);   gdn(d1, &gds[0], 8 * 8, 1);   pad_copy(d1, 8, 8, N, 1, q1);
    deconv_s2(q1, &gs[1], 8, 8, d2);   gdn(d2, &gds[1], 16 * 16, 1); pad_copy(d2, 16, 16, N, 1, q2);
    deconv_s2(q2, &gs[2], 16, 16, d3); gdn(d3, &gds[2], 32 * 32, 1); pad_copy(d3, 32, 32, N, 1, q3);
    deconv_s2(q3, &gs[3], 32, 32, xhat);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s weights.bin images.bin N sortie.csv [check.bin]\n", argv[0]); return 1; }
    load_weights(argv[1]); alloc_buffers();
    int n = atoi(argv[3]);
    FILE *fi = fopen(argv[2], "rb");
    if (!fi) { fprintf(stderr, "impossible d'ouvrir %s\n", argv[2]); return 1; }
    uint8_t *imgs = xmalloc((size_t)n * IMG * IMG * 3);
    if (fread(imgs, IMG * IMG * 3, n, fi) != (size_t)n) { fprintf(stderr, "images.bin trop court\n"); return 1; }
    fclose(fi);
    FILE *fo = fopen(argv[4], "w");
    fprintf(fo, "idx,bytes,psnr,enc_ms,dec_ms\n");
    uint8_t *stream = xmalloc(1 << 16);
    int32_t *sym = xmalloc((size_t)M * 16 * 4), *sym2 = xmalloc((size_t)M * 16 * 4);
    double s_bytes = 0, s_psnr = 0, s_enc = 0, s_dec = 0; int rt_err = 0;
    for (int i = 0; i < n; i++) {
        const uint8_t *img = imgs + (size_t)i * IMG * IMG * 3;
        double t0 = now_ms(); int nb = encode_image(img, stream, sym);
        double t1 = now_ms(); decode_image(stream, sym2);
        double t2 = now_ms();
        for (int k = 0; k < M * 16; k++) if (sym[k] != sym2[k]) { rt_err++; break; }
        double se = 0;
        for (int p = 0; p < IMG * IMG; p++) for (int c = 0; c < 3; c++) {
            float v = xhat[(size_t)p * 8 + c]; v = v < 0 ? 0 : (v > 1 ? 1 : v);
            double d = v - img[(size_t)p * 3 + c] / 255.0; se += d * d;
        }
        double mse = se / (IMG * IMG * 3); if (mse < 1e-10) mse = 1e-10;
        double psnr = -10 * log10(mse);
        fprintf(fo, "%d,%d,%.4f,%.3f,%.3f\n", i, nb, psnr, t1 - t0, t2 - t1);
        s_bytes += nb; s_psnr += psnr; s_enc += t1 - t0; s_dec += t2 - t1;
        if (i == 0 && argc > 5) {                       /* comparaison avec la reference PyTorch/numpy */
            FILE *fc = fopen(argv[5], "rb");
            if (fc) {
                int32_t *rs = xmalloc((size_t)M * 16 * 4); float *rx = xmalloc((size_t)IMG * IMG * 3 * 4);
                if (fread(rs, 4, (size_t)M * 16, fc) == (size_t)M * 16 && fread(rx, 4, (size_t)IMG * IMG * 3, fc) == (size_t)IMG * IMG * 3) {
                    int diff = 0; for (int k = 0; k < M * 16; k++) diff += rs[k] != sym[k];
                    double mx = 0;
                    for (int p = 0; p < IMG * IMG; p++) for (int c = 0; c < 3; c++) {
                        float v = xhat[(size_t)p * 8 + c]; v = v < 0 ? 0 : (v > 1 ? 1 : v);
                        double d = fabs(v - rx[(size_t)p * 3 + c]); if (d > mx) mx = d;
                    }
                    printf("VERIF image 0 : %d symboles differents sur %d | ecart max image reconstruite %.2e\n", diff, M * 16, mx);
                }
                fclose(fc);
            }
        }
    }
    fclose(fo);
    printf("%d images | octets moy %.1f | %.3f bpp | taux %.1f:1 | PSNR moy %.2f dB | compression %.1f ms | decompression %.1f ms | aller-retour entropique : %s\n",
           n, s_bytes / n, s_bytes / n * 8 / (IMG * IMG), IMG * IMG * 24 / (s_bytes / n * 8), s_psnr / n, s_enc / n, s_dec / n,
           rt_err ? "ERREUR" : "OK");
    return rt_err != 0;
}
