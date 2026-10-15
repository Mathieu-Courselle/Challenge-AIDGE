#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define K 5
#define CC 16   /* taille d'un bloc de canaux d'entree (poids du bloc ~13 Ko : tient dans le cache L1) */
typedef struct { const char *name; int cin, cout, hin, win; } layer_t;

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static unsigned rng = 12345;
static float frand(void) { rng = rng * 1664525u + 1013904223u; return (rng >> 8) / 16777216.0f; }

/* NHWC : entree inp[(iy*Wp+ix)*cin+ci] (zeros sur 2 pixels autour), poids wt[((ky*K+kx)*cin+ci)*cout+co],
   sortie out[(y*ow+x)*cout+co] ; conv 5x5 stride 2 padding 2 */

static void conv_ref(const float *inp, const float *wt, float *out, int cin, int cout, int hin, int win) {
    int wp = win + 4, oh = hin / 2, ow = win / 2;
    for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) {
        float *o = out + (size_t)(y * ow + x) * cout;
        for (int co = 0; co < cout; co++) o[co] = 0.f;
        for (int ky = 0; ky < K; ky++) for (int kx = 0; kx < K; kx++) {
            const float *ip = inp + (size_t)((2 * y + ky) * wp + 2 * x + kx) * cin;
            const float *w = wt + (size_t)(ky * K + kx) * cin * cout;
            for (int ci = 0; ci < cin; ci++) {
                float s = ip[ci]; const float *wc = w + (size_t)ci * cout;
                for (int co = 0; co < cout; co++) o[co] += s * wc[co];
            }
        }
    }
}

#ifdef __ARM_NEON
/* v1 : poids dans l'ordre d'origine (acces avec un grand pas) */
static void conv_v1(const float *inp, const float *wt, float *out, int cin, int cout, int hin, int win) {
    int wp = win + 4, oh = hin / 2, ow = win / 2;
    for (int cb = 0; cb < cout; cb += 8)
    for (int y = 0; y < oh; y++)
    for (int x0 = 0; x0 < ow; x0 += 4) {
        float32x4_t z = vdupq_n_f32(0.f);
        float32x4_t a00 = z, a01 = z, a10 = z, a11 = z, a20 = z, a21 = z, a30 = z, a31 = z;
        for (int ky = 0; ky < K; ky++) for (int kx = 0; kx < K; kx++) {
            const float *w = wt + (size_t)(ky * K + kx) * cin * cout + cb;
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
#endif

/* Prepare les poids une fois pour toutes (a l'export, pas a l'execution) :
   dst[cb][bloc de canaux][ky][kx][ci dans le bloc][8 sorties]  -> lecture sequentielle */
static void pack_weights(const float *wt, float *dst, int cin, int cout) {
    for (int cb = 0; cb < cout; cb += 8)
    for (int c0 = 0; c0 < cin; c0 += CC) {
        int cl = cin - c0 < CC ? cin - c0 : CC;
        float *d = dst + (size_t)(cb / 8) * K * K * cin * 8 + (size_t)K * K * c0 * 8;
        for (int ky = 0; ky < K; ky++) for (int kx = 0; kx < K; kx++)
            for (int ci = 0; ci < cl; ci++)
                memcpy(d + (size_t)((ky * K + kx) * cl + ci) * 8,
                       wt + (size_t)((ky * K + kx) * cin + c0 + ci) * cout + cb, 8 * sizeof(float));
    }
}

#ifdef __ARM_NEON
/* v2 : poids pre-ranges + blocs de 16 canaux d'entree (poids lus depuis le cache L1) */
static void conv_v2(const float *inp, const float *pk, float *out, int cin, int cout, int hin, int win) {
    int wp = win + 4, oh = hin / 2, ow = win / 2;
    for (int cb = 0; cb < cout; cb += 8)
    for (int c0 = 0; c0 < cin; c0 += CC) {
        int cl = cin - c0 < CC ? cin - c0 : CC;
        const float *pb = pk + (size_t)(cb / 8) * K * K * cin * 8 + (size_t)K * K * c0 * 8;
        for (int y = 0; y < oh; y++)
        for (int x0 = 0; x0 < ow; x0 += 4) {
            float *o = out + (size_t)(y * ow + x0) * cout + cb;
            float32x4_t a00, a01, a10, a11, a20, a21, a30, a31;
            if (c0 == 0) {
                a00 = a01 = a10 = a11 = a20 = a21 = a30 = a31 = vdupq_n_f32(0.f);
            } else {
                a00 = vld1q_f32(o);              a01 = vld1q_f32(o + 4);
                a10 = vld1q_f32(o + cout);       a11 = vld1q_f32(o + cout + 4);
                a20 = vld1q_f32(o + 2 * cout);   a21 = vld1q_f32(o + 2 * cout + 4);
                a30 = vld1q_f32(o + 3 * cout);   a31 = vld1q_f32(o + 3 * cout + 4);
            }
            const float *w = pb;
            for (int ky = 0; ky < K; ky++) for (int kx = 0; kx < K; kx++) {
                const float *i0 = inp + (size_t)((2 * y + ky) * wp + 2 * x0 + kx) * cin + c0;
                const float *i1 = i0 + 2 * cin, *i2 = i1 + 2 * cin, *i3 = i2 + 2 * cin;
                for (int ci = 0; ci < cl; ci++) {
                    float32x4_t w0 = vld1q_f32(w), w1 = vld1q_f32(w + 4);
                    w += 8;
                    float s = i0[ci]; a00 = vmlaq_n_f32(a00, w0, s); a01 = vmlaq_n_f32(a01, w1, s);
                    s = i1[ci];       a10 = vmlaq_n_f32(a10, w0, s); a11 = vmlaq_n_f32(a11, w1, s);
                    s = i2[ci];       a20 = vmlaq_n_f32(a20, w0, s); a21 = vmlaq_n_f32(a21, w1, s);
                    s = i3[ci];       a30 = vmlaq_n_f32(a30, w0, s); a31 = vmlaq_n_f32(a31, w1, s);
                }
            }
            vst1q_f32(o, a00);            vst1q_f32(o + 4, a01);
            vst1q_f32(o + cout, a10);     vst1q_f32(o + cout + 4, a11);
            vst1q_f32(o + 2 * cout, a20); vst1q_f32(o + 2 * cout + 4, a21);
            vst1q_f32(o + 3 * cout, a30); vst1q_f32(o + 3 * cout + 4, a31);
        }
    }
}
#endif

static double best_of(int reps, void (*f)(const float *, const float *, float *, int, int, int, int),
                      const float *a, const float *b, float *c, int cin, int cout, int hin, int win) {
    double best = 1e9;
    for (int r = 0; r < reps; r++) { double t0 = now_ms(); f(a, b, c, cin, cout, hin, win); double d = now_ms() - t0; if (d < best) best = d; }
    return best;
}
static double maxdiff(const float *a, const float *b, size_t n) {
    double e = 0; for (size_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > e) e = d; } return e;
}

int main(void) {
    layer_t L[4] = { {"L1   3->128 64x64", 3, 128, 64, 64}, {"L2 128->128 32x32", 128, 128, 32, 32},
                     {"L3 128->128 16x16", 128, 128, 16, 16}, {"L4 128->192  8x8 ", 128, 192, 8, 8} };
#ifdef __ARM_NEON
    printf("NEON actif\n");
#else
    printf("NEON ABSENT\n");
#endif
    printf("%-18s %7s | %8s | %8s %7s | %8s %7s | %5s | %s\n", "couche", "M MAC", "ref ms", "v1 ms", "MMAC/s", "v2 ms", "MMAC/s", "v2/ref", "err max v1 / v2");
    double tr = 0, t1 = 0, t2 = 0, tm = 0;
    for (int l = 0; l < 4; l++) {
        int cin = L[l].cin, cout = L[l].cout, hin = L[l].hin, win = L[l].win;
        int hp = hin + 4, wp = win + 4, oh = hin / 2, ow = win / 2;
        size_t ni = (size_t)hp * wp * cin, nw = (size_t)K * K * cin * cout, no = (size_t)oh * ow * cout;
        float *inp = calloc(ni, 4), *wt = malloc(nw * 4), *pk = malloc(nw * 4);
        float *o0 = malloc(no * 4), *o1 = malloc(no * 4), *o2 = malloc(no * 4);
        for (int y = 0; y < hin; y++) for (int x = 0; x < win; x++) for (int c = 0; c < cin; c++)
            inp[(size_t)((y + 2) * wp + x + 2) * cin + c] = frand();
        for (size_t i = 0; i < nw; i++) wt[i] = (frand() - 0.5f) * 0.2f;
        pack_weights(wt, pk, cin, cout);
        double macs = (double)oh * ow * cout * cin * K * K;
        double d0 = best_of(1, conv_ref, inp, wt, o0, cin, cout, hin, win);
#ifdef __ARM_NEON
        double d1 = best_of(3, conv_v1, inp, wt, o1, cin, cout, hin, win);
        double d2 = best_of(3, conv_v2, inp, pk, o2, cin, cout, hin, win);
#else
        double d1 = d0, d2 = d0; memcpy(o1, o0, no * 4); memcpy(o2, o0, no * 4);
#endif
        printf("%-18s %7.1f | %8.1f | %8.1f %7.1f | %8.1f %7.1f | %4.1fx | %.1e / %.1e\n", L[l].name, macs / 1e6, d0,
               d1, macs / 1e3 / d1, d2, macs / 1e3 / d2, d0 / d2, maxdiff(o0, o1, no), maxdiff(o0, o2, no));
        tr += d0; t1 += d1; t2 += d2; tm += macs;
        free(inp); free(wt); free(pk); free(o0); free(o1); free(o2);
    }
    printf("%-18s %7.1f | %8.1f | %8.1f %7.1f | %8.1f %7.1f | %4.1fx |\n", "TOTAL encodeur", tm / 1e6, tr, t1, tm / 1e3 / t1, t2, tm / 1e3 / t2, tr / t2);
    printf("(convolutions seules ; erreur < 1e-3 = identique aux arrondis flottants pres)\n");
    return 0;
}
