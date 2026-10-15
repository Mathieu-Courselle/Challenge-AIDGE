#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define K 5
typedef struct { const char *name; int cin, cout, hin, win; } layer_t;

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static unsigned rng = 12345;
static float frand(void) { rng = rng * 1664525u + 1013904223u; return (rng >> 8) / 16777216.0f; }

/* Formats (NHWC) :
   entree  inp[(iy*Wp+ix)*cin+ci], avec 2 pixels de zeros autour (Wp = win+4)
   poids   wt[((ky*K+kx)*cin+ci)*cout+co]
   sortie  out[(y*ow+x)*cout+co]   (conv 5x5, stride 2, padding 2) */

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

static void conv_neon(const float *inp, const float *wt, float *out, int cin, int cout, int hin, int win) {
#ifdef __ARM_NEON
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
#else
    conv_ref(inp, wt, out, cin, cout, hin, win);
#endif
}

int main(void) {
    /* 4 convolutions de l'encodeur de bmshj2018_factorized (qualite 1-5), image 64x64 */
    layer_t L[4] = { {"L1   3->128 64x64", 3, 128, 64, 64}, {"L2 128->128 32x32", 128, 128, 32, 32},
                     {"L3 128->128 16x16", 128, 128, 16, 16}, {"L4 128->192  8x8 ", 128, 192, 8, 8} };
#ifdef __ARM_NEON
    printf("NEON actif\n");
#else
    printf("NEON ABSENT (version scalaire seulement)\n");
#endif
    printf("%-18s %8s | %9s %8s | %9s %8s | %6s | %s\n", "couche", "M MAC", "ref ms", "MMAC/s", "neon ms", "MMAC/s", "gain", "err max");
    double tot_ref = 0, tot_neon = 0, tot_mac = 0;
    for (int l = 0; l < 4; l++) {
        int cin = L[l].cin, cout = L[l].cout, hin = L[l].hin, win = L[l].win;
        int hp = hin + 4, wp = win + 4, oh = hin / 2, ow = win / 2;
        size_t ni = (size_t)hp * wp * cin, nw = (size_t)K * K * cin * cout, no = (size_t)oh * ow * cout;
        float *inp = calloc(ni, 4), *wt = malloc(nw * 4), *o1 = malloc(no * 4), *o2 = malloc(no * 4);
        for (int y = 0; y < hin; y++) for (int x = 0; x < win; x++) for (int c = 0; c < cin; c++)
            inp[(size_t)((y + 2) * wp + x + 2) * cin + c] = frand();
        for (size_t i = 0; i < nw; i++) wt[i] = (frand() - 0.5f) * 0.2f;
        double macs = (double)oh * ow * cout * cin * K * K;
        double t0 = now_ms(); conv_ref(inp, wt, o1, cin, cout, hin, win); double tr = now_ms() - t0;
        double tn = 1e9;
        for (int r = 0; r < 3; r++) { t0 = now_ms(); conv_neon(inp, wt, o2, cin, cout, hin, win); double d = now_ms() - t0; if (d < tn) tn = d; }
        double emax = 0;
        for (size_t i = 0; i < no; i++) { double d = fabs((double)o1[i] - o2[i]); if (d > emax) emax = d; }
        printf("%-18s %8.1f | %9.1f %8.1f | %9.1f %8.1f | %5.1fx | %.2e\n", L[l].name, macs / 1e6, tr, macs / 1e3 / tr, tn, macs / 1e3 / tn, tr / tn, emax);
        tot_ref += tr; tot_neon += tn; tot_mac += macs;
        free(inp); free(wt); free(o1); free(o2);
    }
    printf("%-18s %8.1f | %9.1f %8.1f | %9.1f %8.1f | %5.1fx |\n", "TOTAL encodeur", tot_mac / 1e6, tot_ref, tot_mac / 1e3 / tot_ref, tot_neon, tot_mac / 1e3 / tot_neon, tot_ref / tot_neon);
    printf("(convolutions seules : GDN et codage entropique non inclus)\n");
    return 0;
}
