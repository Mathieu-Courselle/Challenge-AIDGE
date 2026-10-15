/* Programme de test LoCoD (CPU seul) : 1ere couche de l'encodeur de Balle.
   A placer dans LoCod/demo/conv_test/main.c puis :
   ./locod.sh -t pynqz2 -f demo/conv_test/main.c --no-hard */
#include <stdio.h>
#include <time.h>
#ifndef LOCOD_FPGA
#include "locod.h"
#endif

#define CIN 3
#define COUT 128
#define K 5
#define IN 64
#define OUT 32

struct param_conv  { float in[CIN][IN][IN]; float w[COUT][CIN][K][K]; };
struct result_conv { float out[COUT][OUT][OUT]; };

/* Couche 1 de l'encodeur de Balle : conv 5x5, stride 2, padding 2 */
void conv(struct param_conv *p, struct result_conv *r) {
    for (int co = 0; co < COUT; co++)
        for (int y = 0; y < OUT; y++)
            for (int x = 0; x < OUT; x++) {
                float acc = 0.0f;
                for (int ci = 0; ci < CIN; ci++)
                    for (int ky = 0; ky < K; ky++)
                        for (int kx = 0; kx < K; kx++) {
                            int iy = y * 2 + ky - 2, ix = x * 2 + kx - 2;
                            if (iy >= 0 && iy < IN && ix >= 0 && ix < IN)
                                acc += p->in[ci][iy][ix] * p->w[co][ci][ky][kx];
                        }
                r->out[co][y][x] = acc;
            }
}

static struct param_conv  P;
static struct result_conv R;

#ifndef LOCOD_FPGA
int main(void) {
    for (int c = 0; c < CIN; c++) for (int i = 0; i < IN; i++) for (int j = 0; j < IN; j++)
        P.in[c][i][j] = ((c * 31 + i * 7 + j * 13) % 256) / 255.0f;
    for (int o = 0; o < COUT; o++) for (int c = 0; c < CIN; c++) for (int i = 0; i < K; i++) for (int j = 0; j < K; j++)
        P.w[o][c][i][j] = (((o * 5 + c * 3 + i * 2 + j) % 17) - 8) / 64.0f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int n = 0; n < 5; n++) CPU(conv, &P, &R);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / 5;

    double sum = 0;
    for (int o = 0; o < COUT; o++) for (int y = 0; y < OUT; y++) for (int x = 0; x < OUT; x++) sum += R.out[o][y][x];
    printf("conv 3->128 5x5 s2 sur 64x64 : %.1f ms/appel (CPU) | checksum %.4f\n", ms, sum);
    return 0;
}
#endif
