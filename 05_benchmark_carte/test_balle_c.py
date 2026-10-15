"""Teste balle.c contre une reference numpy (modele aleatoire, memes formules que PyTorch).
Usage : python test_balle_c.py ./balle_x86"""
import sys, subprocess, numpy as np
from balle_format import write_weights

rng = np.random.default_rng(1)
N, M, L = 128, 192, 24

def conv_w(co, ci): return (rng.standard_normal((co, ci, 5, 5)) * np.sqrt(2.0 / (ci * 25))).astype(np.float32)
def deconv_w(ci, co): return (rng.standard_normal((ci, co, 5, 5)) * np.sqrt(2.0 / (ci * 25 / 4))).astype(np.float32)
def bias(c): return (rng.standard_normal(c) * 0.05).astype(np.float32)
def gdn_p(C): return ((np.abs(rng.standard_normal((C, C))) * 0.02).astype(np.float32), (1 + rng.random(C)).astype(np.float32))

convs_a = [(conv_w(N, 3), bias(N)), (conv_w(N, N), bias(N)), (conv_w(N, N), bias(N)), (conv_w(M, N) * 12, bias(M))]
gdns_a = [gdn_p(N) for _ in range(3)]
convs_s = [(deconv_w(M, N), bias(N)), (deconv_w(N, N), bias(N)), (deconv_w(N, N), bias(N)), (deconv_w(N, 3), bias(3))]
gdns_s = [gdn_p(N) for _ in range(3)]
medians = (rng.standard_normal(M) * 0.3).astype(np.float32)
cdf_len = np.array([14 + (c % 6) for c in range(M)], dtype=np.int32)
offsets = np.array([-((int(cdf_len[c]) - 2) // 2) for c in range(M)], dtype=np.int32)
cdf = np.zeros((M, L), dtype=np.int32)
for c in range(M):
    ns = int(cdf_len[c]) - 1                        # symboles reguliers + echappement
    p = rng.random(ns) + 0.2; p[-1] = 0.01
    f = np.floor(p / p.sum() * (65536 - ns)).astype(np.int64) + 1
    f[np.argmax(f)] += 65536 - f.sum()
    cdf[c, :ns + 1] = np.concatenate([[0], np.cumsum(f)])
    assert cdf[c, ns] == 65536 and cdf_len[c] == ns + 1

# ---- reference numpy (semantique PyTorch) ----
def conv(x, w, b):
    co, ci = w.shape[:2]; H, W = x.shape[1:]; oh, ow = H // 2, W // 2
    xp = np.pad(x, ((0, 0), (2, 2), (2, 2))); out = np.zeros((co, oh, ow), np.float32)
    for ky in range(5):
        for kx in range(5):
            out += np.einsum("oc,cyx->oyx", w[:, :, ky, kx], xp[:, ky:ky + 2 * oh:2, kx:kx + 2 * ow:2])
    return out + b[:, None, None]
def deconv(x, w, b):
    ci, co = w.shape[:2]; H, W = x.shape[1:]
    full = np.zeros((co, 2 * H + 4, 2 * W + 4), np.float32)
    for ky in range(5):
        for kx in range(5):
            full[:, ky:ky + 2 * H:2, kx:kx + 2 * W:2] += np.einsum("ic,iyx->cyx", w[:, :, ky, kx], x)
    return full[:, 2:2 + 2 * H, 2:2 + 2 * W] + b[:, None, None]
def gdn(x, gamma, beta, inverse):
    n = np.sqrt(np.einsum("ij,jyx->iyx", gamma, x * x) + beta[:, None, None])
    return x * n if inverse else x / n

def reference(img):                                   # img uint8 [64,64,3]
    x = (img.astype(np.float32) / 255).transpose(2, 0, 1)
    for i in range(3): x = gdn(conv(x, *convs_a[i]), *gdns_a[i], False)
    y = conv(x, *convs_a[3])
    sym = np.rint(y - medians[:, None, None]).astype(np.int32)
    z = sym.astype(np.float32) + medians[:, None, None]
    for i in range(3): z = gdn(deconv(z, *convs_s[i]), *gdns_s[i], True)
    xh = np.clip(deconv(z, *convs_s[3]), 0, 1)
    return sym, xh

n = 3
yy, xx = np.mgrid[0:64, 0:64]
imgs = np.stack([np.clip(np.stack([128 + 100 * np.sin((yy + k * 9) / 7.0 + c) * np.cos(xx / (5.0 + k)) for c in range(3)], -1)
                         + rng.normal(0, 6, (64, 64, 3)), 0, 255).astype(np.uint8) for k in range(n)])
sym0, xh0 = reference(imgs[0])
write_weights("w_test.bin", convs_a, gdns_a, convs_s, gdns_s, medians, offsets, cdf_len, cdf)
imgs.tofile("img_test.bin")
with open("chk_test.bin", "wb") as f:
    f.write(sym0.astype("<i4").tobytes()); f.write(xh0.transpose(1, 2, 0).astype("<f4").tobytes())
print("symboles : min", sym0.min(), "max", sym0.max(), "| hors plage (echappements) :",
      int(sum(((s - offsets[c]) < 0) | ((s - offsets[c]) >= cdf_len[c] - 2) for c in range(M) for s in sym0[c].ravel())))
r = subprocess.run([sys.argv[1], "w_test.bin", "img_test.bin", str(n), "out_test.csv", "chk_test.bin"], capture_output=True, text=True)
print(r.stdout, r.stderr)
print(open("out_test.csv").read())
