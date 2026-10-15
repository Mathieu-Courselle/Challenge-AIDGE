"""Ecriture du fichier de poids lu par balle.c (numpy uniquement, sans PyTorch).

Entrees (en disposition PyTorch) :
  convs_a : 4 x (w[co,ci,5,5], b[co])        encodeur  (nn.Conv2d)
  gdns_a  : 3 x (gamma[C,C], beta[C])         GDN  : out_i = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2)
  convs_s : 4 x (w[ci,co,5,5], b[co])        decodeur (nn.ConvTranspose2d)
  gdns_s  : 3 x (gamma[C,C], beta[C])         IGDN : out_i = x_i * sqrt(...)
  medians[M], offsets[M], cdf_len[M], cdf[M,L]  : modele entropique (EntropyBottleneck)
Disposition ecrite : conv / deconv -> [ky][kx][ci][co] ; gamma -> [ci][co].
La derniere deconvolution (3 canaux de sortie) est completee a 8 canaux avec des zeros.
"""
import struct
import numpy as np

def _f32(a):
    return np.ascontiguousarray(a, dtype="<f4").tobytes()

def _i32(a):
    return np.ascontiguousarray(a, dtype="<i4").tobytes()

def write_weights(path, convs_a, gdns_a, convs_s, gdns_s, medians, offsets, cdf_len, cdf):
    N = convs_a[0][0].shape[0]
    M = convs_a[3][0].shape[0]
    L = cdf.shape[1]
    assert len(convs_a) == 4 and len(convs_s) == 4 and len(gdns_a) == 3 and len(gdns_s) == 3
    with open(path, "wb") as f:
        f.write(b"BAL1" + struct.pack("<3i", N, M, L))
        for i in range(4):
            w, b = convs_a[i]                                  # [co,ci,ky,kx]
            f.write(_f32(w.transpose(2, 3, 1, 0)))            # -> [ky][kx][ci][co]
            f.write(_f32(b))
            if i < 3:
                gamma, beta = gdns_a[i]
                f.write(_f32(gamma.T)); f.write(_f32(beta))   # gamma[co][ci] -> [ci][co]
        for i in range(4):
            w, b = convs_s[i]                                  # [ci,co,ky,kx]
            if i == 3:                                         # 3 -> 8 canaux de sortie
                wp = np.zeros((w.shape[0], 8, 5, 5), dtype=np.float32); wp[:, :w.shape[1]] = w
                bp = np.zeros(8, dtype=np.float32); bp[:b.shape[0]] = b
                w, b = wp, bp
            f.write(_f32(w.transpose(2, 3, 0, 1)))            # -> [ky][kx][ci][co]
            f.write(_f32(b))
            if i < 3:
                gamma, beta = gdns_s[i]
                f.write(_f32(gamma.T)); f.write(_f32(beta))
        f.write(_f32(medians)); f.write(_i32(offsets)); f.write(_i32(cdf_len)); f.write(_i32(cdf))
