"""Prepare les donnees du benchmark carte pour bmshj2018_factorized (CompressAI).
Produit dans le dossier de sortie :
  weights.bin  poids du modele + tables du codeur entropique (lus par balle.c)
  images.bin   N images 64x64 RGB (uint8) du dossier test, tirage identique a bench_cai.py
  ref.csv      resultats PyTorch/CompressAI sur le PC (octets, PSNR, temps) pour chaque image
  check.bin    symboles + image reconstruite de l'image 0 (verification du code C)
Usage : python export_for_board.py [N=200] [qualite=3] [dossier_sortie=board_data]
"""
import sys, os, glob, random, math, time
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from compressai.zoo import bmshj2018_factorized
from compressai.layers import GDN
from balle_format import write_weights

N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 200
Q = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = sys.argv[3] if len(sys.argv) > 3 else "board_data"
DATASET = "/mnt/c/Users/hugop/Downloads/DataSet/DataSet/test"   # <- a adapter
os.makedirs(OUT, exist_ok=True)

net = bmshj2018_factorized(quality=Q, pretrained=True).eval()
net.update(force=True)
eb = net.entropy_bottleneck
np32 = lambda t: t.detach().cpu().numpy().astype(np.float32)

def gdn_params(g):
    with torch.no_grad():
        return np32(g.gamma_reparam(g.gamma)), np32(g.beta_reparam(g.beta))

convs_a = [(np32(m.weight), np32(m.bias)) for m in net.g_a if isinstance(m, nn.Conv2d)]
gdns_a = [gdn_params(m) for m in net.g_a if isinstance(m, GDN)]
convs_s = [(np32(m.weight), np32(m.bias)) for m in net.g_s if isinstance(m, nn.ConvTranspose2d)]
gdns_s = [gdn_params(m) for m in net.g_s if isinstance(m, GDN)]
medians = np32(eb._get_medians()).reshape(-1)
offsets = eb._offset.detach().cpu().numpy().astype(np.int32).reshape(-1)
cdf_len = eb._cdf_length.detach().cpu().numpy().astype(np.int32).reshape(-1)
cdf = eb._quantized_cdf.detach().cpu().numpy().astype(np.int32)
assert len(convs_a) == 4 and len(convs_s) == 4 and len(gdns_a) == 3 and len(gdns_s) == 3
assert all(int(cdf[c, cdf_len[c] - 1]) == 65536 for c in range(len(cdf_len))), "tables CDF inattendues"
write_weights(os.path.join(OUT, "weights.bin"), convs_a, gdns_a, convs_s, gdns_s, medians, offsets, cdf_len, cdf)
print(f"poids ecrits : N={convs_a[0][0].shape[0]} M={convs_a[3][0].shape[0]} longueur max des CDF={cdf.shape[1]}")

files = sorted(glob.glob(DATASET + "/*.png"))
random.seed(0)
files = random.sample(files, min(N_IMG, len(files)))
tt = transforms.ToTensor()
raw, rows = [], []
with torch.no_grad():
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        assert im.size == (64, 64), f"image {f} : {im.size} (attendu 64x64)"
        raw.append(np.asarray(im, dtype=np.uint8))
        x = tt(im).unsqueeze(0)
        t0 = time.perf_counter(); out = net.compress(x); t1 = time.perf_counter()
        rec = net.decompress(out["strings"], out["shape"])["x_hat"].clamp(0, 1); t2 = time.perf_counter()
        nb = len(out["strings"][0][0])
        mse = max(torch.mean((x - rec) ** 2).item(), 1e-10)
        rows.append((i, nb, -10 * math.log10(mse), (t1 - t0) * 1000, (t2 - t1) * 1000))
        if i == 0:
            y = net.g_a(x)
            med = eb._get_medians().detach().reshape(1, -1, 1, 1)
            sym = torch.round(y - med).to(torch.int32).reshape(-1).numpy()
            with open(os.path.join(OUT, "check.bin"), "wb") as fc:
                fc.write(sym.astype("<i4").tobytes())
                fc.write(rec[0].permute(1, 2, 0).contiguous().numpy().astype("<f4").tobytes())
np.stack(raw).tofile(os.path.join(OUT, "images.bin"))
with open(os.path.join(OUT, "ref.csv"), "w") as fr:
    fr.write("idx,bytes,psnr,enc_ms,dec_ms\n")
    for r in rows: fr.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.3f},{r[4]:.3f}\n")
a = np.array(rows)
print(f"{len(rows)} images | PyTorch/PC : octets moy {a[:,1].mean():.1f} | {a[:,1].mean()*8/4096:.3f} bpp | "
      f"taux {4096*24/(a[:,1].mean()*8):.1f}:1 | PSNR moy {a[:,2].mean():.2f} dB | "
      f"compression {a[:,3].mean():.1f} ms | decompression {a[:,4].mean():.1f} ms")
print("fichiers dans", OUT, ":", ", ".join(sorted(os.listdir(OUT))))
