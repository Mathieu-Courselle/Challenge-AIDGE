"""Benchmark de bmshj2018_factorized sur N images du dataset, plusieurs qualites.
Usage : python bench_cai.py [N=200] [qualites ...]   (chemin du dataset a adapter ci-dessous)"""
import sys, math, time, glob, random, statistics as st, torch
from PIL import Image
from torchvision import transforms
from compressai.zoo import bmshj2018_factorized

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
QS = [int(a) for a in sys.argv[2:]] or [1, 3, 5, 8]
D = "/mnt/c/Users/hugop/Downloads/DataSet/DataSet/test"   # <- a adapter
files = sorted(glob.glob(D + "/*.png"))
random.seed(0)
files = random.sample(files, min(N, len(files)))
tt = transforms.ToTensor()
imgs = [tt(Image.open(f).convert("RGB")).unsqueeze(0) for f in files]

def verdict(p):
    if p < 28: return "dégradation visible"
    if p < 30: return "qualité moyenne"
    if p < 35: return "bonne qualité"
    if p < 40: return "très bonne qualité"
    return "quasi sans perte visible"

res = {}
for q in QS:
    net = bmshj2018_factorized(quality=q, pretrained=True).eval()
    net.update()
    bpps, psnrs, tcs, tds = [], [], [], []
    with torch.no_grad():
        net.compress(imgs[0])  # échauffement
        for x in imgs:
            h, w = x.shape[2:]
            H, W = (h + 63) // 64 * 64, (w + 63) // 64 * 64
            xp = torch.nn.functional.pad(x, (0, W - w, 0, H - h), mode="replicate")
            t0 = time.perf_counter(); out = net.compress(xp)
            t1 = time.perf_counter()
            rec = net.decompress(out["strings"], out["shape"])["x_hat"].clamp(0, 1)[:, :, :h, :w]
            t2 = time.perf_counter()
            nb = sum(len(s[0]) for s in out["strings"])
            mse = max(torch.mean((x - rec) ** 2).item(), 1e-10)
            bpps.append(nb * 8 / (h * w)); psnrs.append(-10 * math.log10(mse))
            tcs.append((t1 - t0) * 1000); tds.append((t2 - t1) * 1000)
    res[q] = dict(bpp=st.mean(bpps), bpp_med=st.median(bpps), ratio=24 / st.mean(bpps),
                  psnr=st.mean(psnrs), p_med=st.median(psnrs), p_min=min(psnrs), p_max=max(psnrs),
                  tc=st.mean(tcs), td=st.mean(tds), tc_max=max(tcs))

print(f"\nModèle bmshj2018-factorized | {len(imgs)} images 64x64 | brut = 24 bpp\n")
print(f"{'Q':>2} | {'bpp':>6} | {'taux':>7} | {'PSNR moy':>8} | {'PSNR méd':>8} | {'PSNR min':>8} | {'PSNR max':>8} | {'compr.':>9} | {'décompr.':>9}")
print("-" * 96)
for q, r in res.items():
    print(f"{q:>2} | {r['bpp']:6.3f} | {r['ratio']:5.1f}:1 | {r['psnr']:6.2f}dB | {r['p_med']:6.2f}dB | "
          f"{r['p_min']:6.2f}dB | {r['p_max']:6.2f}dB | {r['tc']:6.1f} ms | {r['td']:6.1f} ms")

print("\nAnalyse")
print("-" * 96)
for q, r in res.items():
    gain = 100 * (1 - r['bpp'] / 24)
    print(f"q{q}: fichier réduit de {gain:.1f} % ; {verdict(r['psnr'])} ({r['psnr']:.1f} dB) ; "
          f"écart pire/meilleure image : {r['p_max'] - r['p_min']:.0f} dB ; "
          f"débit ≈ {1000 / r['tc'] * 64 * 64 * 3 / 1e6:.2f} Mo/s (compression)")
if len(res) > 1:
    qs = list(res)
    a, b = res[qs[0]], res[qs[-1]]
    print(f"\nDe q{qs[0]} à q{qs[-1]}: +{b['psnr'] - a['psnr']:.1f} dB de PSNR, "
          f"taux divisé par {a['ratio'] / b['ratio']:.1f}, temps de compression x{b['tc'] / a['tc']:.1f}.")
    best = min(res, key=lambda k: abs(res[k]['psnr'] - 30))
    print(f"Qualité la plus proche de 30 dB (seuil 'bonne qualité') : q{best} ({res[best]['ratio']:.1f}:1).")
print("\nBarres de PSNR moyen (1 # = 1 dB) :")
for q, r in res.items():
    print(f"q{q} {'#' * int(r['psnr'])} {r['psnr']:.1f}")
print("\nRéférence PC pour la comparaison avec la carte PYNQ-Z2 (compression + décompression).")
