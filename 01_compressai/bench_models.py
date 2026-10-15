"""Compare 5 modeles CompressAI (taux, PSNR, parametres, MAC, temps carte estime).
Usage : python bench_models.py [N=200] [qualite=3]"""
import sys, math, time, glob, random, statistics as st, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior,
                            mbt2018_mean, mbt2018, cheng2020_anchor)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
Q = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BOARD_MMACS = 63.0  # vitesse mesuree sur la PYNQ-Z2 (M MAC/s, code naif)
MODELS = {"factorized": bmshj2018_factorized, "hyperprior": bmshj2018_hyperprior,
          "mbt2018_mean": mbt2018_mean, "mbt2018": mbt2018, "cheng2020_anchor": cheng2020_anchor}

D = "/mnt/c/Users/hugop/Downloads/DataSet/DataSet/test"   # <- a adapter
files = sorted(glob.glob(D + "/*.png"))
random.seed(0)
files = random.sample(files, min(N, len(files)))
tt = transforms.ToTensor()
imgs = [tt(Image.open(f).convert("RGB")).unsqueeze(0) for f in files]

def count_macs(net, x):
    tot = [0]
    def hc(m, i, o): tot[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    def ht(m, i, o): tot[0] += i[0].numel() * (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    hs = []
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d): hs.append(m.register_forward_hook(ht))
        elif isinstance(m, nn.Conv2d): hs.append(m.register_forward_hook(hc))
    with torch.no_grad(): net(x)
    for h in hs: h.remove()
    return tot[0]

rows = []
for name, fn in MODELS.items():
    try:
        net = fn(quality=Q, pretrained=True).eval()
        params = sum(p.numel() for p in net.parameters())
        macs = count_macs(net, imgs[0])
        net.update()
        bpps, psnrs, tcs = [], [], []
        with torch.no_grad():
            for x in imgs:
                h, w = x.shape[2:]
                H, W = (h + 63) // 64 * 64, (w + 63) // 64 * 64
                xp = torch.nn.functional.pad(x, (0, W - w, 0, H - h), mode="replicate")
                t0 = time.perf_counter(); out = net.compress(xp); t1 = time.perf_counter()
                rec = net.decompress(out["strings"], out["shape"])["x_hat"].clamp(0, 1)[:, :, :h, :w]
                nb = sum(len(s[0]) for s in out["strings"])
                mse = max(torch.mean((x - rec) ** 2).item(), 1e-10)
                bpps.append(nb * 8 / (h * w)); psnrs.append(-10 * math.log10(mse)); tcs.append((t1 - t0) * 1000)
        rows.append((name, params / 1e6, params * 4 / 1e6, macs / 1e6, macs / 1e6 / BOARD_MMACS,
                     st.mean(bpps), 24 / st.mean(bpps), st.mean(psnrs), st.mean(tcs)))
    except Exception as e:
        print(f"{name}: erreur -> {e}")

print(f"\n{len(imgs)} images 64x64 | qualite {Q} | MAC = encodeur + decodeur (estimation)\n")
print(f"{'modele':<17}|{'params M':>9}|{'poids Mo':>9}|{'M MAC':>8}|{'carte est.':>11}|{'bpp':>6}|{'taux':>8}|{'PSNR':>9}|{'PC compr.':>10}")
print("-" * 100)
for r in rows:
    print(f"{r[0]:<17}|{r[1]:9.2f}|{r[2]:9.1f}|{r[3]:8.0f}|{r[4]:9.1f} s|{r[5]:6.3f}|{r[6]:6.1f}:1|{r[7]:6.2f} dB|{r[8]:7.1f} ms")
if rows:
    best = max(rows, key=lambda r: r[7]); light = min(rows, key=lambda r: r[3])
    print(f"\nMeilleure qualite (PSNR) : {best[0]} ({best[7]:.1f} dB)")
    print(f"Plus leger en calcul     : {light[0]} ({light[3]:.0f} M MAC, ~{light[4]:.1f} s/image sur la carte)")
    print("Rappel : 'carte est.' = M MAC / 63 M MAC/s, code C naif sans NEON ni FPGA (ordre de grandeur).")
