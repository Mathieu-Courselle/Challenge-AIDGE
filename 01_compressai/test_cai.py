"""Test d'une image avec bmshj2018_factorized (CompressAI).
Usage : python test_cai.py image.png [qualite 1-8]"""
import sys, math, time, torch
from PIL import Image
from torchvision import transforms
from compressai.zoo import bmshj2018_factorized

path = sys.argv[1]
q = int(sys.argv[2]) if len(sys.argv) > 2 else 3
net = bmshj2018_factorized(quality=q, pretrained=True).eval()
net.update()
x = transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)
h, w = x.shape[2:]
H, W = (h + 63) // 64 * 64, (w + 63) // 64 * 64
xp = torch.nn.functional.pad(x, (0, W - w, 0, H - h), mode="replicate")
with torch.no_grad():
    t0 = time.time()
    out = net.compress(xp)
    t1 = time.time()
    rec = net.decompress(out["strings"], out["shape"])["x_hat"].clamp(0, 1)[:, :, :h, :w]
    t2 = time.time()
nbytes = sum(len(s[0]) for s in out["strings"])
mse = torch.mean((x - rec) ** 2).item()
print(f"{w}x{h} | qualité {q} | {nbytes} octets | {nbytes*8/(h*w):.3f} bpp | "
      f"PSNR {-10*math.log10(mse):.2f} dB | taux {(h*w*24)/(nbytes*8):.1f}:1 | "
      f"compression {t1-t0:.2f}s | décompression {t2-t1:.2f}s")
