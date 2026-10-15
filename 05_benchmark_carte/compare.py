"""Compare ref.csv (PC, PyTorch) et board.csv (carte, programme C).
Usage : python compare.py ref.csv board.csv"""
import sys, csv

def load(p):
    with open(p) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]

ref, brd = load(sys.argv[1]), load(sys.argv[2])
n = min(len(ref), len(brd))
ref, brd = ref[:n], brd[:n]
mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)
print(f"{n} images 64x64 RGB (24 bits/pixel brut)\n")
print(f"{'':<26}{'PC (PyTorch)':>14}{'Carte (C)':>14}")
for label, key, fmt in [("octets moyens", "bytes", "{:.1f}"), ("PSNR moyen (dB)", "psnr", "{:.2f}"),
                        ("compression (ms/image)", "enc_ms", "{:.1f}"), ("decompression (ms/image)", "dec_ms", "{:.1f}")]:
    print(f"{label:<26}{fmt.format(mean(ref, key)):>14}{fmt.format(mean(brd, key)):>14}")
bpp = lambda rows: mean(rows, "bytes") * 8 / 4096
print(f"{'bits par pixel':<26}{bpp(ref):>14.3f}{bpp(brd):>14.3f}")
print(f"{'taux de compression':<26}{24 / bpp(ref):>12.1f}:1{24 / bpp(brd):>12.1f}:1")
dbytes = [abs(a['bytes'] - b['bytes']) for a, b in zip(ref, brd)]
dpsnr = [abs(a['psnr'] - b['psnr']) for a, b in zip(ref, brd)]
print(f"\necart de taille par image : moyen {sum(dbytes)/n:.1f} octets, max {max(dbytes):.0f}")
print(f"ecart de PSNR par image   : moyen {sum(dpsnr)/n:.3f} dB, max {max(dpsnr):.3f} dB")
print(f"carte / PC : compression x{mean(brd,'enc_ms')/mean(ref,'enc_ms'):.0f} plus lente, decompression x{mean(brd,'dec_ms')/mean(ref,'dec_ms'):.0f} plus lente")
