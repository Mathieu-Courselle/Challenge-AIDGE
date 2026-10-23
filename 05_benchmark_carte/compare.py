"""Compare plusieurs executions du banc de test : vitesse, taux de compression, perte (PSNR).
Usage : python compare.py fichier.csv:etiquette [fichier.csv:etiquette ...]
Le premier fichier est la reference (PC / PyTorch). Chaque autre colonne est comparee a lui.
Exemple : python compare.py board_data/ref.csv:PC board.csv:Carte_C acetone.csv:Carte_Acetone
Compatible avec l'ancien usage : python compare.py ref.csv board.csv"""
import sys, csv

def load(p):
    with open(p) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]

args = []
for a in sys.argv[1:]:
    chemin, _, nom = a.partition(":")
    args.append((chemin, nom or ("PC" if not args else "Carte")))
if len(args) < 2:
    sys.exit(__doc__)
runs = [(nom, load(p)) for p, nom in args]
n = min(len(r) for _, r in runs)
runs = [(nom, r[:n]) for nom, r in runs]
mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)
bpp = lambda rows: mean(rows, "bytes") * 8 / 4096

print(f"{n} images 64x64 RGB (24 bits/pixel brut)\n")
print(f"{'':<28}" + "".join(f"{nom:>16}" for nom, _ in runs))
lignes = [("octets moyens", lambda r: f"{mean(r, 'bytes'):.1f}"),
          ("bits par pixel", lambda r: f"{bpp(r):.3f}"),
          ("taux de compression", lambda r: f"{24 / bpp(r):.1f}:1"),
          ("perte : PSNR moyen (dB)", lambda r: f"{mean(r, 'psnr'):.2f}"),
          ("compression (ms/image)", lambda r: f"{mean(r, 'enc_ms'):.1f}"),
          ("decompression (ms/image)", lambda r: f"{mean(r, 'dec_ms'):.1f}")]
for label, f in lignes:
    print(f"{label:<28}" + "".join(f"{f(r):>16}" for _, r in runs))

ref_nom, ref = runs[0]
for nom, r in runs[1:]:
    db = [abs(a["bytes"] - b["bytes"]) for a, b in zip(ref, r)]
    dp = [abs(a["psnr"] - b["psnr"]) for a, b in zip(ref, r)]
    print(f"\n{nom} contre {ref_nom} :")
    print(f"  ecart de taille par image : moyen {sum(db)/n:.1f} octets, max {max(db):.0f}")
    print(f"  ecart de PSNR par image   : moyen {sum(dp)/n:.3f} dB, max {max(dp):.3f} dB")
    print(f"  compression x{mean(r,'enc_ms')/mean(ref,'enc_ms'):.0f} plus lente, decompression x{mean(r,'dec_ms')/mean(ref,'dec_ms'):.0f} plus lente")
