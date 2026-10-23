"""Courbes qualite (PSNR) en fonction du debit pour les 5 modeles CompressAI, sur plusieurs niveaux de qualite.
Permet de comparer les modeles A DEBIT EGAL (meme taux de compression) ou A QUALITE EGALE.

Usage :
  python courbes_modeles.py [N=100] [qualites ...]     mesure (ex : python courbes_modeles.py 100 1 2 3 4 5 6 7 8)
  python courbes_modeles.py --analyse                  refait les tableaux et le graphique depuis courbes_modeles.csv

Les mesures sont enregistrees au fur et a mesure dans courbes_modeles.csv : si le script s'arrete
(reseau coupe, erreur), relancer la meme commande reprend la ou il s'etait arrete.
Les poids de chaque modele et qualite sont telecharges au premier passage (environ 1 Go au total).
Le chemin du dataset est a adapter ci dessous."""
import sys, os, csv, math, time, glob, random, statistics as st

CSV = "courbes_modeles.csv"
D = "/mnt/c/Users/hugop/Downloads/DataSet/DataSet/test"   # <- a adapter
MODELES = ["bmshj2018_factorized", "bmshj2018_hyperprior", "mbt2018_mean", "mbt2018", "cheng2020_anchor"]
QUALITES_MAX = {"cheng2020_anchor": 6}     # cheng2020 n'a que les qualites 1 a 6


def mesure(n, qualites):
    import torch
    from PIL import Image
    from torchvision import transforms
    import compressai.zoo as zoo

    fait = set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            for r in csv.DictReader(f):
                fait.add((r["modele"], int(r["qualite"])))
    nouveau = not os.path.exists(CSV)

    files = sorted(glob.glob(D + "/*.png"))
    random.seed(0)
    files = random.sample(files, min(n, len(files)))
    tt = transforms.ToTensor()
    imgs = [tt(Image.open(f).convert("RGB")).unsqueeze(0) for f in files]
    print(len(imgs), "images")

    with open(CSV, "a", newline="") as fo:
        w = csv.writer(fo)
        if nouveau:
            w.writerow(["modele", "qualite", "bpp", "psnr", "images"])
        for nom in MODELES:
            for q in qualites:
                if q > QUALITES_MAX.get(nom, 8):
                    continue
                if (nom, q) in fait:
                    print(nom, "q" + str(q), ": deja mesure")
                    continue
                try:
                    net = getattr(zoo, nom)(quality=q, pretrained=True).eval()
                    net.update(force=True)
                    bpps, psnrs = [], []
                    t0 = time.time()
                    with torch.no_grad():
                        for x in imgs:
                            h, wd = x.shape[2:]
                            H, W = (h + 63) // 64 * 64, (wd + 63) // 64 * 64
                            xp = torch.nn.functional.pad(x, (0, W - wd, 0, H - h), mode="replicate")
                            out = net.compress(xp)
                            rec = net.decompress(out["strings"], out["shape"])["x_hat"].clamp(0, 1)[:, :, :h, :wd]
                            nb = sum(len(s[0]) for s in out["strings"])
                            mse = max(torch.mean((x - rec) ** 2).item(), 1e-10)
                            bpps.append(nb * 8 / (h * wd))
                            psnrs.append(-10 * math.log10(mse))
                    w.writerow([nom, q, "%.5f" % st.mean(bpps), "%.4f" % st.mean(psnrs), len(imgs)])
                    fo.flush()
                    print("%-22s q%d : %.3f bpp | %.2f dB | %.0f s" % (nom, q, st.mean(bpps), st.mean(psnrs), time.time() - t0))
                except Exception as e:
                    print(nom, "q" + str(q), ": erreur ->", repr(e)[:200])


def charge():
    data = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            data.setdefault(r["modele"], []).append((int(r["qualite"]), float(r["bpp"]), float(r["psnr"])))
    for k in data:
        data[k].sort()
    return data


def interp(xs, ys, x):
    """interpolation lineaire, None hors de la plage des mesures (xs croissants)."""
    if len(xs) < 2 or x < xs[0] or x > xs[-1]:
        return None
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return ys[i]
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])


def analyse():
    data = charge()
    noms = [m for m in MODELES if m in data]
    print("\n1. mesures (une ligne par niveau de qualite)\n")
    print("%-22s %7s %8s %10s %9s" % ("modele", "qualite", "bpp", "taux", "PSNR dB"))
    for m in noms:
        for q, b, p in data[m]:
            print("%-22s %7d %8.3f %8.1f:1 %9.2f" % (m, q, b, 24 / b, p))

    courbes = {}
    for m in noms:
        pts = sorted((b, p) for _, b, p in data[m])
        courbes[m] = ([b for b, _ in pts], [p for _, p in pts])

    print("\n2. qualite (PSNR en dB) a debit egal : le meme nombre d'octets pour tous\n")
    cibles = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    print("%-22s" % "debit (bpp)" + "".join("%9.2f" % c for c in cibles))
    print("%-22s" % "taux de compression" + "".join("%7.0f:1" % (24 / c) for c in cibles))
    for m in noms:
        xs, ys = courbes[m]
        print("%-22s" % m + "".join(("%9.2f" % v if v is not None else "%9s" % "-") for v in (interp(xs, ys, c) for c in cibles)))

    print("\n3. debit (bpp) necessaire a qualite egale : plus c'est petit, mieux c'est\n")
    psn = [28, 30, 32, 34, 36]
    print("%-22s" % "PSNR vise (dB)" + "".join("%9d" % c for c in psn))
    inv = {}
    for m in noms:
        pts = sorted((p, b) for b, p in zip(*courbes[m]))
        inv[m] = ([p for p, _ in pts], [b for _, b in pts])
        print("%-22s" % m + "".join(("%9.3f" % v if v is not None else "%9s" % "-") for v in (interp(inv[m][0], inv[m][1], c) for c in psn)))
    ref = "bmshj2018_factorized"
    if ref in inv:
        print("\n   economie de debit par rapport a %s (positif = moins d'octets pour la meme qualite)\n" % ref)
        print("%-22s" % "PSNR vise (dB)" + "".join("%9d" % c for c in psn))
        for m in noms:
            if m == ref:
                continue
            lig = []
            for c in psn:
                a, b = interp(inv[ref][0], inv[ref][1], c), interp(inv[m][0], inv[m][1], c)
                lig.append("%8.0f%%" % (100 * (a - b) / a) if a and b else "%9s" % "-")
            print("%-22s" % m + "".join(lig))
    print("\n   '-' : la qualite visee est hors de la plage mesuree pour ce modele.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5.5))
        for m in noms:
            xs, ys = courbes[m]
            plt.plot(xs, ys, marker="o", label=m)
        plt.xlabel("debit (bits par pixel)  : a gauche = plus compresse")
        plt.ylabel("PSNR (dB)  : en haut = meilleure qualite")
        plt.title("Qualite en fonction du debit, 5 modeles CompressAI")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("courbes_modeles.png", dpi=150)
        print("\ngraphique enregistre : courbes_modeles.png")
    except ImportError:
        print("\npas de graphique : pip install matplotlib puis  python courbes_modeles.py --analyse")


if __name__ == "__main__":
    if "--analyse" in sys.argv:
        analyse()
    else:
        args = [a for a in sys.argv[1:] if a.isdigit()]
        n = int(args[0]) if args else 100
        qs = [int(a) for a in args[1:]] or list(range(1, 9))
        mesure(n, qs)
        analyse()
