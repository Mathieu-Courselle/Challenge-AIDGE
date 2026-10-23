"""Image de comparaison : original, image apres compression + decompression, et carte des erreurs.

Usage :  python visuel_images.py [N=3] [modele=bmshj2018_factorized] [qualite=3]
Resultat : visuel_images.png (une ligne par image)

La compression est la vraie : image -> flux d'octets (codage rANS) -> image reconstruite.
La carte des erreurs montre ou l'image reconstruite differe de l'originale (erreur multipliee par 5 pour etre visible,
plus c'est clair, plus l'erreur est grande). Les images 64x64 sont agrandies a l'affichage sans lissage.
Le chemin du dataset est a adapter ci dessous."""
import sys, glob, random, math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/mnt/c/Users/hugop/Downloads/DataSet/DataSet/test"   # <- a adapter
AMPLI = 5
SORTIE = "visuel_images.png"


def calcule(fichiers, modele, qualite):
    import torch
    import compressai.zoo as zoo
    net = getattr(zoo, modele)(quality=qualite, pretrained=True).eval()
    net.update(force=True)
    res = []
    for f in fichiers:
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
        h, w = x.shape[2:]
        H, W = (h + 63) // 64 * 64, (w + 63) // 64 * 64
        xp = torch.nn.functional.pad(x, (0, W - w, 0, H - h), mode="replicate")
        with torch.no_grad():
            enc = net.compress(xp)
            xh = net.decompress(enc["strings"], enc["shape"])["x_hat"].clamp(0, 1)[:, :, :h, :w]
        rec = xh[0].permute(1, 2, 0).numpy()
        nb = sum(len(s[0]) for s in enc["strings"])
        mse = float(np.mean((rec - a) ** 2))
        res.append(dict(nom=f.split("/")[-1], orig=a, rec=rec, octets=nb,
                        psnr=99.0 if mse == 0 else 10 * math.log10(1.0 / mse)))
    return res


def figure(res, titre, sortie):
    n = len(res)
    fig, ax = plt.subplots(n, 3, figsize=(10.5, 3.6 * n), squeeze=False)
    for i, r in enumerate(res):
        h, w = r["orig"].shape[:2]
        brut = h * w * 3
        erreur = np.clip(np.abs(r["orig"] - r["rec"]).mean(axis=2) * AMPLI, 0, 1)
        ax[i][0].imshow(r["orig"], interpolation="nearest")
        ax[i][0].set_title("Original (%d octets)" % brut, fontsize=10)
        ax[i][1].imshow(r["rec"], interpolation="nearest")
        ax[i][1].set_title("Apres compression : %d octets (%.1f:1), %.2f dB" % (
            r["octets"], brut / r["octets"], r["psnr"]), fontsize=10)
        ax[i][2].imshow(erreur, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[i][2].set_title("Erreur (x%d)" % AMPLI, fontsize=10)
        for a in ax[i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(titre, fontsize=12)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    modele = sys.argv[2] if len(sys.argv) > 2 else "bmshj2018_factorized"
    qualite = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    fichiers = sorted(glob.glob(D + "/*.png"))
    random.seed(1)
    fichiers = random.sample(fichiers, n)
    res = calcule(fichiers, modele, qualite)
    for r in res:
        print(r["nom"], ":", r["octets"], "octets,", "%.2f dB" % r["psnr"])
    figure(res, "%s, qualite %d : avant et apres compression" % (modele, qualite), SORTIE)
    print("Image enregistree :", SORTIE)


if __name__ == "__main__":
    main()
