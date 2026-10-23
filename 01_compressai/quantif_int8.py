"""Etude de quantification entiere (int8) du modele bmshj2018_factorized, simulee sur PC.

Question posee : si les convolutions du codeur (g_a) et du decodeur (g_s) travaillent en entiers
(comme sur le FPGA) au lieu de flottants, combien de qualite et de debit perd-on ?

Ce que le script quantifie (comme sur la chaine cible) :
  les poids et les entrees des convolutions et des convolutions transposees, qui partiraient sur le FPGA
Ce qu'il laisse en flottant (comme sur la chaine cible, ca reste sur le CPU) :
  les GDN / IGDN, les biais, le modele d'entropie et le codage rANS

Methode : quantification symetrique, "fausse quantification" (on arrondit a la grille entiere puis on
revient en flottant, ce qui donne exactement les memes valeurs qu'un calcul entier suivi d'un rescale).
  poids   : une echelle par canal de sortie
  entrees : une echelle par couche, fixee sur 20 images de calibration (percentile 99,99 %)
La calibration et la mesure utilisent des images differentes.

Debit estime par la somme des -log2 des vraisemblances (identique a ce que fait le codeur rANS, a quelques
octets d'en-tete pres), PSNR calcule sur l'image reconstruite comprise entre 0 et 1.

Usage :  python quantif_int8.py [N=100]      (resultat dans quantif_int8.csv)
Le chemin du dataset est a adapter ci dessous."""
import sys, os, csv, glob, math, random
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import compressai.zoo as zoo

D = "/mnt/c/Users/hugop/Downloads/DataSet/DataSet/test"   # <- a adapter
MODELE, QUALITE = "bmshj2018_factorized", 3
N_CALIB = 20
CSV = "quantif_int8.csv"
# (nom, bits poids, bits entrees) ; None = flottant
VARIANTES = [
    ("float (reference)", None, None),
    ("poids 8 bits", 8, None),
    ("poids 8 + entrees 8 (int8)", 8, 8),
    ("poids 6 + entrees 8", 6, 8),
    ("poids 4 + entrees 8", 4, 8),
    ("poids 8 + entrees 6", 8, 6),
    ("poids 6 + entrees 6", 6, 6),
    ("poids 4 + entrees 4", 4, 4),
]

ETAT = {"mode": "normal", "abits": None}


def fait_hook():
    def hook(mod, inp):
        x = inp[0]
        if ETAT["mode"] == "calib":
            v = x.abs().flatten()
            if v.numel() > 1000000:
                v = v[:: v.numel() // 1000000 + 1]
            mod._cal.append(torch.quantile(v, 0.9999).item())
            return None
        if ETAT["abits"]:
            qmax = 2 ** (ETAT["abits"] - 1) - 1
            s = mod._amax / qmax
            return (torch.clamp(torch.round(x / s), -qmax, qmax) * s,)
        return None
    return hook


def quant_poids(m, bits):
    w = m._w0
    qmax = 2 ** (bits - 1) - 1
    # une echelle par canal de sortie (dim 0 pour Conv2d, dim 1 pour ConvTranspose2d)
    dims = (0, 2, 3) if isinstance(m, nn.ConvTranspose2d) else (1, 2, 3)
    amax = w.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12)
    s = amax / qmax
    return torch.clamp(torch.round(w / s), -qmax, qmax) * s


def regle(convs, wbits, abits):
    for m in convs:
        m.weight.data.copy_(m._w0 if wbits is None else quant_poids(m, wbits))
    ETAT["abits"] = abits


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    files = sorted(glob.glob(D + "/*.png"))
    random.seed(0)
    files = random.sample(files, min(n, len(files)))
    tt = transforms.ToTensor()
    imgs = [tt(Image.open(f).convert("RGB")).unsqueeze(0) for f in files]
    calib, test = (imgs[:N_CALIB], imgs[N_CALIB:]) if len(imgs) > N_CALIB + 10 else (imgs, imgs)
    print(len(imgs), "images :", len(calib), "calibration,", len(test), "mesure")

    net = getattr(zoo, MODELE)(quality=QUALITE, pretrained=True).eval()
    convs = [m for sub in (net.g_a, net.g_s) for m in sub.modules()
             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    print(len(convs), "convolutions quantifiees (g_a et g_s)")
    nparam = sum(m.weight.numel() for m in convs)
    for m in convs:
        m._w0 = m.weight.data.clone()
        m._cal = []
        m._amax = 1.0
        m.register_forward_pre_hook(fait_hook())

    def prep(x):
        h, w = x.shape[2:]
        H, W = (h + 63) // 64 * 64, (w + 63) // 64 * 64
        return torch.nn.functional.pad(x, (0, W - w, 0, H - h), mode="replicate"), h, w

    # calibration des echelles des entrees (poids et entrees flottants)
    regle(convs, None, None)
    ETAT["mode"] = "calib"
    with torch.no_grad():
        for x in calib:
            net(prep(x)[0])
    for m in convs:
        m._amax = max(sum(m._cal) / len(m._cal), 1e-8)
    ETAT["mode"] = "normal"

    # latents de reference (flottant) pour compter les symboles qui changent
    regle(convs, None, None)
    with torch.no_grad():
        yref = [torch.round(net.g_a(prep(x)[0])) for x in test]

    lignes = []
    for nom, wb, ab in VARIANTES:
        regle(convs, wb, ab)
        ps, bs, diff, tot = [], [], 0, 0
        with torch.no_grad():
            for x, yr in zip(test, yref):
                xp, h, w = prep(x)
                out = net(xp)
                xh = out["x_hat"].clamp(0, 1)[:, :, :h, :w]
                bpp = sum(-torch.log2(l).sum() for l in out["likelihoods"].values()).item() / (h * w)
                ps.append(psnr(xh, x))
                bs.append(bpp)
                yq = torch.round(net.g_a(xp))
                diff += (yq != yr).sum().item()
                tot += yr.numel()
        lignes.append((nom, wb, ab, sum(ps) / len(ps), sum(bs) / len(bs), 100.0 * diff / tot))
        print(nom, ": PSNR %.2f dB, %.3f bpp, symboles differents %.2f %%" % lignes[-1][3:])
    regle(convs, None, None)

    ref = lignes[0]
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variante", "bits_poids", "bits_entrees", "psnr", "bpp", "delta_psnr", "delta_bpp_pct", "symboles_differents_pct"])
        for l in lignes:
            w.writerow([l[0], l[1], l[2], "%.3f" % l[3], "%.4f" % l[4], "%.3f" % (l[3] - ref[3]),
                        "%.2f" % (100 * (l[4] / ref[4] - 1)), "%.3f" % l[5]])

    print("\n| Variante | PSNR (dB) | Perte PSNR | Debit (bpp) | Variation debit | Symboles differents |")
    print("|---|---|---|---|---|---|")
    for l in lignes:
        print("| %s | %.2f | %+.2f | %.3f | %+.1f %% | %.2f %% |" % (
            l[0], l[3], l[3] - ref[3], l[4], 100 * (l[4] / ref[4] - 1), l[5]))
    print("\nPoids des convolutions : %.2f Mo en float32, %.2f Mo en int8 (%d parametres)" % (
        nparam * 4 / 1e6, nparam / 1e6, nparam))
    print("Resultats enregistres dans", CSV)


if __name__ == "__main__":
    main()
