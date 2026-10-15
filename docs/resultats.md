# Résultats mesurés

## Référence PC – bmshj2018_factorized (CompressAI), 200 images 64×64 du dossier `test`

| Qualité | bpp | Taux | PSNR moyen | Compression | Décompression |
|---|---|---|---|---|---|
| 1 | 0,138 | 174:1 | 25,4 dB* | 3,5 ms | 3,8 ms |
| 3 | 0,314 | 76:1 | 27,6 dB* | 3,3 ms | 3,6 ms |
| 5 | 0,699 | 34:1 | 30,4 dB* | 3,7 ms | 4,2 ms |
| 8 | 2,114 | 11,4:1 | 35,4 dB* | 9,5 ms | 10,5 ms |

\* PSNR calculé à partir de l'erreur moyenne sur les 200 images. En moyennant le PSNR image par image
(`bench_cai.py` actuel) on obtient 28,2 / 30,5 / 33,0 / 37,8 dB. L'écart entre la pire et la meilleure image
est d'environ 23 dB : le modèle ne règle pas son débit selon le contenu.

## Carte PYNQ-Z2 – les 4 convolutions de l'encodeur (150,7 M MAC, convolutions seules)

Conv 5×5, stride 2, float32 : 3→128 (64×64), 128→128 (32×32), 128→128 (16×16), 128→192 (8×8).
GDN, quantification et codage entropique non inclus.

| Version | Encodeur (ms) | M MAC/s |
|---|---|---|
| Naïf, `-O2` (réglage actuel de LoCoD) | 3199 | 47 |
| Naïf, `-O3 -ffast-math` (vectorisation auto) | 1219 | 124 |
| **NEON à la main v1, `-O2`** | **667 – 717** | **210 – 226** |
| NEON v1, `-O3 -ffast-math` | 1251 – 1280 | 118 – 120 |
| NEON v2 (poids pré-rangés, blocs de 16 canaux) | 909 – 968 | 156 – 166 |

Détail par couche (naïf `-O2` → NEON v1 `-O2`) : L1 185 → 30 ms, L2 2240 → 512 ms, L3 559 → 126 ms, L4 210 → 49 ms.

Conclusions :
- NEON v1 en `-O2` est 4,5 à 4,8× plus rapide que le code naïf, avec un résultat identique.
- L'hypothèse « les poids ne tiennent pas dans le cache » n'est pas confirmée : v2 est plus lente que v1.
- Plafond observé vers 220 M MAC/s en float : on arrête l'optimisation CPU en float ici.
- Estimation : l'encodeur seul demande ≈ 0,7 s par image 64×64 sur le CPU, soit environ 200× le temps du PC.

## Programme LoCoD CPU seul (`locod_conv_test`)

Couche 1 (3→128, 64×64) : 156,2 ms par appel sur la carte, checksum 1039,4556 (identique sur PC).

## Export ONNX

- Encodeur `ga.onnx` : image 1×3×64×64 → 1×192×4×4 ; opérateurs `Conv`, `Div`, `Mul`, `Sqrt` ; écart avec PyTorch ≈ 5e-6.
- Décodeur `gs.onnx` : opérateurs `Conv`, `ConvTranspose`, `Mul`, `Sqrt` ; écart ≈ 5e-7.
- Environ 6 Mo chacun (≈ 1,5 M de paramètres en float32).
