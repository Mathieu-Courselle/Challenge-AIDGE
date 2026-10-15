# Challenge AIDGE – DEEPGREEN

Compression d'images par réseau de neurones (modèle de J. Ballé) sur carte **PYNQ-Z2**, avec la chaîne
**Aidge → Acetone → LoCoD** (Viveris / CNES). Mesures demandées : vitesse de compression, taux de compression, taux de perte.

## Chaîne

```
Modèle de Ballé (CompressAI / PyTorch)
   │ export ONNX                      -> 03_export_onnx
   ▼
Aidge  (import, optimisation, quantification)
   ▼
Acetone  (export du réseau en code C)
   ▼
LoCoD  (répartition CPU / FPGA)
   ├─ CPU  : compilation croisée ARM (Docker SDK)   -> 04_docker_sdk
   └─ FPGA : Panda-Bambu (C -> VHDL) puis Vivado (bitstream)
   ▼
Carte PYNQ-Z2 : fpgautil -b fpga.bin puis ./locod-cpu
```

## Contenu du dépôt

| Dossier | Rôle |
|---|---|
| `01_compressai/` | Référence sur PC : `test_cai.py` (une image), `bench_cai.py` (200 images, plusieurs qualités), `bench_models.py` (comparaison de 5 modèles CompressAI) |
| `02_convolution_cpu/` | Mesures de vitesse des 4 convolutions de l'encodeur sur la carte : `conv_bench.c` (v1) et `conv_bench2.c` (v2), plus `locod_conv_test/main.c` (programme LoCoD CPU seul) |
| `03_export_onnx/` | `export_balle.py` : exporte l'encodeur (`ga.onnx`) et le décodeur (`gs.onnx`), GDN réécrite en opérateurs standard |
| `04_docker_sdk/` | `Dockerfile` de l'image `sdk_pynqz2` (compilation ARM) |
| `docs/` | `resultats.md` (chiffres mesurés), `commandes.md` (commandes utiles) |

Les fichiers `.onnx` ne sont pas versionnés : on les régénère avec `export_balle.py`.

## Démarrage rapide

```bash
# 1. Référence PC
python3 -m venv ~/venv-cai && source ~/venv-cai/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install compressai
python 01_compressai/test_cai.py <image.png> 3
python 01_compressai/bench_cai.py 200

# 2. Export ONNX
pip install "numpy<2" onnx onnxscript onnxruntime
cd 03_export_onnx && python export_balle.py 3

# 3. Test de vitesse sur la carte (voir docs/commandes.md)
```

## Avancement

- [x] Carte branchée, SSH, exécution de programmes ARM
- [x] Chaîne LoCoD CPU seul (compilation croisée ARM)
- [x] Référence PC (taux, PSNR, vitesse)
- [x] Vitesse des convolutions sur la carte (naïf vs NEON)
- [x] Export ONNX encodeur + décodeur (vérifié identique à PyTorch)
- [ ] Import dans Aidge
- [ ] Export C avec Acetone
- [ ] Intégration dans LoCoD, test CPU sur la carte
- [ ] FPGA (Vivado 2022.2 + Panda-Bambu)
- [ ] Comparaison CPU / CPU + FPGA

## Questions pour Viveris

- Quel modèle exact : `bmshj2018_factorized` ou `mbt2018` (dépôt `thekoshkina/learned_image_compression`) ?
- Aidge gère-t-il les opérateurs utilisés (`Conv`, `ConvTranspose`, `Mul`, `Div`, `Sqrt`) ?
- Accès à Vivado 2022.2, à l'image Panda-Bambu et au dépôt privé de l'image SDK.
