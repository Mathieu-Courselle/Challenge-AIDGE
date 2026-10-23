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
| `03_export_onnx/` | Un dossier par modèle CompressAI (factorized, hyperprior, mbt2018-mean, mbt2018, cheng2020) avec `export_onnx.py` et les fichiers `.onnx` (GDN réécrite en opérateurs standard) |
| `04_docker_sdk/` | `Dockerfile` de l'image `sdk_pynqz2` (compilation ARM) |
| `05_benchmark_carte/` | Benchmark complet sur la carte : encodeur + codeur rANS + décodeur en C, 200 images, comparaison avec le PC |
| `06_test_fpga/` | Test du FPGA sur la carte : conversion du bitstream (`bit2bin.py`), rapport lisible (`rapport_fpga.py`), sortie du test |
| `docs/` | `resultats.md` (chiffres mesurés), `commandes.md` (commandes utiles) |

Les fichiers `.onnx` sont versionnés dans `03_export_onnx/<modèle>/`. On peut les régénérer avec `export_onnx.py`.

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
cd 03_export_onnx/bmshj2018-factorized && python export_onnx.py 3 64   # ou : bash 03_export_onnx/tout_exporter.sh 3 64

# 3. Test de vitesse sur la carte (voir docs/commandes.md)
```

## Avancement

- [x] Carte branchée, SSH, exécution de programmes ARM
- [x] Chaîne LoCoD CPU seul (compilation croisée ARM)
- [x] Référence PC (taux, PSNR, vitesse)
- [x] Vitesse des convolutions sur la carte (naïf vs NEON)
- [x] Export ONNX encodeur + décodeur (vérifié identique à PyTorch)
- [x] Export ONNX des 5 modèles CompressAI (`03_export_onnx/`)
- [x] Test du FPGA sur la carte avec un exemple LoCoD (`06_test_fpga/`)
- [x] Benchmark 200 images sur la carte (implémentation C de référence, `05_benchmark_carte/`) : voir tableau ci-dessous
- [x] Banc de test prêt à brancher le code Acetone / FPGA (`05_benchmark_carte/build_acetone.sh`, testé sur PC)
- [ ] Import dans Aidge
- [ ] Export C avec Acetone (encodeur bmshj2018-factorized généré, pas encore mesuré sur la carte)
- [ ] Intégration dans LoCoD, test CPU sur la carte
- [ ] FPGA (Vivado 2022.2 + Panda-Bambu)
- [ ] Comparaison CPU / CPU + FPGA

## Résultats actuels (200 images 64x64, bmshj2018_factorized q3)

| | PC (PyTorch) | Carte PYNQ-Z2 (C, CPU seul) |
|---|---|---|
| Octets moyens / image | 160.8 | 158.3 |
| PSNR moyen | 30.47 dB | 30.47 dB |
| Taux de compression | 76.4:1 | 77.6:1 |
| Compression | 3.5 ms | 922.7 ms |
| Décompression | 3.7 ms | 861.8 ms |

La carte donne le même résultat que le PC (0 symbole différent, écart PSNR max 0.027 dB) mais ~250x plus lentement : c'est la base de comparaison pour l'accélération FPGA. Détails dans `docs/resultats.md`.
