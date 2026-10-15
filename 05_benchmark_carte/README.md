# 05 – Benchmark complet sur la carte (200 images)

Compresse puis décompresse 200 images 64×64 du dataset **sur la PYNQ-Z2**, avec `bmshj2018_factorized`
réécrit en C (float32, NEON) et un codeur entropique rANS, puis compare au PC (PyTorch/CompressAI) :
**vitesse, taux de compression, perte (PSNR)**.

Aucun outil Aidge / Acetone / LoCoD n'est nécessaire : c'est une implémentation de référence écrite à la main.
Elle servira ensuite à valider le code C produit par Acetone (mêmes entrées, mêmes sorties attendues).

## Fichiers
| Fichier | Rôle |
|---|---|
| `export_for_board.py` | PC : exporte poids, tables du codeur entropique, 200 images, résultats PyTorch (`ref.csv`) |
| `balle_format.py` | Format du fichier de poids (numpy seul) |
| `balle.c` | Carte : encodeur + rANS + décodeur, mesure taille / PSNR / temps |
| `compare.py` | Tableau PC contre carte |
| `test_balle_c.py` + `inc/arm_neon.h` | Test sur PC x86 du code C contre une référence numpy (modèle aléatoire) |

## Utilisation
```bash
# 1. PC (Debian, venv-cai) : préparer les données (quelques minutes)
cd ~/Challenge-AIDGE/05_benchmark_carte
source ~/venv-cai/bin/activate
python export_for_board.py 200 3 board_data

# 2. Compiler pour la carte
docker run --rm -v $PWD:/workdir sdk_pynqz2 bash -c '. /opt/petalinux-sdk/environment-setup-* && $CC -O2 -std=gnu11 balle.c -o balle -lm'
file balle

# 3. Envoyer sur la carte (scp.exe : WSL2 n'atteint pas toujours la carte directement)
mkdir -p /mnt/c/Users/hugop/balle_carte
cp balle board_data/weights.bin board_data/images.bin board_data/check.bin /mnt/c/Users/hugop/balle_carte/
cd /mnt/c/Users/hugop/balle_carte
scp.exe balle weights.bin images.bin check.bin root@192.168.0.10:/home/root/

# 4. Essai rapide (5 images), puis les 200
ssh.exe root@192.168.0.10 "chmod +x balle && ./balle weights.bin images.bin 5 board.csv check.bin"
ssh.exe root@192.168.0.10 "./balle weights.bin images.bin 200 board.csv"

# 5. Récupérer et comparer
scp.exe root@192.168.0.10:/home/root/board.csv .
cp board.csv ~/Challenge-AIDGE/05_benchmark_carte/
cd ~/Challenge-AIDGE/05_benchmark_carte && python compare.py board_data/ref.csv board.csv
```

## Ce qu'il faut vérifier
- Ligne `VERIF image 0` : peu ou pas de symboles différents (un arrondi peut basculer à cause de l'ordre
  des additions en float), écart de reconstruction petit (< 1e-2).
- `aller-retour entropique : OK` : le décodeur retrouve exactement les symboles encodés.

## Limites
- Codeur rANS écrit pour ce test : l'échappement des valeurs hors plage est codé sur 16 bits, différemment de CompressAI.
  La taille peut différer de quelques octets par image de celle de CompressAI.
- La dernière déconvolution (3 canaux de sortie) est complétée à 8 canaux : environ 10 % de calcul inutile.
- float32 uniquement, pas de quantification int8 ; compilé en `-O2` (le NEON y est plus rapide qu'en `-O3`).
- Validé sur PC x86 (émulation NEON) contre une référence numpy avec des poids aléatoires ; la vérification avec les
  vrais poids PyTorch est la ligne `VERIF` ci-dessus.
