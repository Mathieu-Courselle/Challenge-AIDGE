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
| `compare.py` | Tableau des trois métriques pour 2 résultats ou plus (PC, carte, Acetone, FPGA…) |
| `build_acetone.sh` | Compile `balle.c` pour la carte en y branchant le code C exporté par Acetone |
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

## Brancher le code Acetone (ou LoCoD / FPGA)

Les trois métriques (vitesse, taux de compression, perte) se mesurent toujours de la même façon : même images, même arrondi,
même codeur rANS, même calcul du PSNR. Seul le réseau de neurones change. `balle.c` sait remplacer :

- l'encodeur `g_a` par `ga_forward()` (option `-DUSE_ACETONE_ENC`) ;
- le décodeur `g_s` par `gs_forward()` (option `-DUSE_ACETONE_DEC`).

Format attendu (c'est celui des exports Acetone de `ga.onnx` et `gs.onnx`, interface `model_forward(entrée, &sortie)`) :

| Fonction | Entrée | Sortie |
|---|---|---|
| `ga_forward(const float *x, float **y)` | `x` : 3×64×64 flottants, ordre CHW, valeurs entre 0 et 1 | `y` : 192×4×4, ordre CHW (canal, puis pixel) |
| `gs_forward(const float *y, float **x)` | `y` : 192×4×4, ordre CHW (valeurs décodées plus médianes) | `x` : 3×64×64, ordre CHW |

Le tampon de sortie appartient au code exporté (il met son adresse dans `*sortie`).

```bash
# 1. copier les exports Acetone (dossiers complets, avec dnn/src/forward.c)
cd ~/Challenge-AIDGE/05_benchmark_carte
cp -r /chemin/export_ga_test acetone_ga        # encodeur
cp -r /chemin/export_gs_test acetone_gs        # décodeur (facultatif)

# 2. compiler pour la carte (-O2, un seul cœur ; OMP=1 pour essayer -fopenmp, il faut alors libgomp sur la carte)
bash build_acetone.sh

# 3. envoyer et lancer comme avant (mêmes poids : médianes et tables du codeur entropique)
scp.exe balle_acetone weights.bin images.bin root@192.168.0.10:/home/root/
ssh.exe root@192.168.0.10 "chmod +x balle_acetone && ./balle_acetone weights.bin images.bin 200 acetone.csv"
scp.exe root@192.168.0.10:/home/root/acetone.csv .

# 4. comparer les trois versions
python compare.py board_data/ref.csv:PC board.csv:Carte_C acetone.csv:Carte_Acetone
```

Points d'attention :
- Le `Makefile` des exports Acetone compile en `-O0 -g` : c'est lent. `build_acetone.sh` compile en `-O2`.
- Le script renomme `model_forward` en `ga_forward` / `gs_forward` et rend privés les autres symboles (`objcopy`) : les deux exports peuvent avoir des tableaux de poids de même nom sans conflit.
- Si le code exporté utilise OpenMP, il faut `libgomp` sur la carte, sinon compiler sans (`OMP` non défini).
- Pour un FPGA via LoCoD : si le code généré garde la même interface (une fonction qui prend l'entrée et rend la sortie), il se branche de la même façon. Cela reste à vérifier avec les camarades.

### Test du branchement sur PC (sans Acetone)

L'option `-DSTUB_ACETONE` remplace `ga_forward` / `gs_forward` par des fonctions écrites avec le code de `balle.c`, mais qui parlent le format Acetone (CHW).
Le résultat doit être identique à la version de référence (mêmes octets, même PSNR, 0 symbole différent) : cela vérifie les conversions de format.

```bash
gcc -O2 -std=gnu11 -D__ARM_NEON -Iinc balle.c -DSTUB_ACETONE -DUSE_ACETONE_ENC -DUSE_ACETONE_DEC -o balle_stub -lm
python test_balle_c.py ./balle_stub
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
