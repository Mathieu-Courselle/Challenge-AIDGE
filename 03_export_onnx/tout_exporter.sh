#!/bin/bash
# Lance l'export ONNX des 5 modeles. Usage : bash tout_exporter.sh [qualite] [taille_image]
# A lancer depuis Debian avec l'environnement venv-cai actif, dans le dossier contenant les 5 dossiers.
Q=${1:-3}
S=${2:-64}
for d in bmshj2018-factorized bmshj2018-hyperprior mbt2018-mean mbt2018 cheng2020-; do
  echo "================ $d ================"
  (cd "$d" && python export_onnx.py "$Q" "$S") || echo "ECHEC pour $d"
done
echo "fichiers .onnx crees :"
ls */*.onnx
