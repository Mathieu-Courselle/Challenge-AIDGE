#!/bin/bash
# Lance tout le banc de test en une commande : preparer, compiler, envoyer, lancer sur la carte, comparer.
#
# A lancer depuis Debian, dans 05_benchmark_carte, avec l'environnement venv-cai actif (pour preparer les donnees) :
#   source ~/venv-cai/bin/activate
#   bash lancer_banc.sh              200 images, version de reference
#   bash lancer_banc.sh 50           50 images
#   bash lancer_banc.sh 200 acetone  ajoute la version avec le code Acetone (dossiers acetone_ga/ et/ou acetone_gs/)
#
# Reglages (variables d'environnement) :
#   BOARD_IP   adresse de la carte           (defaut 192.168.0.10)
#   WIN_STAGE  dossier d'echange cote Windows (defaut /mnt/c/Users/hugop/balle_carte)
#
# Le mot de passe de la carte est demande 3 fois (envoi, lancement, recuperation).
set -e
set -o pipefail
N=${1:-200}
MODE=${2:-}
BOARD_IP=${BOARD_IP:-192.168.0.10}
WIN_STAGE=${WIN_STAGE:-/mnt/c/Users/hugop/balle_carte}
cd "$(dirname "$0")"

etape() { echo ""; echo "=== $1"; }

etape "1/6 donnees ($N images)"
if [ -f board_data/weights.bin ] && [ -f board_data/images.bin ] && [ -f board_data/ref.csv ] \
   && [ "$(wc -c < board_data/images.bin)" -ge $((N * 12288)) ]; then
  echo "deja prepares dans board_data/ (on les reutilise)"
else
  python3 -c "import torch, compressai" 2>/dev/null || { echo "torch / compressai introuvables : lance d'abord  source ~/venv-cai/bin/activate"; exit 1; }
  python3 export_for_board.py "$N" 3 board_data
fi

etape "2/6 compilation pour la carte (docker sdk_pynqz2)"
docker run --rm -v "$PWD":/workdir sdk_pynqz2 bash -c '. /opt/petalinux-sdk/environment-setup-* && $CC -O2 -std=gnu11 balle.c -o balle -lm'
file balle
EXEC="balle"
CSV_DEMANDES="board.csv"
if [ "$MODE" = "acetone" ]; then
  [ -d acetone_ga ] || [ -d acetone_gs ] || { echo "mode acetone : il faut acetone_ga/ et/ou acetone_gs/ (voir README)"; exit 1; }
  bash build_acetone.sh
  EXEC="balle balle_acetone"
  CSV_DEMANDES="board.csv acetone.csv"
fi

etape "3/6 envoi sur la carte ($BOARD_IP)"
mkdir -p "$WIN_STAGE"
cp $EXEC board_data/weights.bin board_data/images.bin board_data/check.bin "$WIN_STAGE"/
( cd "$WIN_STAGE" && scp.exe $EXEC weights.bin images.bin check.bin root@"$BOARD_IP":/home/root/ )

etape "4/6 lancement sur la carte"
CMD="cd /home/root && chmod +x $EXEC && ./balle weights.bin images.bin $N board.csv check.bin"
[ "$MODE" = "acetone" ] && CMD="$CMD && ./balle_acetone weights.bin images.bin $N acetone.csv check.bin"
ssh.exe root@"$BOARD_IP" "$CMD" | tee resultats_banc.txt

etape "5/6 recuperation des resultats"
for f in $CSV_DEMANDES; do
  ( cd "$WIN_STAGE" && scp.exe root@"$BOARD_IP":/home/root/$f . )
  cp "$WIN_STAGE/$f" .
done

etape "6/6 comparaison"
ARGS="board_data/ref.csv:PC board.csv:Carte_C"
[ "$MODE" = "acetone" ] && ARGS="$ARGS acetone.csv:Carte_Acetone"
python3 compare.py $ARGS | tee comparaison.txt
echo ""
echo "termine : resultats dans comparaison.txt, resultats_banc.txt, board.csv"
