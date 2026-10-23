#!/bin/bash
# Compile balle.c pour la carte en branchant le code C exporte par Acetone.
#
# A lancer dans 05_benchmark_carte, avec l'image Docker sdk_pynqz2.
# Placer les exports Acetone dans :
#   acetone_ga/   export Acetone de l'encodeur (ga.onnx)   -> remplace g_a
#   acetone_gs/   export Acetone du decodeur  (gs.onnx)    -> remplace g_s
# On peut n'en mettre qu'un des deux : l'autre reste l'implementation de reference.
#
# Chaque dossier doit contenir dnn/src/forward.c avec la fonction
#   void model_forward(const float *entree, float **sortie);
# (c'est le cas de l'export de ga.onnx deja genere).
#
# Option : OMP=1 bash build_acetone.sh   pour compiler avec -fopenmp (necessite libgomp sur la carte).
# Resultat : ./balle_acetone  (meme usage que ./balle)
set -e
[ -d acetone_ga ] || [ -d acetone_gs ] || { echo "rien a faire : ni acetone_ga/ ni acetone_gs/"; exit 1; }

cat > .build_inner.sh <<'EOS'
set -e
. /opt/petalinux-sdk/environment-setup-*
OMPF=""; [ "$OMP" = "1" ] && OMPF="-fopenmp"
DEFS=""; OBJS=""
build_one() {   # $1 = dossier de l'export, $2 = prefixe (ga ou gs)
  d=$1; n=$2
  $CC -O2 -std=gnu11 $OMPF -I$d -I$d/include -I$d/dnn -I$d/dnn/include -I$d/dnn/layers -I$d/dnn/parameters -I$d/dnn/include/parameters \
      -c $d/dnn/src/forward.c -o $n.o
  # on renomme model_forward en ga_forward / gs_forward et on cache tout le reste (evite les conflits de noms entre exports)
  $OBJCOPY --redefine-sym model_forward=${n}_forward $n.o
  $OBJCOPY --keep-global-symbol=${n}_forward $n.o
}
if [ -d acetone_ga ]; then build_one acetone_ga ga; DEFS="$DEFS -DUSE_ACETONE_ENC"; OBJS="$OBJS ga.o"; fi
if [ -d acetone_gs ]; then build_one acetone_gs gs; DEFS="$DEFS -DUSE_ACETONE_DEC"; OBJS="$OBJS gs.o"; fi
$CC -O2 -std=gnu11 $OMPF $DEFS balle.c $OBJS -o balle_acetone -lm
echo "compile : balle_acetone ($DEFS)"
EOS
docker run --rm -e OMP="$OMP" -v "$PWD":/workdir sdk_pynqz2 bash /workdir/.build_inner.sh
rm -f .build_inner.sh ga.o gs.o
file balle_acetone
