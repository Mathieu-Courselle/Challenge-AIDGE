# Commandes utiles

## Carte PYNQ-Z2
- IP : `192.168.0.10`, utilisateur `root`. Console série : 115200 8N1.
- Depuis Windows (PowerShell) ou depuis WSL via `ssh.exe` / `scp.exe` (WSL2 n'atteint pas toujours la carte directement) :
  ```
  scp.exe fichier root@192.168.0.10:/home/root/
  ssh.exe root@192.168.0.10 "chmod +x fichier && ./fichier"
  ```
  `scp.exe` ne comprend pas les chemins Linux : copier d'abord le fichier dans `/mnt/c/Users/<nom>/`.
- La carte utilise BusyBox : pas de `tree`, `head -c` ; utiliser `ls -l`, `od -c`, `strings`.
- L'horloge de la carte n'est pas réglée (dates fausses).

## Compiler pour la carte avec l'image SDK
```
docker run --rm -v ~/optconv:/workdir sdk_pynqz2 bash -c \
 '. /opt/petalinux-sdk/environment-setup-* && $CC -O2 conv_bench.c -o conv_O2 -lm'
```

## LoCoD (CPU seul)
```
sudo service docker start
cd ~/LoCod
./locod.sh -t pynqz2 -f demo/conv_test/main.c --no-hard
# résultat : locod-output/locod-cpu
```
Sans Vivado ni Panda-Bambu installés, `locod.sh` s'arrête à la vérification des dépendances :
on a utilisé des faux exécutables (stubs) `vivado` et `bambu` pour le mode `--no-hard`.
Ne pas lancer sur la carte un exécutable qui appelle `FPGA()` sans bitstream chargé.

## Attention disque (WSL2)
Le disque virtuel de WSL est sur C: et ne rétrécit pas tout seul. Garder au moins 10 Go libres sur C:
(un disque plein a déjà fait planter Docker et WSL).
