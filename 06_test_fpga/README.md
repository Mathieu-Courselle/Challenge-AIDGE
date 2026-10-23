# Test du FPGA sur la PYNQ-Z2

Premier test réussi du FPGA sur la carte, avec un exemple LoCoD (`exemple_pipeline`) fourni par un camarade :
un accélérateur `matvec` (produit matrice vecteur) suivi d'un accélérateur `relu_quant` (ReLU et quantification).
Les fichiers de cet exemple (`fpga.bit`, `fpga.bin`, `locod-cpu`) ne sont pas dans le dépôt.

Résultat : `Verification FPGA == CPU : OK` (voir `sortie_locod_cpu.txt`).
Temps mesurés par le FPGA : matvec 217,2 µs, relu_quant 16,0 µs.

## Fichiers

| Fichier | Rôle |
|---|---|
| `bit2bin.py` | Convertit un `.bit` Vivado en `.bin` « byte swapped » accepté par `fpgautil` |
| `rapport_fpga.py` | Lit la sortie de `locod-cpu` et affiche un rapport simple |
| `sortie_locod_cpu.txt` | Sortie complète du test |

## Procédure

```bash
# 1. sur le PC : convertir le bitstream
python3 bit2bin.py fpga.bit fpga_swapped.bin

# 2. envoyer sur la carte
scp fpga_swapped.bin locod-cpu root@192.168.0.10:/root/

# 3. sur la carte : charger le FPGA puis lancer le programme
ssh root@192.168.0.10 "fpgautil -b /root/fpga_swapped.bin; cat /sys/class/fpga_manager/fpga0/state"
ssh root@192.168.0.10 "chmod +x /root/locod-cpu; /root/locod-cpu"

# 4. rapport lisible (depuis le PC)
ssh root@192.168.0.10 "/root/locod-cpu" | python3 rapport_fpga.py
```

Après le chargement, l'état de `/sys/class/fpga_manager/fpga0/state` doit rester `operating`.

## Problème rencontré : « could not find a sync word »

Le `fpga.bin` d'origine était refusé par `fpgautil` :

```
fpga_manager fpga0: Invalid bitstream, could not find a sync word. Bitstream must be a byte swapped .bin file
```

Le pilote Linux `fpga_manager` du Zynq-7000 exige un `.bin` dont les octets sont inversés par mots de 32 bits
(le mot de synchronisation `AA 99 55 66` doit apparaître sous la forme `66 55 99 AA`).
`bit2bin.py` retire l'en-tête du `.bit` et fait cette inversion.
La règle vient du pilote Linux, pas de la puce : un autre mécanisme de chargement peut se comporter autrement.
Pour la suite, demander un `.bin` déjà « byte swapped » (`bootgen -process_bitstream bin`, non testé ici) ou le `.bit`.

## Ce que ce test prouve et ne prouve pas

- Prouvé : la chaîne conversion, `fpgautil`, `locod-cpu` fonctionne sur notre carte, et le FPGA calcule le même résultat que le CPU.
- Pas prouvé : que notre modèle de Ballé tourne dans le FPGA. Il faut pour cela Vivado 2022.2 et Panda-Bambu pour générer notre propre bitstream.
