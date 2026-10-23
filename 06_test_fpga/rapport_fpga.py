import re
import sys

if len(sys.argv) > 1:
    txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
else:
    txt = sys.stdin.read()


def cherche(motif):
    m = re.search(motif, txt)
    if m:
        return m.group(1)
    return None


ctrl = cherche(r"control registers at address (0x[0-9a-fA-F]+)")
mv = cherche(r"matvec\s*:\s*(\d+)\s*ns")
rq = cherche(r"relu_quant\s*:\s*(\d+)\s*ns")
z = cherche(r"sortie z\s*:\s*([\-\d ]+)")
verif = cherche(r"Verification FPGA == CPU\s*:\s*(\w+)")

print("")
print("test du fpga sur la carte pynq z2")
print("")

print("1. ouverture du fpga (l'arm se branche sur le fpga)")
print("adresse utilisee : " + str(ctrl))
print("")

print("2. envoi des donnees (l'arm depose les donnees dans la memoire que le fpga peut lire)")
print("")

print("3. calculs (le fpga fait deux calculs a la suite, matvec puis relu_quant)")
print("matvec : multiplie une matrice par un vecteur")
print("relu_quant : met les valeurs negatives a 0 et arrondit les autres")
print("")

print("4. temps (combien de temps a pris chaque calcul dans le fpga)")
if mv is not None and rq is not None:
    a = int(mv)
    b = int(rq)
    print("matvec : " + str(a) + " ns")
    print("relu_quant : " + str(b) + " ns")
    print("total : " + str(a + b) + " ns")
else:
    print("temps pas trouves")
print("")

print("5. resultats (les 16 nombres que le fpga a calcules, 0 veut dire eteint)")
if z is not None:
    vals = [int(v) for v in z.split()]
    i = 0
    for v in vals:
        print("z" + str(i) + " : " + str(v))
        i = i + 1
    print("le plus grand : " + str(max(vals)))
    print("nombre de zeros : " + str(vals.count(0)) + " sur " + str(len(vals)))
else:
    print("resultats pas trouves")
print("")

print("6. verification (on compare le resultat du fpga avec celui du cpu)")
if verif == "OK":
    print("c'est pareil, le fpga marche et calcule juste")
else:
    print("ca ne colle pas, a verifier")
print("")
