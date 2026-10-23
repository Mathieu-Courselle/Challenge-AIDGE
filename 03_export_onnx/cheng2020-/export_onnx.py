"""Export ONNX du modele cheng2020-anchor de CompressAI (pour Aidge).

Usage :  python export_onnx.py [qualite] [taille_image]
Exemple : python export_onnx.py 3 64

Produit un fichier .onnx par sous reseau, nomme <modele>_q<qualite>_<sous reseau>.onnx
(exemple : bmshj2018-hyperprior_q3_g_a.onnx), : g_a (encodeur), g_s (decodeur) et,
si le modele en a, h_a, h_s, context_prediction, entropy_parameters.
Les GDN/IGDN sont reecrits avec des operateurs ONNX standard (Conv, Mul, Sqrt, Div).
Le codage entropique (CDF, codeur arithmetique) n'est pas dans l'ONNX.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from compressai.zoo import image_models
from compressai.layers import GDN

MODELE = "cheng2020-anchor"
Q = int(sys.argv[1]) if len(sys.argv) > 1 else 3
S = int(sys.argv[2]) if len(sys.argv) > 2 else 64

net = image_models[MODELE](quality=Q, pretrained=True).eval()
print("modele :", MODELE, "| qualite", Q, "| classe", type(net).__name__, "| image", S, "x", S)


class XGDN(nn.Module):
    """GDN / IGDN avec des operateurs ONNX standard."""

    def __init__(self, g):
        super().__init__()
        C = g.beta.shape[0]
        with torch.no_grad():
            self.register_buffer("gamma", g.gamma_reparam(g.gamma).reshape(C, C, 1, 1).clone())
            self.register_buffer("beta", g.beta_reparam(g.beta).clone())
        self.inverse = bool(g.inverse)

    def forward(self, x):
        n = torch.sqrt(F.conv2d(x * x, self.gamma, self.beta))
        if self.inverse:
            return x * n
        return x / n


def remplace_gdn(module):
    """Remplace tous les GDN du module (a n'importe quelle profondeur)."""
    for nom, enfant in module.named_children():
        if isinstance(enfant, GDN):
            setattr(module, nom, XGDN(enfant))
        else:
            remplace_gdn(enfant)
    return module


class ValeurAbsolue(nn.Module):
    """h_a du ScaleHyperprior travaille sur abs(y)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, y):
        return self.m(torch.abs(y))


def conv_sans_masque(m):
    """Le MaskedConv2d devient un Conv2d normal avec les poids deja masques."""
    c = nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride,
                  padding=m.padding, bias=m.bias is not None)
    with torch.no_grad():
        c.weight.copy_(m.weight * m.mask)
        if m.bias is not None:
            c.bias.copy_(m.bias)
    return c.eval()


# on rejoue la chaine avec le vrai modele pour obtenir les bonnes tailles d'entree
x = torch.rand(1, 3, S, S)
parties = {}  # nom -> (module, entree)
with torch.no_grad():
    y = net.g_a(x)
    parties["g_a"] = (net.g_a, x)
    parties["g_s"] = (net.g_s, y)
    print("latent y :", tuple(y.shape))
    if hasattr(net, "h_a"):
        scale_seul = type(net).__name__ == "ScaleHyperprior"
        h_a = ValeurAbsolue(net.h_a) if scale_seul else net.h_a
        z = h_a(y)
        h_s_out = net.h_s(z)
        parties["h_a"] = (h_a, y)
        parties["h_s"] = (net.h_s, z)
        print("hyper latent z :", tuple(z.shape), "| sortie h_s :", tuple(h_s_out.shape))
        if hasattr(net, "context_prediction"):
            ctx = conv_sans_masque(net.context_prediction)
            parties["context_prediction"] = (ctx, y)
            ep_in = torch.cat((h_s_out, ctx(y)), dim=1)
            parties["entropy_parameters"] = (net.entropy_parameters, ep_in)
            print("entropy_parameters : entree", tuple(ep_in.shape))

# remplacement des GDN apres avoir garde les modules d'origine pour la comparaison
originaux = {}
for nom, (m, e) in parties.items():
    originaux[nom] = m
import copy
parties = {nom: (remplace_gdn(copy.deepcopy(m)).eval(), e) for nom, (m, e) in parties.items()}


def exporte(model, dummy, chemin):
    try:
        torch.onnx.export(model, dummy, chemin, input_names=["entree"], output_names=["sortie"],
                          opset_version=13, dynamo=False)
    except Exception as e:
        print("  exporteur classique indisponible :", repr(e)[:150], "| essai dynamo")
        torch.onnx.export(model, (dummy,), chemin, input_names=["entree"], output_names=["sortie"], dynamo=True)


def mmac(model, dummy):
    """Nombre de multiplications accumulations des convolutions (en millions)."""
    total = [0]

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            total[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.ConvTranspose2d):
            total[0] += i[0].numel() * (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    with torch.no_grad():
        model(dummy)
    for h in hs:
        h.remove()
    return total[0] / 1e6


print("")
for nom, (m, e) in parties.items():
    chemin = MODELE + "_q" + str(Q) + "_" + nom + ".onnx"
    exporte(m, e, chemin)
    with torch.no_grad():
        ref = m(e).numpy()
        ref_orig = originaux[nom](e).numpy()
    sess = ort.InferenceSession(chemin, providers=["CPUExecutionProvider"])
    sortie = sess.run(None, {sess.get_inputs()[0].name: e.numpy()})[0]
    ops = sorted({n.op_type for n in onnx.load(chemin).graph.node})
    nb_params = sum(p.numel() for p in m.parameters()) + sum(b.numel() for b in m.buffers())
    print(nom + " :", tuple(e.shape), "->", sortie.shape)
    print("   ecart onnx / pytorch :", float(abs(ref - sortie).max()))
    print("   ecart reecriture GDN / modele d'origine :", float(abs(ref - ref_orig).max()))
    print("   parametres :", nb_params, "| convolutions :", round(mmac(m, e), 1), "MMAC")
    print("   operateurs :", ops)
print("")
print("termine : fichiers .onnx crees dans le dossier courant")
