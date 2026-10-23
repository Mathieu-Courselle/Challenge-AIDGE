"""Exporte l'encodeur (ga.onnx) et le decodeur (gs.onnx) de bmshj2018_factorized en ONNX.
La GDN de CompressAI est reecrite avec des operateurs standard (Conv, Mul, Sqrt, Div).
Usage : python export_balle.py [qualite=3]
Prerequis : pip install "numpy<2" onnx onnxscript onnxruntime"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
import onnx, onnxruntime as ort
from compressai.zoo import bmshj2018_factorized
from compressai.layers import GDN

Q = int(sys.argv[1]) if len(sys.argv) > 1 else 3
net = bmshj2018_factorized(quality=Q, pretrained=True).eval()

class XGDN(nn.Module):
    """GDN/IGDN reecrit avec des operateurs ONNX standard."""
    def __init__(self, g):
        super().__init__()
        C = g.beta.shape[0]
        with torch.no_grad():
            self.register_buffer("gamma", g.gamma_reparam(g.gamma).reshape(C, C, 1, 1).clone())
            self.register_buffer("beta", g.beta_reparam(g.beta).clone())
        self.inverse = bool(g.inverse)
    def forward(self, x):
        n = torch.sqrt(F.conv2d(x * x, self.gamma, self.beta))
        return x * n if self.inverse else x / n

def convert(seq):
    return nn.Sequential(*[XGDN(m) if isinstance(m, GDN) else m for m in seq]).eval()

ga, gs = convert(net.g_a), convert(net.g_s)
x = torch.rand(1, 3, 64, 64)
with torch.no_grad():
    y = ga(x)
    print("encodeur : entree", tuple(x.shape), "-> sortie", tuple(y.shape))
    print("ecart avec le modele d'origine : encodeur", float((ga(x) - net.g_a(x)).abs().max()),
          "| decodeur", float((gs(y) - net.g_s(y)).abs().max()))

def export(model, dummy, path, i, o):
    try:
        torch.onnx.export(model, dummy, path, input_names=[i], output_names=[o], opset_version=13, dynamo=False)
    except Exception as e:
        print("exporteur classique indisponible :", repr(e)[:200], "-> essai avec l'exporteur dynamo")
        torch.onnx.export(model, (dummy,), path, input_names=[i], output_names=[o], dynamo=True)

def check(model, dummy, path):
    with torch.no_grad(): ref = model(dummy).numpy()
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = s.run(None, {s.get_inputs()[0].name: dummy.numpy()})[0]
    ops = sorted({n.op_type for n in onnx.load(path).graph.node})
    print(f"{path} : sortie {out.shape} | ecart max onnx/pytorch {abs(ref - out).max():.2e} | operateurs : {ops}")

export(ga, x, "ga.onnx", "x", "y"); check(ga, x, "ga.onnx")
export(gs, y, "gs.onnx", "y", "x_hat"); check(gs, y, "gs.onnx")
