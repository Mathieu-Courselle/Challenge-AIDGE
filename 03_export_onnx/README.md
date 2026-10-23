# Export ONNX des modèles CompressAI

Un dossier par modèle. Chaque dossier contient `export_onnx.py` et les fichiers `.onnx` produits,
nommés `<modèle>_q<qualité>_<sous réseau>.onnx` (exemple : `mbt2018-mean_q3_h_s.onnx`).

| Dossier | Sous réseaux exportés | Opérateurs ONNX |
|---|---|---|
| `bmshj2018-factorized/` | g_a, g_s | Conv, ConvTranspose, Mul, Div, Sqrt |
| `bmshj2018-hyperprior/` | g_a, g_s, h_a, h_s | + Abs, Relu |
| `mbt2018-mean/` | g_a, g_s, h_a, h_s | + LeakyRelu |
| `mbt2018/` | g_a, g_s, h_a, h_s, context_prediction, entropy_parameters | + LeakyRelu |
| `cheng2020-/` | idem, variante `cheng2020-anchor` | + Add, DepthToSpace |

`cheng2020` n'est pas un modèle de Ballé : on le garde comme point de comparaison.
Le dossier `bmshj2018-factorized/` contient aussi `export_balle.py`, le premier script d'export (g_a et g_s seulement).

## Coût de calcul (qualité 3, image 64x64, millions de multiplications des convolutions)

| Modèle | g_a | g_s | h_a | h_s | contexte | entropie |
|---|---|---|---|---|---|---|
| bmshj2018-factorized | 150,7 | 150,7 | | | | |
| bmshj2018-hyperprior | 150,7 | 150,7 | 5,6 | 5,6 | | |
| mbt2018-mean | 150,7 | 150,7 | 5,6 | 22,1 | | |
| mbt2018 | 324,4 | 324,4 | 9,9 | 22,4 | 29,5 | 16,3 |
| cheng2020-anchor | 653,3 | 1009,8 | 6,0 | 14,0 | 13,1 | 7,2 |

Les écarts entre chaque ONNX et PyTorch sont tous inférieurs à 3e-5.
Le codage entropique (tables CDF, codeur arithmétique) n'est pas dans les ONNX.
Dans `mbt2018` et `cheng2020`, le contexte est autorégressif : un ONNX ne représente qu'une passe sur le latent complet.

## Utilisation

```bash
source ~/venv-cai/bin/activate
pip install "numpy<2" onnx onnxscript onnxruntime

# un modèle (qualité 3, image 64x64)
cd mbt2018-mean && python export_onnx.py 3 64

# les cinq à la suite
bash tout_exporter.sh 3 64
```

Au premier lancement, les poids pré-entraînés sont téléchargés (il faut internet).
Si le téléchargement échoue (erreur DNS sous WSL), relancer.
