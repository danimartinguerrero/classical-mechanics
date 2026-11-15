"""Representación esquemática de un cúmulo de galaxias (tipo Coma).

Galaxias elípticas distribuidas con un perfil de King proyectado, dos
elípticas gigantes en el centro y un halo difuso intracúmulo. El tono de
cada galaxia (azulado/rojizo) codifica su velocidad en la línea de visión,
que es lo que se mide para estimar la dispersión de velocidades.
No es una imagen real. Salida: PNG.

Uso:  python3 cumulo.py [salida.png]
"""
import sys
import numpy as np
from PIL import Image

out = sys.argv[1] if len(sys.argv) > 1 else "cumulo.png"
rng = np.random.default_rng(7)
N = 1400                      # píxeles
L = 2.4                       # semiancho del campo (unidades de r_c)
x = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, x)
img = np.zeros((N, N, 3))

def blob(cx, cy, a, q, ang, amp, col):
    c, s = np.cos(ang), np.sin(ang)
    dx, dy = X - cx, Y - cy
    u = (c * dx + s * dy) / a
    w = (-s * dx + c * dy) / (a * q)
    r2 = u * u + w * w
    prof = amp * (np.exp(-r2 / 2) + 0.35 * np.exp(-np.sqrt(r2) * 1.2))
    for k in range(3):
        img[..., k] += prof * col[k]

# halo difuso intracúmulo
R = np.hypot(X, Y)
halo = 0.07 / (1 + (R / 0.9) ** 2) ** 1.1
img += halo[..., None] * np.array([1.0, 0.78, 0.55])

# galaxias: radios con perfil de King proyectado, Sigma ~ (1 + r^2)^-1
n_gal = 480
u = rng.random(n_gal)
rr = np.sqrt((1 / (1 - u * 0.92)) - 1) * 0.75
th = rng.random(n_gal) * 2 * np.pi
gx, gy = rr * np.cos(th), rr * np.sin(th)
vlos = rng.normal(0, 1, n_gal)
warm = np.array([1.0, 0.86, 0.62])
for cx, cy, v in zip(gx, gy, vlos):
    if abs(cx) > L or abs(cy) > L:
        continue
    a = 0.0075 * np.exp(rng.normal(0, 0.5))
    q = 0.55 + 0.45 * rng.random()
    t = np.clip(v / 2.0, -1, 1)
    col = warm * (1 - 0.55 * abs(t)) + 0.55 * abs(t) * (np.array([0.55, 0.75, 1.0]) if t < 0 else np.array([1.0, 0.55, 0.55]))
    blob(cx, cy, a, q, rng.random() * np.pi, 0.7 + 0.9 * rng.random(), col)

# dos elípticas gigantes centrales
blob(-0.18, 0.06, 0.032, 0.82, 0.4, 2.0, warm)
blob(0.26, -0.1, 0.028, 0.75, -0.6, 1.8, warm)

# galaxias de fondo, muy débiles
for _ in range(160):
    cx, cy = (rng.random(2) - 0.5) * 2 * L
    blob(cx, cy, 0.004, 0.6 + 0.4 * rng.random(), rng.random() * np.pi, 0.25, np.array([0.7, 0.75, 0.9]))

# estrellas de campo (muy pocas, con espícula sutil)
for _ in range(14):
    cx, cy = (rng.random(2) - 0.5) * 2 * L * 0.95
    blob(cx, cy, 0.006, 1.0, 0, 2.5, np.array([0.85, 0.9, 1.0]))

# tono: compresión logarítmica
img = np.log1p(6 * img) / np.log1p(6 * 2.2)
img = np.clip(img, 0, 1) ** 0.9
Image.fromarray((img * 255).astype(np.uint8)).save(out)
print("ok", out)
