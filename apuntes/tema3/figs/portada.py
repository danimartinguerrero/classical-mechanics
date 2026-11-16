"""Ilustración de portada (Tema 3, Mecánica I: problema de dos cuerpos, colisiones).

Un haz paralelo de partículas incide desde la izquierda sobre un blanco con
interacción repulsiva central (potencial k/r). Cada trayectoria se colorea según
su parámetro de impacto b: las de b pequeño se dispersan hacia atrás, las de b
grande apenas se desvían. Salida: PNG con fondo transparente.

Uso:  python3 portada.py  [salida.png]
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from PIL import Image

out = sys.argv[1] if len(sys.argv) > 1 else "portada.png"

k = 0.30         # intensidad de la repulsión (k/r)
v0 = 1.0
x0 = -9.0
cmap = matplotlib.colormaps["turbo"]

def rhs(t, s):
    x, y, vx, vy = s
    r3 = (x * x + y * y) ** 1.5
    return [vx, vy, k * x / r3, k * y / r3]

def escapa(t, s):
    return np.hypot(s[0], s[1]) - 13.5
escapa.terminal = True
escapa.direction = 1

fig = plt.figure(figsize=(12, 5.0), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

bs = np.concatenate([-np.linspace(0.06, 4.2, 26)[::-1], np.linspace(0.06, 4.2, 26)])
bmax = 4.2
for b in bs:
    sol = solve_ivp(rhs, (0, 60), [x0, b, v0, 0.0], events=escapa,
                    max_step=0.02, rtol=1e-9, atol=1e-11)
    x, y = sol.y[0], sol.y[1]
    pts = np.stack([x, y], 1)
    segs = np.stack([pts[:-1], pts[1:]], 1)
    c = np.array(cmap(0.08 + 0.84 * (abs(b) - 0.06) / (bmax - 0.06)))
    # desvanecimiento a lo largo de la trayectoria saliente
    s = np.linspace(0, 1, len(segs))
    cols = np.tile(c, (len(segs), 1))
    cols[:, 3] = 0.85 * (1 - 0.55 * s ** 2)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.95))

# blanco: núcleo con halo
for r, a in [(0.75, 0.05), (0.5, 0.08), (0.32, 0.14), (0.2, 0.3)]:
    ax.add_patch(plt.Circle((0, 0), r, color="#e8e8e8", alpha=a, lw=0))
ax.add_patch(plt.Circle((0, 0), 0.12, color="white", lw=0))

ax.set_xlim(-9.2, 11.0)
ax.set_ylim(-4.6, 4.6)
ax.set_aspect("equal")

fig.savefig(out, transparent=True, dpi=300)
im = Image.open(out)
x0_, y0_, x1_, y1_ = im.getbbox()
mg = 30
im.crop((max(x0_ - mg, 0), max(y0_ - mg, 0), min(x1_ + mg, im.width), min(y1_ + mg, im.height))).save(out)
print("ok", out, im.size)
