"""Ilustración de portada (Tema 2, Mecánica I: sistemas de partículas).

Un sistema rígido de N partículas (un anillo con radios) es lanzado y gira
mientras su centro de masas describe una parábola (blanco). Cada partícula
deja su estela, coloreada por partícula; se dibujan además algunas
instantáneas del sistema. Salida: PNG con fondo transparente.

Uso:  python3 portada.py  [salida.png]
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

out = sys.argv[1] if len(sys.argv) > 1 else "portada.png"

N = 9                       # partículas del anillo
a = 1.0                     # radio del anillo
t = np.linspace(0.0, 1.0, 1400)
# centro de masas: parábola en el plano xz
Xc = 10.0 * t
Zc = 12.0 * t * (1.0 - t)
Yc = 0.0 * t
# rotación: ángulo de giro alrededor de un eje inclinado
w = 2 * np.pi * 2.25
phi = w * t
axis = np.array([0.35, 1.0, 0.25]); axis /= np.linalg.norm(axis)
e1 = np.cross(axis, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
e2 = np.cross(axis, e1)

def posiciones(k):
    th = 2 * np.pi * k / N + phi
    p = a * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    return np.stack([Xc, Yc, Zc], -1) + p

fig = plt.figure(figsize=(12, 5.0), dpi=300)
ax = fig.add_axes([-0.08, -0.35, 1.16, 1.7], projection="3d")
ax.set_axis_off()
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
cmap = matplotlib.colormaps["turbo"]

# estelas de las partículas
for k in range(N):
    P = posiciones(k)
    segs = np.stack([P[:-1], P[1:]], axis=1)
    col = np.array(cmap(0.06 + 0.88 * k / (N - 1)))
    cols = np.tile(col, (len(segs), 1))
    cols[:, 3] = 0.75
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=0.9))

# instantáneas del sistema (anillo + radios)
for tk in np.linspace(0.05, 0.95, 7):
    i = int(tk * (len(t) - 1))
    pts = np.array([posiciones(k)[i] for k in range(N)])
    c = np.array([Xc[i], Yc[i], Zc[i]])
    ring = np.vstack([pts, pts[:1]])
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="white", lw=0.9, alpha=0.55)
    for p in pts:
        ax.plot([c[0], p[0]], [c[1], p[1]], [c[2], p[2]], color="white", lw=0.5, alpha=0.35)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=[cmap(0.06 + 0.88 * k / (N - 1)) for k in range(N)],
               s=16, depthshade=False)

# trayectoria del CM
ax.plot(Xc, Yc, Zc, color="#f2f2f2", lw=2.4, solid_capstyle="round")
for tk in np.linspace(0.05, 0.95, 7):
    i = int(tk * (len(t) - 1))
    ax.scatter([Xc[i]], [Yc[i]], [Zc[i]], color="white", s=14, depthshade=False)

ax.set_xlim(-1.2, 11.2); ax.set_ylim(-3.0, 3.0); ax.set_zlim(-1.4, 4.4)
ax.set_box_aspect((12.4, 6.0, 5.8), zoom=1.0)
ax.view_init(elev=14, azim=-72)
ax.set_proj_type("persp", focal_length=0.5)

fig.savefig(out, transparent=True, dpi=300, bbox_inches=None)
im = Image.open(out)
x0, y0, x1, y1 = im.getbbox()
mg = 30
im.crop((max(x0 - mg, 0), max(y0 - mg, 0), min(x1 + mg, im.width), min(y1 + mg, im.height))).save(out)
print("ok", out)
