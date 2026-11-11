"""Ilustración de portada (Tema 1, Mecánica I).

Hélice circular con su «cinta de Frenet»: en cada punto se traza el
segmento r(s) -> r(s) + w n(s) a lo largo de la normal principal,
coloreado según la longitud de arco. La envolvente de esos segmentos es
un helicoide; la curva se dibuja en blanco. Salida: PNG con fondo
transparente (se compone sobre la portada negra).

Uso:  python3 portada.py  [salida.png]
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

out = sys.argv[1] if len(sys.argv) > 1 else "portada.png"

R, c = 1.0, 0.24          # radio y paso reducido (b = 2*pi*c)
turns = 4.5
t = np.linspace(0.0, 2 * np.pi * turns, 4000)

def helix(t):
    return np.stack([R * np.cos(t), R * np.sin(t), c * t], axis=-1)

# triedro de Frenet de la hélice (forma cerrada)
L = np.hypot(R, c)
tau = np.stack([-R * np.sin(t), R * np.cos(t), c * np.ones_like(t)], -1) / L
nor = np.stack([-np.cos(t), -np.sin(t), 0 * t], -1)
binor = np.cross(tau, nor)

r = helix(t)

# eje de la hélice en horizontal: (x, y, z) -> (z, x, y)
P = lambda a: a[..., [2, 0, 1]]
r, tau, nor, binor = P(r), P(tau), P(nor), P(binor)

fig = plt.figure(figsize=(12, 5.0), dpi=300)
ax = fig.add_axes([-0.08, -0.35, 1.16, 1.7], projection="3d")
ax.set_axis_off()
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

cmap = matplotlib.colormaps["turbo"]

# --- cinta de normales (helicoide) ------------------------------------
k = np.linspace(0, len(t) - 1, 420).astype(int)
w = 0.92
segs = np.stack([r[k], r[k] + w * nor[k]], axis=1)
u = (t[k] - t[0]) / (t[-1] - t[0])
cols = cmap(0.08 + 0.84 * u)
cols[:, 3] = 0.80
ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=0.55))

# --- binormales cortas (destellos verticales) --------------------------
kb = k[::6]
segb = np.stack([r[kb], r[kb] + 0.42 * binor[kb]], axis=1)
ub = (t[kb] - t[0]) / (t[-1] - t[0])
cb = cmap(0.08 + 0.84 * ub)
cb[:, 3] = 0.95
ax.add_collection3d(Line3DCollection(segb, colors=cb, linewidths=0.9))

# --- eje de la hélice (tenue) ------------------------------------------
ax.plot([-0.3, c * t[-1] + 0.3], [0, 0], [0, 0], color="white", lw=0.6,
        alpha=0.25, ls=(0, (4, 4)))

# --- la curva ------------------------------------------------------------
ax.plot(r[:, 0], r[:, 1], r[:, 2], color="#f2f2f2", lw=2.6,
        solid_capstyle="round")

X = c * t[-1]
ax.set_xlim(-0.1, X + 0.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-1.1, 1.1)
ax.set_box_aspect((X + 0.2, 2.2, 2.2), zoom=1.0)
ax.view_init(elev=20, azim=-55)
ax.set_proj_type("persp", focal_length=0.45)

fig.savefig(out, transparent=True, dpi=300, bbox_inches=None)

# recorte al contenido (con un pequeño margen)
from PIL import Image
im = Image.open(out)
x0, y0, x1, y1 = im.getbbox()
m = 30
im.crop((max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, im.width),
         min(y1 + m, im.height))).save(out)
print("ok", out)
