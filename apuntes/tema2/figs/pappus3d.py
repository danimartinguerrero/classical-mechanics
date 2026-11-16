"""Figuras 3D de Pappus--Guldin (Tema 2): sólido de revolución de un semicírculo y toro.

Genera figs/pappus_a.png, figs/pappus_b.png (fondo transparente) y
figs/pappus_coords.tex con las posiciones (en cm, para el ancho dado) de los
puntos donde la figura TikZ coloca las etiquetas.

Uso:  python3 pappus3d.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

AZUL = np.array([0x5A, 0xA9, 0xE6]) / 255
HIELO = np.array([0xC9, 0xDC, 0xF0]) / 255
ORO = np.array([0xD6, 0xB2, 0x5E]) / 255
TENUE = np.array([0x9C, 0xA0, 0xA8]) / 255
ANCHO_CM = 6.4
ANCHO = {"a": 5.2, "b": 6.6}
ALFA = 0.42   # transparencia de las superficies (como los rellenos del documento)
ls = LightSource(azdeg=315, altdeg=40)


def superficie(ax, X, Y, Z, base=AZUL, alpha=1.0):
    rgb = np.ones(Z.shape + (3,)) * base
    # sombreado según la normal
    n = _normales(X, Y, Z)
    luz = np.array([-0.45, -0.55, 0.70]); luz /= np.linalg.norm(luz)
    shade = np.abs(n @ luz)
    col = rgb * (0.35 + 0.65 * shade[..., None])
    col = np.clip(col, 0, 1)
    ax.plot_surface(X, Y, Z, facecolors=np.concatenate([col, np.full(Z.shape + (1,), alpha)], -1),
                    linewidth=0, antialiased=False, shade=False, rstride=1, cstride=1)


def _normales(X, Y, Z):
    du = np.stack([np.gradient(X, axis=0), np.gradient(Y, axis=0), np.gradient(Z, axis=0)], -1)
    dv = np.stack([np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)], -1)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n


def cuadros(X, Y, Z, base, alfa=1.0):
    """Polígonos (cuadriláteros) de una malla con su color sombreado."""
    n = _normales(X, Y, Z)
    luz = np.array([-0.45, -0.55, 0.70]); luz /= np.linalg.norm(luz)
    sh = 0.30 + 0.85 * np.abs(n @ luz)
    polys, cols = [], []
    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1] - 1):
            q = [(X[i, j], Y[i, j], Z[i, j]), (X[i + 1, j], Y[i + 1, j], Z[i + 1, j]),
                 (X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]), (X[i, j + 1], Y[i, j + 1], Z[i, j + 1])]
            polys.append(q); cols.append((*np.clip(base * sh[i, j], 0, 1), alfa))
    return polys, cols


def abanico(P, color, alfa=1.0):
    c = P.mean(0)
    return [[tuple(c), tuple(P[k]), tuple(P[(k + 1) % len(P)])] for k in range(len(P))], [(*color, alfa)] * len(P)


def preparar(lims, elev, azim, aspect):
    fig = plt.figure(figsize=(6.4, 6.4 * aspect[2] / aspect[0] * 1.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_axis_off(); fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_box_aspect(aspect, zoom=1.35)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    return fig, ax


def a_cm(fig, ax, P, recorte, tam, ancho):
    """Coordenadas (cm) de un punto 3D en la imagen recortada."""
    x2, y2, _ = proj3d.proj_transform(*P, ax.get_proj())
    px, py = ax.transData.transform((x2, y2))
    W, H = tam
    py = fig.bbox.height - py  # origen arriba
    x0, y0, x1, y1 = recorte
    escala = ancho / (x1 - x0)
    return ((px - x0) * escala, ((y1 - y0) - (py - y0)) * escala)


def guardar(fig, nombre):
    fig.savefig(nombre, transparent=True, dpi=300)
    im = Image.open(nombre)
    bb = im.getbbox()
    mg = 12
    rec = (max(bb[0] - mg, 0), max(bb[1] - mg, 0), min(bb[2] + mg, im.width), min(bb[3] + mg, im.height))
    im.crop(rec).save(nombre)
    return rec, (im.width, im.height)


coords = []

# ------------------------------------------------------------------ (a)
# región plana arbitraria (suave) en el semiplano meridiano (x, rho), sin cortar el eje
tt = np.linspace(0, 2 * np.pi, 361)
Rb = 0.62 * (1 + 0.20 * np.cos(2 * tt - 0.6) + 0.10 * np.sin(3 * tt))
bx = 0.95 * Rb * np.cos(tt) + 0.05
brho = 1.55 + 0.80 * Rb * np.sin(tt)
# centroide (fórmula del polígono)
xs_, ys_ = bx[:-1], brho[:-1]; xn, yn = np.roll(xs_, -1), np.roll(ys_, -1)
cr = xs_ * yn - xn * ys_; Aar = cr.sum() / 2
gx = ((xs_ + xn) * cr).sum() / (6 * Aar); ybar = ((ys_ + yn) * cr).sum() / (6 * Aar)
f0, f1 = np.radians(450), np.radians(212)       # sólido entre 212 y 450 grados: corte delantero
fig, ax = preparar(((-1.9, 1.9), (-2.6, 2.6), (-2.6, 2.6)), elev=18, azim=-66, aspect=(3.8, 5.2, 5.2))
phi = np.linspace(f1, f0, 150)
idx = np.arange(0, len(tt), 3)
T, F = np.meshgrid(idx, phi, indexing="ij")
X = bx[T]; RHO = brho[T]
polys, cols = cuadros(X, RHO * np.cos(F), RHO * np.sin(F), AZUL, ALFA)
for fc, colr, al in [(f1, AZUL * 0.8, 0.30), (f0, AZUL, 0.45)]:
    Pf = np.stack([bx[:-1], brho[:-1] * np.cos(fc), brho[:-1] * np.sin(fc)], -1)
    pp, cc = abanico(Pf, colr, al); polys += pp; cols += cc
# eje como tubo fino (participa en la ocultación)
xa = np.linspace(-2.0, 2.15, 60); pa = np.linspace(0, 2 * np.pi, 8)
XA, PA = np.meshgrid(xa, pa, indexing="ij")
pp, cc = cuadros(XA, 0.008 * np.cos(PA), 0.008 * np.sin(PA), TENUE * 0.9)
polys += pp; cols += cc
col3 = Poly3DCollection(polys, facecolors=cols, edgecolors="none", linewidths=0)
ax.add_collection3d(col3)
for fc in (f1, f0):
    ax.plot(bx, brho * np.cos(fc), brho * np.sin(fc), color=AZUL, lw=1.1)
# circunferencia del centroide (discontinua, por encima)
ps = np.linspace(0, 2 * np.pi, 400)
ax.plot(gx + 0 * ps, ybar * np.cos(ps), ybar * np.sin(ps), color=ORO, lw=1.0, ls=(0, (4, 3)), zorder=6)
G = (gx, 0.0, ybar)
ax.plot([gx, gx], [0, 0], [0, ybar], color=ORO, lw=0.8, zorder=6)
ax.scatter(*G, color=ORO, s=14, depthshade=False, zorder=7)
rec, tam = guardar(fig, "pappus_a.png")
P = {"G": G, "A": (gx + 0.36, 0, ybar + 0.06), "eje": (2.15, 0, 0), "ybar": (gx, 0, ybar / 2),
     "ruta": (gx, ybar * np.cos(np.radians(128)), ybar * np.sin(np.radians(128)))}
for k, v in P.items():
    x, y = a_cm(fig, ax, v, rec, tam, ANCHO["a"])
    coords.append(r"\def\papA%s{(%.3f,%.3f)}" % (k, x, y))
im = Image.open("pappus_a.png"); coords.append(r"\def\papAh{%.3f}" % (ANCHO["a"] * im.height / im.width))
plt.close(fig)

# ------------------------------------------------------------------ (b)
R, r = 1.75, 0.55
fig, ax = preparar(((-2.4, 2.4), (-2.4, 2.4), (-1.0, 1.0)), elev=30, azim=-60, aspect=(4.8, 4.8, 2.0))
ax.computed_zorder = False
g0, g1 = np.radians(-30), np.radians(-30 + 300)
phi = np.linspace(g0, g1, 300)
th = np.linspace(0, 2 * np.pi, 120)
TH, PH = np.meshgrid(th, phi, indexing="ij")
X = (R + r * np.cos(TH)) * np.cos(PH); Y = (R + r * np.cos(TH)) * np.sin(PH); Z = r * np.sin(TH)
polys, cols = cuadros(X[::2, ::2], Y[::2, ::2], Z[::2, ::2], AZUL, ALFA)
for gc, colr, al in [(g0, AZUL, 0.45), (g1, AZUL * 0.8, 0.30)]:
    pts = np.stack([(R + r * np.cos(th)) * np.cos(gc), (R + r * np.cos(th)) * np.sin(gc), r * np.sin(th)], -1)
    pp, cc = abanico(pts[:-1], colr, al); polys += pp; cols += cc
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=AZUL, lw=1.1, zorder=4)
ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none", linewidths=0, zorder=1))
# ecuadores exterior e interior como contornos finos
for rr_, zz in [(R + r, 0), (R - r, 0)]:
    ax.plot(rr_ * np.cos(phi), rr_ * np.sin(phi), 0 * phi + zz, color=AZUL, lw=0.5, alpha=0.6, zorder=3)
ax.plot([0, 0], [0, 0], [-1.0, 1.15], color=TENUE, lw=0.9, zorder=5)
ps = np.linspace(0, 2 * np.pi, 400)
ax.plot(R * np.cos(ps), R * np.sin(ps), 0 * ps, color=ORO, lw=1.0, ls=(0, (4, 3)), alpha=0.95, zorder=6)
C = (R * np.cos(g0), R * np.sin(g0), 0)
ax.plot([0, C[0]], [0, C[1]], [0, 0], color=HIELO, lw=0.7, zorder=6)
ax.plot([C[0], C[0] + r * np.cos(g0)], [C[1], C[1] + r * np.sin(g0)], [0, 0], color=HIELO, lw=0.7, zorder=6)
ax.scatter(*C, color=ORO, s=14, depthshade=False, zorder=7)
rec, tam = guardar(fig, "pappus_b.png")
P = {"C": C, "eje": (0, 0, 1.15), "Rm": (C[0] / 2, C[1] / 2, 0),
     "r": (C[0] + 0.5 * r * np.cos(g0), C[1] + 0.5 * r * np.sin(g0), 0.0),
     "ruta": (R * np.cos(np.radians(150)), R * np.sin(np.radians(150)), 0)}
for k, v in P.items():
    x, y = a_cm(fig, ax, v, rec, tam, ANCHO["b"])
    coords.append(r"\def\papB%s{(%.3f,%.3f)}" % (k, x, y))
im = Image.open("pappus_b.png"); coords.append(r"\def\papBh{%.3f}" % (ANCHO["b"] * im.height / im.width))
plt.close(fig)

open("pappus_coords.tex", "w").write("\n".join(coords) + "\n")
print("\n".join(coords))
