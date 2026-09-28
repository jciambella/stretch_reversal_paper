#!/usr/bin/env python3
"""
Figures for: 'Stable Stretch Reversal in the Biaxial Tension of Isotropic
Hyperelastic Membranes'.

Generates (paper figure numbers in brackets; Figs 3 and 4 are produced by
biaxial_equilibrium_mapper.py, see README.md):
  fig1_reversal_paths.pdf   - [Fig. 1] the phenomenon: lambda_2(P) for benign (NH) and
                              counter-intuitive (MR) reversals; degenerate
                              equibiaxial limit with the Kearsley point.
  fig2_universal_relations.pdf - [not used; Fig. 2 is fig2a below] universality: loading trajectories of three
                              W(I1) and three W(I2) materials collapsing onto
                              the analytic universal curves; common reversal
                              states.
  fig2a_universal_with_detH.pdf - [Fig. 2] as fig2, with the criticality
                              locus det H = 0 overlaid.
  fig3_census_terminal.pdf  - [Fig. 5] census and terminal states: r(lambda_1) against
                              alpha (crossings = stationary points) on a
                              symlog y-axis (r->0 spreading, r->infty fibre),
                              and the two attractors in log-log (slope +1
                              spreading lock v->alpha; slope -1/2 fibre collapse).

All paths are traced by continuation of the equilibrium constraint
Psi_2 = alpha * Psi_1 with Brent root-finding (xtol 1e-13); the Hessian of the
plane-stress energy is monitored pointwise (central differences) so that
stable (det H > 0) and post-critical portions can be distinguished.
"""
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 7.5, "lines.linewidth": 1.4,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "figure.dpi": 200,
})

# ----------------------------------------------------------------------
# constitutive machinery:  W = W(I1, I2),  user supplies W1, W2 callables
# ----------------------------------------------------------------------
def invariants(l1, l2):
    I1 = l1**2 + l2**2 + 1.0/(l1*l2)**2
    I2 = 1.0/l1**2 + 1.0/l2**2 + (l1*l2)**2
    return I1, I2

def model(W1, W2):
    def P1(l1, l2):
        i1, i2 = invariants(l1, l2)
        return (W1(i1, i2)*(2*l1 - 2.0/(l1**3*l2**2))
                + W2(i1, i2)*(-2.0/l1**3 + 2*l1*l2**2))
    def P2(l1, l2):
        i1, i2 = invariants(l1, l2)
        return (W1(i1, i2)*(2*l2 - 2.0/(l2**3*l1**2))
                + W2(i1, i2)*(-2.0/l2**3 + 2*l2*l1**2))
    def hess(l1, l2):
        h = 1e-6*max(1.0, l1); k = 1e-6*max(1e-3, l2)
        P11 = (P1(l1+h, l2) - P1(l1-h, l2))/(2*h)
        P12 = (P1(l1, l2+k) - P1(l1, l2-k))/(2*k)
        P22 = (P2(l1, l2+k) - P2(l1, l2-k))/(2*k)
        return P11, P12, P22
    return P1, P2, hess

def trace(P1, P2, hess, alpha, l1max, n=20000):
    """Continuation of Psi_2 = alpha Psi_1 in lambda_1. Returns columns:
       l1, l2, P, r=Psi12/Psi11, detH."""
    l1s = np.geomspace(1.0 + 1e-5, l1max, n)
    rows, g = [], 1.0
    for l1 in l1s:
        f = lambda l2: P2(l1, l2) - alpha*P1(l1, l2)
        lo, hi, ok = g*0.7, g*1.4, False
        for _ in range(90):
            try:
                if f(lo)*f(hi) <= 0: ok = True; break
            except Exception: pass
            lo *= 0.85; hi *= 1.18
            if lo < 1e-8: lo = 1e-8
        if not ok: break
        l2 = brentq(f, lo, hi, xtol=1e-13, rtol=1e-13); g = l2
        P11, P12, P22 = hess(l1, l2)
        rows.append((l1, l2, P1(l1, l2), P12/P11, P11*P22 - P12**2))
    return np.array(rows)

def stationary_points(arr, alpha):
    """Transversal crossings of r = alpha on the det H > 0 portion."""
    r, det = arr[:, 3], arr[:, 4]
    idx = np.argmax(det <= 0) if (det <= 0).any() else len(arr)
    s = np.sign(alpha - r[:idx]); c = np.where(np.diff(s) != 0)[0]
    return [(arr[i, 0], arr[i, 1], arr[i, 2], 'max' if s[i] > 0 else 'min')
            for i in c], idx

def detH_field(W1, W2, l1g, l2g):
    """det H = Psi11 Psi22 - Psi12^2 on the (l1,l2) grid. Depends on the
    material only, not on the load ratio alpha, so the locus det H = 0 is a
    single model-dependent curve in the stretch plane (same for every alpha)."""
    _, _, H = model(W1, W2)
    Z = np.empty((l2g.size, l1g.size))
    for i, l2 in enumerate(l2g):
        for j, l1 in enumerate(l1g):
            P11, P12, P22 = H(l1, l2)
            Z[i, j] = P11*P22 - P12**2
    return Z

# colour-blind-safe palette (Okabe-Ito)
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# ======================================================================
# Figure 1 - the phenomenon and the degenerate equibiaxial limit
# ======================================================================
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55), constrained_layout=True)

# (a) neo-Hookean, benign reversal (minimum of lambda_2)
P1, P2, H = model(lambda a, b: 1.0, lambda a, b: 0.0)
for col, al in zip(C, [0.20, 1/3, 0.45]):
    arr = trace(P1, P2, H, al, 60.0)
    m = arr[:, 2] <= 12.0
    ax[0].plot(arr[m, 2], arr[m, 1], color=col,
               label=rf"$\alpha={al:.2f}$" if al != 1/3 else r"$\alpha=1/3$")
    for (x1, x2, p, kind) in stationary_points(arr, al)[0]:
        ax[0].plot(p, x2, marker="o", ms=6, mfc="gold", mec="k", zorder=5)
ax[0].set_xlabel(r"$P/c_1$"); ax[0].set_ylabel(r"$\lambda_2$")
ax[0].set_title(r"(a) neo-Hookean, $0<\alpha<1/2$: benign reversal")
ax[0].legend(frameon=False); ax[0].set_xlim(0, 12)

# (b) Mooney-Rivlin c2/c1 = 1/3, counter-intuitive reversal -> Kearsley limit
c1, c2 = 1.0, 1.0/3.0
P1, P2, H = model(lambda a, b: c1, lambda a, b: c2)
for col, al in zip(C, [0.70, 0.85, 0.97]):
    arr = trace(P1, P2, H, al, 60.0)
    m = arr[:, 2] <= 12.0
    ax[1].plot(arr[m, 2], arr[m, 1], color=col, label=rf"$\alpha={al}$")
    for (x1, x2, p, kind) in stationary_points(arr, al)[0]:
        ax[1].plot(p, x2, marker="o", ms=6, mfc="gold", mec="k", zorder=5)
# symmetric equibiaxial branch alpha=1 and the Kearsley critical point
lam = np.linspace(1.0, 2.6, 400)
Psym = 2*(lam - lam**-5)*(c1 + c2*lam**2)
ax[1].plot(Psym, lam, color="k", ls="--", lw=1.1, label=r"$\alpha=1$ (symmetric)")
# Kearsley state at c2/c1 = 1/3: root of lambda^8 - 3 lambda^6 - 3 lambda^2 - 3
lam_c = brentq(lambda t: t**8 - 3*t**6 - 3*t**2 - 3, 1.5, 2.5, xtol=1e-15)
P_c = 2*(lam_c - lam_c**-5)*(c1 + c2*lam_c**2)
ax[1].plot(P_c, lam_c, marker="X", ms=8, mfc="gold", mec="k", mew=1.0, zorder=6, ls="",
           label=r"$\det\mathbf{H}=0$ (Kearsley)")
ax[1].set_xlabel(r"$P/c_1$"); ax[1].set_ylabel(r"$\lambda_2$")
ax[1].set_title("(b) Mooney–Rivlin $c_2/c_1=1/3$, $1/2<\\alpha\\leq1$")
ax[1].legend(frameon=False, loc="upper left"); ax[1].set_xlim(0, 12)
fig.savefig("fig1_reversal_paths.pdf")
fig.savefig("fig1_reversal_paths.png")
plt.close(fig)

# ======================================================================
# Figure 2 - universality of trajectory and reversal state per class,
#            shown for several biaxiality ratios alpha.  Colour/style
#            encodes the material (fixed across alpha); each alpha gives a
#            distinct bundle onto which the materials collapse, annotated
#            inline.  Equation numbers for the analytic curves are given
#            in the figure caption (via \eqref), not in the legend labels.
#
# A companion figure (fig2a) overlays the criticality locus det H = 0.  Since
# det H = Psi11 Psi22 - Psi12^2 is a function of (l1,l2) alone (independent of
# the load ratio alpha), the locus is a single model-dependent curve per
# material, drawn here by contouring det H over each panel's window.
# ======================================================================
MSTYLES = ["-", "--", ":"]

mats_I1 = [
    ("neo-Hookean",            lambda a, b: 1.0),
    (r"Gent ($J_m=30$)",       lambda a, b: 0.5/(1.0 - (a - 3.0)/30.0)),
    ("Yeoh",                   lambda a, b: 1.0 - 0.1*(a-3) + 0.015*(a-3)**2),
]
mats_I2 = [
    ("constant $W_2$",    lambda a, b: 1.0),
    ("growing $W_2$",     lambda a, b: 1.0 + 0.2*(b-3)),
    ("decaying $W_2$",    lambda a, b: 1.0/(1.0 + 0.05*(b-3))),
]
L1MAX_A, L1MAX_B = 3.6, 2.4
XLIM_A, YLIM_A = (1.0, 4.0), (0.75, 1.70)
XLIM_B, YLIM_B = (1.0, 2.4), (0.78, 1.30)

def overlay_detH(axis, mats, is_I1, xlim, ylim, tag):
    """Contour det H = 0 for every material in the panel and report whether
    the locus enters the window. Each material's locus is drawn in its own
    colour (thick, semi-transparent) so model dependence is visible; a single
    legend proxy (added last) labels the family. The equibiaxial Kearsley point
    3^(1/6) is marked but carries no legend entry."""
    l1g = np.linspace(xlim[0], xlim[1], 240)
    l2g = np.linspace(ylim[0], ylim[1], 240)
    L1, L2 = np.meshgrid(l1g, l2g)
    any_present = False
    for (name, Wf), col in zip(mats, C):
        W1f = Wf if is_I1 else (lambda a, b: 0.0)
        W2f = (lambda a, b: 0.0) if is_I1 else Wf
        Z = detH_field(W1f, W2f, l1g, l2g)
        zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
        present = (zmin < 0.0 < zmax)
        any_present = any_present or present
        if present:
            axis.contour(L1, L2, Z, levels=[0.0], colors=[col],
                         linewidths=2.6, alpha=0.45, zorder=2)
        # diagonal (equibiaxial) crossing, for reference vs Kearsley 3^(1/6)
        diag = np.array([detH_field(W1f, W2f, np.array([t]), np.array([t]))[0, 0]
                         for t in l1g])
        sgn = np.sign(diag); cr = np.where(np.diff(sgn) != 0)[0]
        diagx = [float(0.5*(l1g[i] + l1g[i+1])) for i in cr]
        print(f"[{tag}] {name:>16s}: det H=0 in window? {present} "
              f"(min={zmin:.3g}, max={zmax:.3g}); "
              f"equibiaxial crossings lambda={['%.4f'%x for x in diagx]}")
    # equibiaxial Kearsley reference point 3^(1/6) (marker only, no legend), and
    # a single legend proxy added last so it trails every other entry; both only
    # when a criticality locus actually enters this panel (none does for W(I1)).
    if any_present:
        lk = 3.0**(1/6.0)
        if xlim[0] <= lk <= xlim[1] and ylim[0] <= lk <= ylim[1]:
            axis.plot(lk, lk, marker="X", ms=8, mfc="gold", mec="k", mew=1.0, zorder=7, ls="",
                      label=r"Kearsley $\lambda=3^{1/6}$")
        axis.plot([], [], color="0.35", lw=2.6, alpha=0.55,
                  label=r"$\det\mathbf{H}=0$ locus")

def legend_loci_last(axis, loc):
    """Draw the legend with the two locus curves forced to the end, in the
    order  ... , reversal locus, det H = 0 locus  (Kearsley, if present, sits
    among the markers just before them)."""
    last = ["reversal locus", r"$\det\mathbf{H}=0$ locus"]
    h, l = axis.get_legend_handles_labels()
    keep = [(hi, li) for hi, li in zip(h, l) if li not in last]
    tail = [(hi, li) for hi, li in zip(h, l) if li in last]
    axis.legend([x[0] for x in keep + tail], [x[1] for x in keep + tail],
                frameon=False, loc=loc, fontsize=7)

def mark_detH_crossing(axis, arr, col, with_detH):
    """Hollow square at the first det H = 0 crossing of a loading trajectory
    (the limit point / fold). No legend entry."""
    if with_detH and (arr[:, 4] <= 0).any():
        ic = int(np.argmax(arr[:, 4] <= 0))
        if ic > 0:
            axis.plot(arr[ic, 0], arr[ic, 1], marker="X", color="k", mfc="0.6",
                      ms=8, zorder=7)

def draw_fig2(ax, with_detH=False):
    # ---- (a) class W(I1): benign reversal for alpha < 1/2 --------------
    for j, (al, allab) in enumerate([(0.20, "0.20"), (1/3, "1/3"), (0.45, "0.45")]):
        for (name, W1), col, ls in zip(mats_I1, C, MSTYLES):
            P1, P2, H = model(W1, lambda a, b: 0.0)
            arr = trace(P1, P2, H, al, L1MAX_A)
            ax[0].plot(arr[:, 0], arr[:, 1], color=col, ls=ls,
                       label=(name if j == 0 else None))
            mark_detH_crossing(ax[0], arr, col, with_detH)
        # closed-form reversal state (universal; marker only, no legend entry)
        A = 1 - 2*al**2; u3 = (A + np.sqrt(A**2 - al**2))/al; u_r = u3**(1/3.0)
        v_r = (1 + al*u3)/(u3 + al); l1r, l2r = np.sqrt(u_r/v_r), np.sqrt(u_r*v_r)
        ax[0].plot(l1r, l2r, marker="o", ms=6, mfc="gold", mec="k", zorder=6)
        # annotate alpha at the right end of the (collapsed) trajectory
        ax[0].annotate(rf"$\alpha={allab}$", xy=(arr[-1, 0], arr[-1, 1]),
                       xytext=(3, 0), textcoords="offset points",
                       fontsize=8, color="0.25", va="center")
        print(f"[fig2a] alpha={al:.3f}: reversal (l1,l2)=({l1r:.4f},{l2r:.4f})")
    # reversal-state locus (envelope), plotted last so it trails the materials
    # in the legend:  3s + v = s v (s + 3v)
    aa = np.linspace(0.02, 0.4999, 300)
    Aa = 1 - 2*aa**2; u3a = (Aa + np.sqrt(Aa**2 - aa**2))/aa
    vea = (1 + aa*u3a)/(u3a + aa); uea = u3a**(1/3.0)
    ax[0].plot(np.sqrt(uea/vea), np.sqrt(uea*vea), color="0.45", lw=1.3, ls=":",
               zorder=1, label="reversal locus")
    if with_detH:
        overlay_detH(ax[0], mats_I1, True, XLIM_A, YLIM_A, "detH-a")
    ax[0].set_xlabel(r"$\lambda_1$"); ax[0].set_ylabel(r"$\lambda_2$")
    ax[0].set_title(r"(a) class $W(I_1)$, $\alpha<1/2$")
    ax[0].set_xlim(*XLIM_A); ax[0].set_ylim(*YLIM_A)
    legend_loci_last(ax[0], "upper left")

    # ---- (b) class W(I2): counter-intuitive reversal for alpha > 1/2 ---
    # when the det H locus is overlaid, drop the bottom edge so the fold
    # (trajectory x locus) crossings near lambda_1 ~ 1.9 are not clipped
    ylim_b = (0.70, 1.30) if with_detH else YLIM_B
    for j, (al, allab, off) in enumerate([(0.60, "0.60", (1, -14)),
                                           (0.75, "0.75", (9, -6)),
                                           (0.88, "0.88", (7, 5))]):
        for (name, W2), col, ls in zip(mats_I2, C, MSTYLES):
            P1, P2, H = model(lambda a, b: 0.0, W2)
            arr = trace(P1, P2, H, al, L1MAX_B)
            ax[1].plot(arr[:, 0], arr[:, 1], color=col, ls=ls,
                       label=(name if j == 0 else None))
            mark_detH_crossing(ax[1], arr, col, with_detH)
        # reversal state: root of the quartic (marker only, no legend entry)
        p = lambda x: 4*al**2*x**4 - 5*al*x**3 - al*x + 2
        v_r = brentq(p, 1e-6, 1.0 - 1e-9)
        s_r = 3*al*v_r**2/(2 - al*v_r); u_r = s_r**(1/3.0)
        l1r, l2r = np.sqrt(u_r/v_r), np.sqrt(u_r*v_r)
        ax[1].plot(l1r, l2r, marker="o", ms=6, mfc="gold", mec="k", zorder=6)
        ax[1].annotate(rf"$\alpha={allab}$", xy=(l1r, l2r), xytext=off,
                       textcoords="offset points", fontsize=8, color="0.25")
        print(f"[fig2b] alpha={al:.2f}: v*={v_r:.4f}, reversal (l1,l2)=({l1r:.4f},{l2r:.4f})")
    # reversal-state locus (envelope), plotted last:  s + 3v = s v (5v - s)
    ab = np.linspace(0.5001, 0.99, 300)
    veb = np.array([brentq(lambda x: 4*a**2*x**4 - 5*a*x**3 - a*x + 2, 1e-6, 1 - 1e-9)
                    for a in ab])
    seb = 3*ab*veb**2/(2 - ab*veb); ueb = seb**(1/3.0)
    ax[1].plot(np.sqrt(ueb/veb), np.sqrt(ueb*veb), color="0.45", lw=1.3, ls=":",
               zorder=1, label="reversal locus")
    if with_detH:
        overlay_detH(ax[1], mats_I2, False, XLIM_B, ylim_b, "detH-b")
    ax[1].set_xlabel(r"$\lambda_1$"); ax[1].set_ylabel(r"$\lambda_2$")
    ax[1].set_title(r"(b) class $W(I_2)$, $\alpha>1/2$")
    ax[1].set_xlim(*XLIM_B); ax[1].set_ylim(*ylim_b)
    legend_loci_last(ax[1], "upper right")

# fig2: variant without the det H locus (not used in the paper)
fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
draw_fig2(ax, with_detH=False)
fig.savefig("fig2_universal_relations.pdf")
fig.savefig("fig2_universal_relations.png")
plt.close(fig)

# fig2a: Fig. 2 of the paper -- fig2 with the criticality locus det H = 0 overlaid
fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
draw_fig2(ax, with_detH=True)
fig.savefig("fig2a_universal_with_detH.pdf")
fig.savefig("fig2a_universal_with_detH.png")
plt.close(fig)

# ======================================================================
# Figure 5 of the paper (file fig3_*) - the r ratio against alpha and the two terminal states
# ======================================================================
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7), constrained_layout=True)

# legend labels are model names only; loading ratios / coefficients are given
# in the note caption. Colours are shared across both panels per model.
cases = [
    ("neo-Hookean",        lambda a,b:1.0, lambda a,b:0.0,   1/3,  C[0]),
    (r"MR $c_2/c_1=1/3$",  lambda a,b:1.0, lambda a,b:1/3.0,  0.7,  C[1]),
    (r"MR $c_2/c_1=0.02$", lambda a,b:1.0, lambda a,b:0.02,   0.3,  C[2]),
    ("Gent–Thomas",        lambda a,b:1.0, lambda a,b:1.0/b,  0.55, C[3]),
]
store = {}
for name, W1f, W2f, al, col in cases:
    P1, P2, H = model(W1f, W2f)
    arr = trace(P1, P2, H, al, 1000.0, n=80000)
    store[name] = (arr, al, col)
    ax[0].plot(arr[:, 0], arr[:, 3], color=col, label=name)
    ax[0].axhline(al, color=col, lw=0.7, ls=":", alpha=0.8)
    sp, _ = stationary_points(arr, al)
    for (x1, x2, p, kind) in sp:
        ax[0].plot(x1, al, marker="o", ms=6, mfc="gold", mec="k", zorder=6)
# r=Psi12/Psi11 stays bounded for the spreading attractor (r->0) but diverges
# for the fibre attractor (r->infty); a symmetric-log y-axis keeps the census
# crossings of r=alpha (all in 0<r<1) legible while showing the fibre blow-up
# without clipping the curves at the frame.
ax[0].set_xscale("log"); ax[0].set_yscale("symlog", linthresh=1.0)
ax[0].set_xlabel(r"$\lambda_1$"); ax[0].set_ylabel(r"$r=\Psi_{12}/\Psi_{11}$")
ax[0].set_ylim(0.0, 30.0); ax[0].set_xlim(1, 1000)
ax[0].axhline(0.5, color="k", lw=0.6, alpha=0.5)
ax[0].annotate(r"$r(1,1)=\frac{1}{2}$", xy=(950, 0.6), ha="right", fontsize=8)
# fibre attractor (r->infty) labels the upper family; spreading (r->0) the lower
# one, placed at large lambda_1 where the neo-Hookean / Gent--Thomas curves
# actually flatten onto zero (not near the reference state).
ax[0].annotate(r"$r\to\infty$ (fibre)", xy=(250, 14.0), ha="center",
               fontsize=7.5, color="0.30")
ax[0].annotate(r"$r\to0$ (spreading)", xy=(90, 0.02), ha="center",
               fontsize=7.5, color="0.30")
ax[0].plot([], [], marker="o", ms=6, mfc="gold", mec="k", ls="", label="reversal point")
ax[0].legend(frameon=True, framealpha=0.85, edgecolor="none",
             loc="upper left", fontsize=6.0)

# (b) terminal states, log-log, for the same four materials as panel (a):
# neo-Hookean and Gent--Thomas lock onto the spreading attractor (slope +1),
# the two Mooney--Rivlin collapse into the fibre state (slope -1/2).
for name, _w1, _w2, _al, _c in cases:
    arr, al, col = store[name]
    ax[1].loglog(arr[:, 0], arr[:, 1], color=col, label=name)
# guides: spreading locks v -> v_inf (slope +1, neo-Hookean and Gent--Thomas)
# and the fibre slope -1/2
x = np.array([20, 1000])
for vinf, col in [(1/3, C[0]), (0.55, C[3])]:
    ax[1].loglog(x, vinf*x, color=col, lw=0.7, ls="--", alpha=0.7)
xg = np.array([4, 1000])
ax[1].loglog(xg, 1.05*xg**-0.5, color="0.3", lw=0.8, ls="--")
ax[1].annotate(r"$\propto\lambda_1^{-1/2}$", xy=(400, 0.018), ha="center",
               fontsize=8.5, color="0.25")
ax[1].annotate(r"$\propto\lambda_1$", xy=(120, 250), ha="center",
               fontsize=8.5, color="0.25")
ax[1].set_xlabel(r"$\lambda_1$"); ax[1].set_ylabel(r"$\lambda_2$")
ax[1].set_xlim(1, 1000); ax[1].set_ylim(8e-3, 700)
ax[1].legend(frameon=True, framealpha=0.85, edgecolor="none", loc="upper left", fontsize=6.0)
# panel labels (a),(b) just above the top-left corner, clear of the legends
for _a, _lab in zip(ax, ["(a)", "(b)"]):
    _a.text(0.0, 1.02, _lab, transform=_a.transAxes, va="bottom", ha="left",
            fontsize=10)
fig.savefig("fig3_census_terminal.pdf")
fig.savefig("fig3_census_terminal.png")
plt.close(fig)
print("done")
