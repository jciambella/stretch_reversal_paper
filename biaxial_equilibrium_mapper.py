#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
biaxial_equilibrium_mapper.py
=============================
Global equilibrium-set mapper for the proportional dead loading of an
incompressible, isotropic, hyperelastic membrane in plane stress.  The module
is self-contained -- pure NumPy/SciPy.

Problem
-------
Condensed plane-stress energy  Psi(l1,l2) = W(I1,I2),  l3 = 1/(l1 l2)  (exact
incompressibility, no pressure field).  Invariants

    I1 = l1^2 + l2^2 + 1/(l1 l2)^2 ,   I2 = 1/l1^2 + 1/l2^2 + (l1 l2)^2 .

Proportional dead (nominal) loading  Psi_1 = P,  Psi_2 = alpha P,  0 <= alpha < 1.
Eliminating P (P = Psi_1) the entire equilibrium set at fixed alpha is the
single scalar level curve

    g(l1,l2) := Psi_2 - alpha Psi_1 = 0 .                                  (E)

Every plane point is an equilibrium for exactly one load ratio
alpha_loc = Psi_2/Psi_1; (E) is its alpha = const level set.  The natural state
(1,1) lies on (E) for every alpha because Psi_1(1,1)=Psi_2(1,1)=0.

Quantities (all proved; symbolic checks reproduced by ``--selftest``):
    H   = [[Psi_11, Psi_12],[Psi_12, Psi_22]]        plane-stress Hessian
    D   = det H = Psi_11 Psi_22 - Psi_12^2           criticality measure
    N1  = Psi_22 - alpha Psi_12 ,  N2 = alpha Psi_11 - Psi_12
    r   = Psi_12 / Psi_11                             modulus ratio

Identities (verified symbolically):
    H (N1, N2)^T = D (1, alpha)^T          -> (N1,N2) is the branch tangent
    dl_i/dP = N_i / D                       -> reversal (dl2/dP=0) <=> N2 = 0
    grad g  = (-N2, N1)                      -> (E) is a regular curve unless
                                               N1=N2=0, which forces D=0.
Hence, on the equilibrium curve:
    * reversal point        : N2 = 0  with D > 0   (transverse stretch turns)
    * dual (l1) reversal     : N1 = 0              (never on accessible branch)
    * critical / limit point : D = 0
    * fold discriminant      : at D=0, with null vector n=(r,-1),
                               N1 + alpha N2 = Psi_11 (r - alpha)^2, which
                               vanishes with n.(1,alpha) = r - alpha;
                               != 0 -> limit point, = 0 -> bifurcation (alpha=1).

What the mapper does
--------------------
For a fixed alpha and a window in (l1,l2):
  1. enumerates the connected components of (E) crossing the seeding grid, by
     grid-seeded tangent-predictor / Newton-projection continuation in
     arclength, with proximity de-duplication of already-traced components;
  2. classifies each component as accessible (connected by continuation to the
     natural state (1,1)) or inaccessible;
  3. on each component locates the reversal locus (N2=0, D>0), the dual locus
     (N1=0), and the criticality locus (D=0), and reports the on-curve
     separation of the nearest reversal and critical points, flagging any
     coincidence (separation below tolerance);
  4. optionally lists every coexisting equilibrium at a chosen load P
     (Eriksson & Nordmark, Comput. Struct. 144, 2014: non-unique response).

Scope.  The output is computational support for the conjecture of the paper
that the reversal and criticality loci stay disjoint on branches accessible
from the natural state for alpha < 1; it is not a proof.  Enumeration is
resolution-limited: a component crossing no grid edge can be missed.  The 2x2
Hessian D governs only homogeneous coaxial perturbations.

Run
---
    python biaxial_equilibrium_mapper.py --selftest
    python biaxial_equilibrium_mapper.py --material gt --alpha 0.20 --plot
    python biaxial_equilibrium_mapper.py --material constant --alpha 0.70 --load 1.5
    python biaxial_equilibrium_mapper.py --conjecture-sweep gt
"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.optimize import brentq
except Exception:                                  # pragma: no cover
    brentq = None


# =============================================================================
# Numerical helpers
# =============================================================================

class _NP:
    """numpy math shim."""
    log = staticmethod(np.log)


def _invariants_np(l1, l2):
    J = l1 * l2
    trC = l1**2 + l2**2
    return trC + 1.0 / J**2, J**2 + trC / J**2


def _W12_np(W, I1, I2):
    """dW/dI1, dW/dI2 by central differences on the smooth scalar energy."""
    hI = 1e-6 * max(1.0, abs(I1))
    kI = 1e-6 * max(1.0, abs(I2))
    W1 = (W(I1 + hI, I2, _NP) - W(I1 - hI, I2, _NP)) / (2 * hI)
    W2 = (W(I1, I2 + kI, _NP) - W(I1, I2 - kI, _NP)) / (2 * kI)
    return W1, W2


def P1_np(W, l1, l2):
    I1, I2 = _invariants_np(l1, l2)
    W1, W2 = _W12_np(W, I1, I2)
    return W1 * (2 * l1 - 2.0 / (l1**3 * l2**2)) + W2 * (-2.0 / l1**3 + 2 * l1 * l2**2)


def P2_np(W, l1, l2):
    I1, I2 = _invariants_np(l1, l2)
    W1, W2 = _W12_np(W, I1, I2)
    return W1 * (2 * l2 - 2.0 / (l2**3 * l1**2)) + W2 * (-2.0 / l2**3 + 2 * l2 * l1**2)


def hess_np(W, l1, l2):
    """Plane-stress Hessian by one FD level on the closed-form gradients."""
    h = 1e-6 * max(1.0, l1)
    k = 1e-6 * max(1e-3, l2)
    P11 = (P1_np(W, l1 + h, l2) - P1_np(W, l1 - h, l2)) / (2 * h)
    P12 = (P1_np(W, l1, l2 + k) - P1_np(W, l1, l2 - k)) / (2 * k)
    P22 = (P2_np(W, l1, l2 + k) - P2_np(W, l1, l2 - k)) / (2 * k)
    return P11, P12, P22


def r_detH_np(W, l1, l2):
    P11, P12, P22 = hess_np(W, l1, l2)
    return P12 / P11, P11 * P22 - P12**2


# =============================================================================
# Material library
# (constitutive constants set with c1 = 1; the nominal load is reported and
#  plotted normalised by the small-strain modulus g0 = (W1+W2)|_ref = c10+c01,
#  carried out in analyze(), so P is O(1) on the right scale for every material)
# =============================================================================

def W_neo_hookean(I1, I2, xp):
    return (I1 - 3.0)

def W_mooney(c2):
    def W(I1, I2, xp):
        return (I1 - 3.0) + c2 * (I2 - 3.0)
    return W

def W_gent(Jm=30.0):
    def W(I1, I2, xp):
        return -0.5 * Jm * xp.log(1.0 - (I1 - 3.0) / Jm)
    return W

def W_yeoh(I1, I2, xp):
    e = (I1 - 3.0)
    return e - 0.05 * e**2 + 0.005 * e**3

def W_gent_thomas(c1=1.0, c2=1.0):
    def W(I1, I2, xp):
        return c1 * (I1 - 3.0) + c2 * xp.log(I2 / 3.0)
    return W

def W_I2_constant(I1, I2, xp):
    return (I2 - 3.0)

def W_I2_growing(I1, I2, xp):
    e = (I2 - 3.0)
    return e + 0.1 * e**2

def W_I2_decaying(I1, I2, xp):
    return 20.0 * xp.log(1.0 + 0.05 * (I2 - 3.0))


MATERIALS = {
    "neo":      (W_neo_hookean,           r"$W=(I_1-3)$"),
    "mooney13": (W_mooney(1.0 / 3.0),     "Mooney–Rivlin $c_2/c_1=1/3$"),
    "mooney02": (W_mooney(0.02),          "Mooney–Rivlin $c_2/c_1=0.02$"),
    "mooney100":(W_mooney(100.0),         "Mooney–Rivlin $c_2/c_1=100$"),
    "mooney1000":(W_mooney(1000.0),       "Mooney–Rivlin $c_2/c_1=1000$"),
    "gent":     (W_gent(30.0),            r"Gent $J_m=30$"),
    "yeoh":     (W_yeoh,                  r"Yeoh"),
    "gt":       (W_gent_thomas(1.0, 1.0), "Gent–Thomas $c_1=c_2=1$"),
    "constant": (W_I2_constant,           r"$W=(I_2-3)$"),
    "growing":  (W_I2_growing,            r"$W=(I_2-3)+0.1(I_2-3)^2$"),
    "decaying": (W_I2_decaying,           r"$W=20\ln(1+0.05(I_2-3))$"),
}

# Loads are normalised in analyze() by g0 = (W_1 + W_2) at the natural state
# (c1 + c2 for Mooney--Rivlin).
# The axis labels state the normalisation explicitly.
LOAD_LABELS = {
    "neo":       r"$P/c_1$",
    "mooney13":  r"$P/(c_1+c_2)$",
    "mooney02":  r"$P/(c_1+c_2)$",
    "mooney100": r"$P/(c_1+c_2)$",
    "mooney1000":r"$P/(c_1+c_2)$",
    "gent":      r"$2P/c_1$",         # g0 = W_1(3) = c_1/2
    "yeoh":      r"$P/c_1$",
    "gt":        r"$P/(c_1+c_2/3)$",  # g0 = c_1 + c_2/3
    "constant":  r"$P/c_2$",
    "growing":   r"$P/c_2$",
    "decaying":  r"$P/c_2$",
}

# Okabe-Ito palette (matches make_figures.py)
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
KEARSLEY = 3.0 ** (1.0 / 6.0)


# =============================================================================
# Pointwise equilibrium fields
# =============================================================================

def fields(W, alpha, l1, l2):
    """All equilibrium-relevant scalars at (l1,l2) for load ratio alpha.

    Returns dict with g, grad g, N1, N2, D, r, P(=Psi1), and the local load
    ratio alpha_loc = Psi2/Psi1 that makes (l1,l2) an equilibrium.
    """
    p1 = P1_np(W, l1, l2)
    p2 = P2_np(W, l1, l2)
    P11, P12, P22 = hess_np(W, l1, l2)
    g = p2 - alpha * p1
    N1 = P22 - alpha * P12
    N2 = alpha * P11 - P12
    D = P11 * P22 - P12**2
    return dict(g=g, grad=np.array([-N2, N1]), N1=N1, N2=N2, D=D,
                r=(P12 / P11 if P11 != 0 else np.nan), P=p1,
                alpha_loc=(p2 / p1 if abs(p1) > 1e-14 else np.nan))


# =============================================================================
# Arclength (tangent-predictor / Newton-projection) continuation of the equilibrium curve g = 0
# =============================================================================

def _project(W, alpha, p, iters=12, tol=1e-12):
    """Newton projection of p onto g=0 along grad g (minimal-norm correction)."""
    p = np.asarray(p, float).copy()
    for _ in range(iters):
        f = fields(W, alpha, p[0], p[1])
        gg = f["grad"]
        ng2 = gg @ gg
        if ng2 < 1e-300:
            break
        p = p - (f["g"] / ng2) * gg
        if abs(f["g"]) < tol:
            break
    return p


def _trace_one_way(W, alpha, seed, window, h0=0.02, h_max=0.06, h_min=1e-5,
                   max_steps=20000, sign=+1, close_tol=None):
    """Trace g=0 from `seed` in one direction; return Nx2 array of (l1,l2)."""
    lo, hi = window
    if close_tol is None:
        close_tol = 1.5 * h0
    p = _project(W, alpha, np.asarray(seed, float))
    pts = []
    h = h0
    tprev = None
    for _ in range(max_steps):
        if not (lo < p[0] < hi and lo < p[1] < hi):
            break
        pts.append(p.copy())
        f = fields(W, alpha, p[0], p[1])
        t = np.array([f["N1"], f["N2"]])          # branch tangent (N1, N2)
        nt = math.hypot(*t)
        if nt < 1e-13:
            break
        t = t / nt
        if tprev is not None and t @ tprev < 0:
            t = -t
        if tprev is None and sign < 0:
            t = -t
        tprev = t
        pc = _project(W, alpha, p + h * t)
        step = math.hypot(*(pc - p))
        if step > 3 * h or step < 1e-10:          # corrector overshoot/stall
            h *= 0.5
            if h < h_min:
                break
            continue
        p = pc
        h = min(h * 1.1, h_max)
        if len(pts) > 80 and math.hypot(*(p - pts[0])) < close_tol:
            break                                  # closed loop
    return np.array(pts) if pts else np.empty((0, 2))


def trace_component(W, alpha, seed, window, h0=0.02):
    """Trace the full connected component of g=0 through `seed` (both ways)."""
    fwd = _trace_one_way(W, alpha, seed, window, h0=h0, sign=+1)
    bwd = _trace_one_way(W, alpha, seed, window, h0=h0, sign=-1)
    if len(bwd) > 1 and len(fwd) > 0:
        comp = np.vstack([bwd[::-1], fwd])
    elif len(fwd) > 0:
        comp = fwd
    else:
        comp = bwd
    return comp


# =============================================================================
# Component container + analysis
# =============================================================================

@dataclass
class Component:
    pts: np.ndarray                 # (M,2) polyline in (l1,l2)
    alpha: float
    accessible: bool = False
    P: np.ndarray = field(default=None)        # load Psi1 along the polyline
    N1: np.ndarray = field(default=None)
    N2: np.ndarray = field(default=None)
    D: np.ndarray = field(default=None)
    reversals: list = field(default_factory=list)   # (l1,l2,P)  N2=0, D>0
    duals: list = field(default_factory=list)       # (l1,l2,P)  N1=0
    criticals: list = field(default_factory=list)   # (l1,l2,P,discriminant,kind)
    coincidence: float = None       # min stretch distance reversal<->critical


def _refine_zero(W, alpha, a, b, key):
    """Bisection on a scalar field `key` between curve points a,b."""
    a = a.copy(); b = b.copy()
    fa = fields(W, alpha, a[0], a[1])[key]
    for _ in range(40):
        m = _project(W, alpha, 0.5 * (a + b))
        fm = fields(W, alpha, m[0], m[1])[key]
        if abs(fm) < 1e-11 or math.hypot(*(b - a)) < 1e-9:
            return m
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def analyze(W, comp_pts, alpha):
    """Fill a Component: per-point fields, reversal/dual/critical points, folds."""
    c = Component(pts=comp_pts, alpha=alpha)
    M = len(comp_pts)
    P = np.empty(M); N1 = np.empty(M); N2 = np.empty(M); D = np.empty(M)
    for i, (l1, l2) in enumerate(comp_pts):
        f = fields(W, alpha, l1, l2)
        P[i], N1[i], N2[i], D[i] = f["P"], f["N1"], f["N2"], f["D"]
    c.P, c.N1, c.N2, c.D = P, N1, N2, D
    c.accessible = (np.hypot(comp_pts[:, 0] - 1.0, comp_pts[:, 1] - 1.0).min() < 0.05)

    # reversal points: N2 sign change with D>0 (genuine, away from natural state)
    for i in np.where(np.diff(np.sign(N2)) != 0)[0]:
        if D[i] <= 0 or D[i + 1] <= 0:
            continue
        if math.hypot(comp_pts[i, 0] - 1.0, comp_pts[i, 1] - 1.0) < 1e-3:
            continue
        pr = _refine_zero(W, alpha, comp_pts[i], comp_pts[i + 1], "N2")
        c.reversals.append((pr[0], pr[1], P1_np(W, pr[0], pr[1])))
    # dual l1-reversal points: N1 sign change
    for i in np.where(np.diff(np.sign(N1)) != 0)[0]:
        if math.hypot(comp_pts[i, 0] - 1.0, comp_pts[i, 1] - 1.0) < 1e-3:
            continue
        pd = _refine_zero(W, alpha, comp_pts[i], comp_pts[i + 1], "N1")
        c.duals.append((pd[0], pd[1], P1_np(W, pd[0], pd[1])))
    # critical / limit points: D sign change, classify by discriminant N1+alpha N2
    for i in np.where(np.diff(np.sign(D)) != 0)[0]:
        pc = _refine_zero(W, alpha, comp_pts[i], comp_pts[i + 1], "D")
        ff = fields(W, alpha, pc[0], pc[1])
        disc = ff["N1"] + alpha * ff["N2"]                # n.(1,alpha)
        kind = "limit point" if abs(disc) > 1e-6 else "bifurcation"
        c.criticals.append((pc[0], pc[1], P1_np(W, pc[0], pc[1]), disc, kind))

    # coincidence: nearest reversal <-> nearest critical, in stretch distance
    if c.reversals and c.criticals:
        c.coincidence = min(math.hypot(r[0] - q[0], r[1] - q[1])
                            for r in c.reversals for q in c.criticals)

    # Normalise the nominal load by the small-strain modulus
    #   g0 = (W_1 + W_2)|_ref = c10 + c01   (for Mooney--Rivlin),
    # i.e. half the infinitesimal shear modulus 2(W_1+W_2).  This makes P an
    # O(1) quantity on the right physical scale for every material -- in
    # particular it removes the spurious blow-up of P for strongly I2-dominated
    # Mooney--Rivlin, where c1 alone is a tiny fraction of the stiffness.  It
    # leaves neo-Hookean and W(I2) unchanged (g0 = 1 there).
    W1r, W2r = _W12_np(W, 3.0, 3.0)
    g0 = W1r + W2r
    if np.isfinite(g0) and g0 > 0:
        c.P = c.P / g0
        c.reversals = [(x, y, p / g0) for (x, y, p) in c.reversals]
        c.duals = [(x, y, p / g0) for (x, y, p) in c.duals]
        c.criticals = [(x, y, p / g0, disc, kind)
                       for (x, y, p, disc, kind) in c.criticals]
    return c


# =============================================================================
# Global enumeration
# =============================================================================

def enumerate_equilibria(W, alpha, window=(0.1, 18.0), n_grid=180,
                         h0=0.02, log_grid=True, verbose=False):
    """Enumerate all connected components of g=0 in `window` at load ratio alpha.

    Grid-seeding (sign changes of g across grid edges) + tangent-predictor
    continuation + proximity de-duplication of already-traced components.  The
    natural state (1,1) is always seeded so the accessible branch is found even
    when it crosses no grid edge cleanly.
    """
    lo, hi = window
    gv = (np.geomspace(lo, hi, n_grid) if log_grid
          else np.linspace(lo, hi, n_grid))
    G = np.empty((n_grid, n_grid))
    for i, l2 in enumerate(gv):
        for j, l1 in enumerate(gv):
            G[i, j] = fields(W, alpha, l1, l2)["g"]

    seeds = [(1.0, 1.0)]                            # always seed the natural state
    for i in range(n_grid):                         # horizontal edges
        row = G[i]
        for j in range(n_grid - 1):
            if row[j] * row[j + 1] < 0:
                f = row[j] / (row[j] - row[j + 1])
                seeds.append((gv[j] + f * (gv[j + 1] - gv[j]), gv[i]))
    for j in range(n_grid):                         # vertical edges
        col = G[:, j]
        for i in range(n_grid - 1):
            if col[i] * col[i + 1] < 0:
                f = col[i] / (col[i] - col[i + 1])
                seeds.append((gv[j], gv[i] + f * (gv[i + 1] - gv[i])))

    comps = []
    covered = []           # flat list of all traced points, for dedup
    merge_tol = max(3.0 * h0, 0.05)

    def is_covered(p, tol):
        for arr in covered:
            if np.hypot(arr[:, 0] - p[0], arr[:, 1] - p[1]).min() < tol:
                return True
        return False

    for s in seeds:
        sp = _project(W, alpha, np.asarray(s, float))
        if not (lo < sp[0] < hi and lo < sp[1] < hi):
            continue
        if is_covered(sp, merge_tol):
            continue
        pts = trace_component(W, alpha, sp, window, h0=h0)
        if len(pts) <= 3:
            continue
        # reject if the freshly traced curve mostly overlaps an existing one
        if covered:
            sample = pts[::max(1, len(pts) // 40)]
            on_old = sum(is_covered(q, merge_tol) for q in sample)
            if on_old > 0.6 * len(sample):
                continue
        comps.append(analyze(W, pts, alpha))
        covered.append(pts)
        if verbose:
            c = comps[-1]
            print(f"   traced component: {len(pts)} pts, "
                  f"{'ACCESSIBLE' if c.accessible else 'inaccessible'}")
    comps.sort(key=lambda c: (not c.accessible, -len(c.pts)))
    return comps


# =============================================================================
# Non-unique response at a fixed load (Eriksson & Nordmark 2014)
# =============================================================================

def equilibria_at_load(components, P_target):
    """All coexisting equilibria at load P_target: intersect components with P=const."""
    out = []
    for k, c in enumerate(components):
        Pc = c.P
        for i in np.where(np.diff(np.sign(Pc - P_target)) != 0)[0]:
            denom = (Pc[i + 1] - Pc[i])
            t = (P_target - Pc[i]) / denom if denom != 0 else 0.5
            p = c.pts[i] + t * (c.pts[i + 1] - c.pts[i])
            stab = "stable" if c.D[i] > 0 else "unstable"
            acc = "ACCESSIBLE" if c.accessible else "inaccessible"
            out.append((p[0], p[1], stab, acc, k))
    return out


# =============================================================================
# Closed-form reversal states (for the self-test)
# =============================================================================

def reversal_state_I1(alpha):
    A = 1.0 - 2.0 * alpha**2
    disc = A**2 - alpha**2
    if disc < 0:
        return None
    u3 = (A + math.sqrt(disc)) / alpha
    v = (1.0 + alpha * u3) / (u3 + alpha)
    u = u3 ** (1.0 / 3.0)
    return math.sqrt(u / v), math.sqrt(u * v)

def reversal_state_I2(alpha):
    if brentq is None:
        return None
    p = lambda x: 4 * alpha**2 * x**4 - 5 * alpha * x**3 - alpha * x + 2
    try:
        v = brentq(p, 1e-6, 1.0 - 1e-9)
    except Exception:
        return None
    s = 3 * alpha * v**2 / (2 - alpha * v)
    u = s ** (1.0 / 3.0)
    return math.sqrt(u / v), math.sqrt(u * v)


# =============================================================================
# Coincidence finder (alpha-correct): roots of {det H = 0, reversal locus = 0}
# =============================================================================

def find_coincidences(W, window=(0.2, 25.0), n_grid=420, verbose=True):
    """Locate genuine reversal/criticality coincidences and classify each.

    A coincidence is a point where det H = 0 meets a reversal locus, read with
    its own local load ratio alpha_loc = Psi2/Psi1:
        transverse  : Psi2 Psi11 - Psi1 Psi12 = 0   (N2 = 0)
        dual (l1)    : Psi1 Psi22 - Psi2 Psi12 = 0   (N1 = 0, never accessible)

    Crossings are found by intersecting the zero contours (robust where the
    loci run nearly tangent), then each is returned as
    (l1, l2, alpha_loc, kind, accessible), with `accessible` True iff the point
    lies on the component of g=0 (at alpha_loc) reachable from (1,1).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = window
    g = np.geomspace(lo, hi, n_grid)
    L1, L2 = np.meshgrid(g, g)
    D = np.empty_like(L1); E2 = np.empty_like(L1); E1 = np.empty_like(L1)
    for i in range(L1.shape[0]):
        for j in range(L1.shape[1]):
            l1, l2 = L1[i, j], L2[i, j]
            p1 = P1_np(W, l1, l2); p2 = P2_np(W, l1, l2)
            P11, P12, P22 = hess_np(W, l1, l2)
            D[i, j] = P11 * P22 - P12**2
            E2[i, j] = p2 * P11 - p1 * P12
            E1[i, j] = p1 * P22 - p2 * P12

    def _segs(field):
        cs = plt.contour(L1, L2, field, levels=[0.0])
        out = [s for s in cs.allsegs[0] if len(s) > 2]
        plt.clf()
        return out

    def _intersect(A, B):
        pts = []
        for a in A:
            for b in B:
                if (a[:, 0].max() < b[:, 0].min() or b[:, 0].max() < a[:, 0].min()
                        or a[:, 1].max() < b[:, 1].min() or b[:, 1].max() < a[:, 1].min()):
                    continue
                for i in range(len(a) - 1):
                    p, r = a[i], a[i + 1] - a[i]
                    for j in range(len(b) - 1):
                        q, s = b[j], b[j + 1] - b[j]
                        rxs = r[0] * s[1] - r[1] * s[0]
                        if abs(rxs) < 1e-14:
                            continue
                        t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / rxs
                        u = ((q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]) / rxs
                        if 0 <= t <= 1 and 0 <= u <= 1:
                            pts.append(p + t * r)
        # deduplicate
        uniq = []
        for p in pts:
            if not any(math.hypot(p[0] - q[0], p[1] - q[1]) < 0.1 for q in uniq):
                uniq.append(p)
        return uniq

    Dsegs = _segs(D)
    cross = ([(p, "transverse") for p in _intersect(Dsegs, _segs(E2))]
             + [(p, "dual") for p in _intersect(Dsegs, _segs(E1))])
    plt.close("all")

    found = []
    for (p, kd) in cross:
        l1, l2 = float(p[0]), float(p[1])
        p1 = P1_np(W, l1, l2)
        a_loc = (P2_np(W, l1, l2) / p1) if abs(p1) > 1e-12 else np.nan
        found.append((l1, l2, a_loc, kd))

    # Accessibility is decided structurally, not by long stiff continuations
    # (which hop between near-tangent strands at extreme stretch):
    #   * transverse coincidence (N2=0) is on the accessible branch only if its
    #     own local alpha lies in [0,1] and the point sits on the lambda_2-turning
    #     branch reachable from (1,1);
    #   * dual coincidence (N1=0) is a lambda_1-reversal; the accessible branch is
    #     monotone in lambda_1 under proportional dead loading from the natural
    #     state (alpha<1), so it carries no lambda_1-reversal -> never accessible.
    if verbose:
        tr = sorted([f for f in found if f[3] == "transverse"], key=lambda t: t[2])
        du = sorted([f for f in found if f[3] == "dual"], key=lambda t: t[2])
        print("Reversal / criticality coincidences (det H = 0 meets a reversal locus)")
        print(f"window lambda in [{lo:g}, {hi:g}]\n")
        print(f"transverse reversal (N2=0)  x  det H=0 :  {len(tr)} crossing(s)")
        if tr:
            amin = min(f[2] for f in tr); amax = max(f[2] for f in tr)
            n_le1 = sum(1 for f in tr if 0.0 <= f[2] <= 1.0)
            print(f"   local alpha in [{amin:.3f}, {amax:.3f}];  "
                  f"with 0<=alpha<=1: {n_le1}")
            for (l1, l2, a, _) in tr[:6]:
                print(f"     l1={l1:7.3f} l2={l2:7.3f}  alpha_loc={a:7.3f}")
            if len(tr) > 6:
                print(f"     ... ({len(tr) - 6} more, all alpha={amin:.1f}-{amax:.1f})")
        print(f"\ndual reversal (N1=0)        x  det H=0 :  {len(du)} crossing(s)")
        if du:
            amin = min(f[2] for f in du); amax = max(f[2] for f in du)
            l1lo = min(f[0] for f in du)
            print(f"   local alpha in [{amin:.3f}, {amax:.3f}];  "
                  f"all at extreme stretch (lambda_1 >= {l1lo:.1f}, lambda_2 ~ 1)")
            print("   on the dual (lambda_1-reversal) locus -> not on the "
                  "lambda_1-monotone accessible branch")
        n_phys = sum(1 for f in tr if 0.0 <= f[2] <= 1.0)
        print("\n  SUMMARY")
        print(f"  transverse crossings with local 0<=alpha<=1 : {n_phys}")
        print("  These are intersections of finite-difference zero contours and are")
        print("  not certified: at extreme stretch the difference scheme can produce")
        print("  crossings that are not common zeros.  The exact result for the")
        print("  energies of the paper is given by exact_elimination.py.")
    return found


# =============================================================================
# Reporting
# =============================================================================

def report(label, alpha, components):
    print("=" * 74)
    print(f"Equilibrium map :  {label}   alpha = {alpha:g}")
    print("=" * 74)
    n_acc = sum(c.accessible for c in components)
    print(f"{len(components)} connected component(s)  "
          f"({n_acc} accessible, {len(components) - n_acc} inaccessible)\n")
    for k, c in enumerate(components):
        tag = "ACCESSIBLE" if c.accessible else "inaccessible"
        l1lo, l1hi = c.pts[:, 0].min(), c.pts[:, 0].max()
        l2lo, l2hi = c.pts[:, 1].min(), c.pts[:, 1].max()
        print(f"-- component {k}  [{tag}]  {len(c.pts)} pts  "
              f"l1 in [{l1lo:.3f},{l1hi:.3f}]  l2 in [{l2lo:.3f},{l2hi:.3f}]  "
              f"P in [{c.P.min():.3f},{c.P.max():.3f}]")
        if c.reversals:
            for (x, y, p) in c.reversals:
                print(f"     reversal  (N2=0, D>0) : l1={x:.4f} l2={y:.4f} P={p:.4f}")
        else:
            print("     reversal  (N2=0, D>0) : none")
        for (x, y, p, disc, kind) in c.criticals:
            print(f"     critical  (det H=0)   : l1={x:.4f} l2={y:.4f} P={p:.4f}"
                  f"   N1+aN2={disc:+.3e} -> {kind}")
        if not c.criticals:
            print("     critical  (det H=0)   : none")
        if c.coincidence is not None:
            flag = "  *** COINCIDENCE ***" if c.coincidence < 0.1 else ""
            print(f"     nearest reversal<->critical stretch separation: "
                  f"{c.coincidence:.4f}{flag}")
    print()


# =============================================================================
# Plotting  (paper style: Okabe-Ito, det H=0 + reversal loci + Kearsley point)
# =============================================================================

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 12, "axes.labelsize": 15, "axes.titlesize": 15,
        "legend.fontsize": 11, "lines.linewidth": 1.5,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "figure.dpi": 200,
    })
    return plt


def _panel_label(ax, text):
    """Stamp a plain panel label (a)/(b)/(c) just above the top-left of an axis."""
    ax.text(0.0, 1.015, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="normal")


def plot_map(W, label, alpha, components, window, outfile,
             logscale=True, n_locus=320, loadlabel=r"$P$ (normalized)"):
    """2x2 figure (each event is a stationary point of one path coordinate):
       (a) (l1,l2) equilibrium map with det H=0 and reversal loci [spans left col];
       (b) load curve l2(P): the reversal is the l2 turning point;
       (c) load curve l1(P): the limit point is the P turning point (det H=0),
           drawn with l1 on the vertical so the fold appears as a vertical tangent.
    """
    plt = _setup_mpl()
    from matplotlib.lines import Line2D
    lo, hi = window
    fig = plt.figure(figsize=(9.0, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_l2 = fig.add_subplot(gs[0, 1])
    ax_l1 = fig.add_subplot(gs[1, 1])

    # ---------------- (a) equilibrium map ----------------
    gg = (np.geomspace(lo, hi, n_locus) if logscale else np.linspace(lo, hi, n_locus))
    L1, L2 = np.meshgrid(gg, gg)
    DD = np.empty_like(L1)        # det H
    EREV = np.empty_like(L1)      # N2 = 0 (reversal locus), read with local alpha
    EDUAL = np.empty_like(L1)     # N1 = 0 (dual l1-reversal locus), read with local alpha
    for i in range(L1.shape[0]):
        for j in range(L1.shape[1]):
            l1, l2 = L1[i, j], L2[i, j]
            p1 = P1_np(W, l1, l2); p2 = P2_np(W, l1, l2)
            P11, P12, P22 = hess_np(W, l1, l2)
            DD[i, j] = P11 * P22 - P12**2
            # N2 = alpha*Psi_11 - Psi_12 and N1 = Psi_22 - alpha*Psi_12, with the
            # prescribed alpha of the figure.  The local ratio Psi_2/Psi_1 must
            # not be used here: it is undefined at the natural state, where
            # Psi_1 = Psi_2 = 0, and would force the contour through (1,1).
            EREV[i, j] = alpha * P11 - P12        # N2 = 0
            EDUAL[i, j] = P22 - alpha * P12       # N1 = 0
    if (DD.min() < 0 < DD.max()):
        ax_map.contour(L1, L2, DD, levels=[0.0], colors=["0.15"],
                       linewidths=1.5, alpha=0.85, zorder=3)
        ax_map.plot([], [], color="0.15", lw=1.5, alpha=0.85, label=r"$\det\mathbf{H}=0$")
    if (EREV.min() < 0 < EREV.max()):
        ax_map.contour(L1, L2, EREV, levels=[0.0], colors=[OKABE_ITO[2]],
                       linewidths=1.0, linestyles="-", alpha=0.9, zorder=2)
        ax_map.plot([], [], color=OKABE_ITO[2], lw=1.0, label=r"reversal locus $N_2=0$")
    # dual locus N1=0: where the reversal/criticality coincidence sits for
    # non-polyconvex energies (Gent--Thomas), always off the accessible branch.
    if (EDUAL.min() < 0 < EDUAL.max()):
        ax_map.contour(L1, L2, EDUAL, levels=[0.0], colors=[OKABE_ITO[4]],
                       linewidths=0.8, linestyles="--", alpha=0.9, zorder=2)
        ax_map.plot([], [], color=OKABE_ITO[4], lw=0.8, ls="--",
                    label=r"dual locus $N_1=0$")

    # Reachability from the natural state, including dynamic snaps: a stable
    # branch the system jumps to across a fold is physically accessible.  Two
    # branches are snap-connected when they share a limit-point (fold) state, so
    # the upper branch beyond the second fold inherits accessibility.  Marking
    # then matches panels (b),(c): stable & reachable -> blue solid;
    # stable & disconnected -> orange dashed; unstable -> grey dotted points.
    def _limit_states(c):
        return [(l1, l2) for (l1, l2, p, d, k) in c.criticals if k == "limit point"]
    reach = {id(c): c.accessible for c in components}
    for _ in range(len(components) + 1):
        acc_lps = [q for c in components if reach[id(c)] for q in _limit_states(c)]
        grew = False
        for c in components:
            if reach[id(c)]:
                continue
            if any(math.hypot(l1 - q[0], l2 - q[1]) < 0.1
                   for (l1, l2) in _limit_states(c) for q in acc_lps):
                reach[id(c)] = True; grew = True
        if not grew:
            break

    labeled = set()
    def _branch(c, mask, color, lw, ls, lab):
        if not np.any(mask):
            return
        x = c.pts[:, 0].astype(float).copy(); y = c.pts[:, 1].astype(float).copy()
        x[~mask] = np.nan; y[~mask] = np.nan
        ax_map.plot(x, y, color=color, lw=lw, ls=ls, zorder=4,
                    label=(lab if lab not in labeled else None))
        labeled.add(lab)

    for c in components:
        stab = c.D > 0
        in_tension = c.P >= 0
        show = stab & in_tension
        if reach[id(c)]:
            _branch(c, show, OKABE_ITO[0], 4.5, "-", "accessible branch")
        else:
            _branch(c, show, OKABE_ITO[1], 1.0, (0, (4, 2)), "inaccessible branch")
        if np.any(~stab):       # unstable equilibria: grey dotted points, as in (b),(c)
            ax_map.plot(c.pts[~stab, 0], c.pts[~stab, 1], ".", ms=3.0, color="0.7",
                        zorder=3.2,
                        label=("unstable" if "unstable" not in labeled else None))
            labeled.add("unstable")
        for (x, y, p) in c.reversals:
            ax_map.plot(x, y, "o", ms=8, mfc="gold", mec="k", mew=1.0, zorder=6)
        for (x, y, p, disc, kind) in c.criticals:
            ax_map.plot(x, y, "X", ms=10, color="k", mfc="0.6", zorder=7)
    # reversal-point marker for the legend
    ax_map.plot([], [], "o", ms=8, mfc="gold", mec="k", mew=1.0,
                ls="", label="reversal point")
    # reference axes through the natural state (1,1)
    ax_map.axhline(1.0, color="0.6", lw=0.8, zorder=1)
    ax_map.axvline(1.0, color="0.6", lw=0.8, zorder=1)
    if logscale:
        ax_map.set_xscale("log"); ax_map.set_yscale("log")
    ax_map.set_xlim(lo, hi); ax_map.set_ylim(lo, hi)
    ax_map.set_xlabel(r"$\lambda_1$"); ax_map.set_ylabel(r"$\lambda_2$")
    ax_map.legend(frameon=True, framealpha=0.85, edgecolor="none", loc="upper right", fontsize=9.5, borderpad=0.5)
    _panel_label(ax_map, "(a)")

    # ---------------- load-curve panels ----------------
    # Colour = stable (det H>0, dead-load accessible); grey = unstable continuation.
    acc = [c for c in components if c.accessible]
    Pref = max((c.P.max() for c in acc), default=None)
    if Pref is None or Pref <= 0:
        Pref = max((c.P.max() for c in components), default=1.0)
    xr = 1.22 * Pref       # P-axis right limit (a little headroom for the snap)

    def load_panel(ax, j, ylabel):
        # j = 1 -> lambda_2 on the vertical;  j = 0 -> lambda_1 on the vertical
        # Colour by stability, not by structural accessibility: every stable
        # equilibrium (det H>0) is physically realizable -- the upper branch is
        # reached dynamically by the dead-load snap, so it is drawn blue too.
        # Unstable equilibria (det H<0) are dotted grey points.
        for c in components:
            stab = c.D > 0
            # draw the stable run as a continuous line, but break it at unstable
            # points (NaN) so the line never jumps across an unstable gap.
            Pline = c.P.astype(float).copy()
            yline = c.pts[:, j].astype(float).copy()
            Pline[~stab] = np.nan
            yline[~stab] = np.nan
            lw_c = 4.5 if reach[id(c)] else 1.0
            ax.plot(Pline, yline, "-", lw=lw_c, color=OKABE_ITO[0], zorder=3)
            ax.plot(c.P[~stab], c.pts[~stab, j], ".", ms=3.4, color="0.7", zorder=2)
            for (x, y, p) in c.reversals:
                ax.plot(p, (y if j == 1 else x), "o", ms=8, mfc="gold",
                        mec="k", mew=1.0, zorder=6)
            for (x, y, p, disc, kind) in c.criticals:
                ax.plot(p, (y if j == 1 else x), "X", ms=10, color="k", mfc="0.6", zorder=7)
        # reference axes through the natural state (P=0, lambda=1)
        ax.axhline(1.0, color="0.6", lw=0.8, zorder=1)
        ax.axvline(0.0, color="0.6", lw=0.8, zorder=1)
        ax.set_xlim(0, xr)
        ax.set_xlabel(loadlabel)
        ax.set_ylabel(ylabel)

    # (b) l2(P): reversal = l2 turning point.  Frame y on the physically
    # reachable branches inside the P-window (continuation-accessible plus any
    # branch reached by a dead-load snap), so the loading curve and its snap
    # target set the scale and a disconnected, inaccessible high-stretch branch
    # does not stretch the axis.
    load_panel(ax_l2, 1, r"$\lambda_2$")
    xmax = xr
    vis = [c.pts[c.P <= xmax, 1] for c in components if reach[id(c)]]
    vis = np.concatenate([v for v in vis if len(v)]) if vis else None
    if vis is not None and len(vis):
        ax_l2.set_ylim(0.9 * vis.min(), 1.1 * vis.max())
    _panel_label(ax_l2, "(b)")

    # ---- load-controlled snap-through / hysteresis loop (panel b) ----
    # Two folds on one S-shaped load curve bound a dead-load hysteresis loop.
    # Under increasing load the system rides the reference-connected stable arm
    # to the higher-load fold and snaps, at fixed P, to the other stable arm of
    # the same equilibrium curve; unloading, it rides that arm to the lower-load
    # fold and snaps back.  The snap direction in lambda_2 is read from the
    # geometry, not assumed, so the annotation is correct whether the post-snap
    # arm lies above the loading arm (e.g. Gent--Thomas, snap up) or below it
    # (e.g. strongly second-invariant Mooney--Rivlin, snap down).
    loadC = min((c for c in components if c.accessible),
                key=lambda c: float(c.P.min()), default=None)
    # contiguous stable arms of the loading S-curve, each (P[], l2[]) sorted in P
    runs = []
    if loadC is not None:
        st = loadC.D > 0
        idx = np.where(st)[0]
        for g in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(g):
                o = np.argsort(loadC.P[g])
                runs.append((loadC.P[g][o].astype(float),
                             loadC.pts[g, 1][o].astype(float)))
    # limit points of the loading S-curve only.  De-duplicate repeated detections
    # of the same fold by proximity in lambda_1 (which is invariant to the load
    # normalisation), not in P: once P is normalised by c10+c01 the two genuine
    # folds can sit ~0.1 apart in P yet remain far apart (~10) in lambda_1.
    lps = []
    for (l1, l2, p, disc, kind) in (loadC.criticals if loadC else []):
        if kind == "limit point" and not any(abs(l1 - q[1]) < 0.5 for q in lps):
            lps.append((p, l1, l2))

    def _on_run(run, P_t):
        Pg, Lg = run
        if Pg.min() - 1e-9 <= P_t <= Pg.max() + 1e-9:
            return float(np.interp(P_t, Pg, Lg))
        return None

    if len(lps) >= 2:
        lps.sort()
        P_unload, _l1u, l2_unload = lps[0]    # lower-load fold (unloading snap-back)
        P_load,   _l1l, l2_load   = lps[-1]   # higher-load fold (loading snap)

        def _target(P_t, l2_fold):        # stable arm farthest from the folding one
            cands = [v for v in (_on_run(r, P_t) for r in runs) if v is not None]
            cands = [v for v in cands if abs(v - l2_fold) > 1e-3]
            return max(cands, key=lambda v: abs(v - l2_fold)) if cands else None

        l2_jump = _target(P_load,   l2_load)     # loading snap target
        l2_back = _target(P_unload, l2_unload)   # unloading snap-back target
        jump = dict(arrowstyle="-|>", color="k", lw=1.8, mutation_scale=15)
        ride = dict(arrowstyle="-|>", color=OKABE_ITO[0], lw=2.2, mutation_scale=15)
        if l2_jump is not None:           # vertical snap at the loading fold
            ax_l2.annotate("", xy=(P_load, l2_jump), xytext=(P_load, l2_load),
                           arrowprops=jump, zorder=8)
            ax_l2.text(P_load, 0.5 * (l2_load + l2_jump), "snap  ", fontsize=9,
                       ha="right", va="center", color="k")
        if l2_back is not None:           # vertical snap-back at the unloading fold
            ax_l2.annotate("", xy=(P_unload, l2_back), xytext=(P_unload, l2_unload),
                           arrowprops=jump, zorder=8)
            ax_l2.text(P_unload, 0.5 * (l2_unload + l2_back), "  snap", fontsize=9,
                       ha="left", va="center", color="k")

        # loading arm = stable arm reaching the lowest P (carries the reference);
        # post-snap arm = stable arm carrying the loading snap target at P_load.
        load_run = min(runs, key=lambda r: r[0].min()) if runs else None
        post_run = None
        if l2_jump is not None:
            post_run = min((r for r in runs if _on_run(r, P_load) is not None),
                           key=lambda r: abs(_on_run(r, P_load) - l2_jump),
                           default=None)
        yl0, yl1 = ax_l2.get_ylim(); dy = 0.05 * (yl1 - yl0)
        if load_run is not None:          # loading direction: up the loading arm
            Pa, Pb = 0.28 * P_load, 0.42 * P_load
            ya, yb = _on_run(load_run, Pa), _on_run(load_run, Pb)
            if ya is not None and yb is not None:
                ax_l2.annotate("", xy=(Pb, yb), xytext=(Pa, ya), arrowprops=ride, zorder=9)
                ax_l2.text(0.5 * (Pa + Pb), 0.5 * (ya + yb) + dy, "loading",
                           fontsize=8.5, ha="center", va="bottom", color=OKABE_ITO[0])
        if post_run is not None and post_run is not load_run:   # unloading direction
            Pc = P_unload + 0.72 * (P_load - P_unload)
            Pd = P_unload + 0.32 * (P_load - P_unload)
            yc, yd = _on_run(post_run, Pc), _on_run(post_run, Pd)
            if yc is not None and yd is not None:
                ax_l2.annotate("", xy=(Pd, yd), xytext=(Pc, yc), arrowprops=ride, zorder=9)
                ax_l2.text(0.5 * (Pc + Pd), 0.5 * (yc + yd) + dy, "unloading",
                           fontsize=8.5, ha="center", va="bottom", color=OKABE_ITO[0])

    # (c) l1(P): limit point = P turning point (det H=0); l1 on vertical -> vertical fold
    load_panel(ax_l1, 0, r"$\lambda_1$")
    l1cap = next((2.6 * c.criticals[0][0] for c in acc if c.criticals), None)
    y1 = np.concatenate([c.pts[:, 0] for c in acc]) if acc else None
    if y1 is not None and len(y1):
        top = min(y1.max(), l1cap) if l1cap else y1.max()
        ax_l1.set_ylim(0.95, 1.1 * top)
    _panel_label(ax_l1, "(c)")

    handles = [
        Line2D([], [], ls="-", lw=3.0, color=OKABE_ITO[0], label="stable"),
        Line2D([], [], marker=".", ls="", ms=9, color="0.7", label="unstable"),
        Line2D([], [], marker="o", ls="", ms=8, mfc="gold",
               mec="k", mew=1.0, label="reversal"),
        Line2D([], [], marker="X", ls="", ms=9, color="k", mfc="0.6",
               label="limit point"),
    ]
    if len(lps) >= 2:        # snap arrows only drawn when a hysteresis loop exists
        handles.append(Line2D([], [], color="k", lw=1.8, label="snap (dead load)"))
    ax_l2.legend(handles=handles, fontsize=9.5, loc="best", frameon=True, framealpha=0.85, edgecolor="none")

    fig.savefig(outfile)
    plt.close(fig)
    print(f"  wrote {outfile}")


# =============================================================================
# Conjecture sweep: accessible-branch reversal vs criticality over alpha
# =============================================================================

def conjecture_sweep(W, label, alphas=None, window=(0.03, 80.0), outfile=None):
    """For each alpha, trace the accessible branch and record (i) whether the
    branch carries a critical point (det H=0) at all, and (ii) where it does,
    the stretch separation to the nearest reversal point (N2=0).

    Two regimes must be distinguished, because the disjointness question is
    only meaningful when a criticality exists on the accessible branch:

      * 'reversal, no criticality' : the accessible branch reverses but has no
        fold for this alpha.  The two loci cannot meet -- the separation
        question is vacuous.  This is the generic case for polyconvex test
        materials and for Gent--Thomas outside its fold window.
      * 'reversal & criticality'   : both events sit on the accessible branch;
        the separation ||d lambda|| is then the quantity the conjecture bounds
        away from zero for alpha<1.

    Returns rows (alpha, rev, crit, sep, status) with status in
    {'pair', 'rev_only', 'none'}."""
    if alphas is None:
        alphas = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    print(f"\nConjecture sweep (accessible branch):  {label}")
    print("  alpha |   reversal P (l1,l2)      |  criticality P (l1,l2)     "
          "| separation | accessible criticality?")
    for a in alphas:
        pts = _trace_one_way(W, a, np.array([1.0, 1.0]), window, h0=0.02, sign=+1)
        if len(pts) < 5:
            rows.append((a, None, None, None, "none")); continue
        c = analyze(W, pts, a)
        rev = c.reversals[0] if c.reversals else None
        crit = None; sep = None
        if c.reversals and c.criticals:
            # nearest reversal<->critical pair (the quantity the conjecture bounds)
            best = min(((r, q) for r in c.reversals for q in c.criticals),
                       key=lambda rq: math.hypot(rq[0][0] - rq[1][0],
                                                 rq[0][1] - rq[1][1]))
            rev, crit = best
            sep = math.hypot(rev[0] - crit[0], rev[1] - crit[1])
            status = "pair"
        elif c.reversals:
            status = "rev_only"
        elif c.criticals:
            # criticality but no reversal: keep the critical for completeness
            crit = c.criticals[0]; status = "crit_only"
        else:
            status = "none"
        rows.append((a, rev, crit, sep, status))
        rs = f"P={rev[2]:7.3f} ({rev[0]:.3f},{rev[1]:.3f})" if rev else "none"
        cs = f"P={crit[2]:7.3f} ({crit[0]:.3f},{crit[1]:.3f})" if crit else "none"
        ss = f"{sep:.4f}" if sep is not None else "   -   "
        present = "yes" if crit is not None else "no"
        print(f"   {a:.2f} | {rs:25s} | {cs:26s} | {ss:10s} | {present}")

    pairs = [r for r in rows if r[4] == "pair"]
    rev_only = [r for r in rows if r[4] == "rev_only"]
    print("\n  SUMMARY")
    if rev_only:
        a_lo = min(r[0] for r in rev_only); a_hi = max(r[0] for r in rev_only)
        print(f"  reversal but NO accessible criticality at {len(rev_only)} of "
              f"{len(rows)} sampled alpha (e.g. alpha in [{a_lo:.2f},{a_hi:.2f}]):")
        print("    the reversal and criticality loci cannot meet there -- the "
              "accessible branch carries no fold.")
    if pairs:
        a_lo = min(r[0] for r in pairs); a_hi = max(r[0] for r in pairs)
        amin = min(pairs, key=lambda r: r[3])
        print(f"  accessible criticality present only for alpha in "
              f"[{a_lo:.2f},{a_hi:.2f}] ({len(pairs)} sampled alpha):")
        print(f"    minimum reversal<->criticality separation there: "
              f"{amin[3]:.4f} at alpha={amin[0]:.2f}  (bounded away from 0).")
    else:
        print("  no alpha in the sweep places both a reversal and a criticality "
              "on the accessible branch:")
        print("    reversal and criticality are never coincident there "
              "(separation question is vacuous) -> support for the conjecture.")

    if outfile:
        plt = _setup_mpl()
        fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
        if pairs:
            xs = [r[0] for r in pairs]; ys = [r[3] for r in pairs]
            ax.plot(xs, ys, "o-", color=OKABE_ITO[0],
                    label="accessible fold present")
            ax.set_ylim(bottom=0)
        # mark alpha with a reversal but no accessible fold along the x-axis
        if rev_only:
            xr = [r[0] for r in rev_only]
            ax.plot(xr, [0.0] * len(xr), "|", color=OKABE_ITO[2], markersize=10,
                    markeredgewidth=1.4, label="reversal, no accessible fold")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"reversal$\leftrightarrow$criticality separation  $\|\Delta\lambda\|$")
        ax.set_title(f"accessible-branch disjointness: {label}", fontsize=8.5)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlim(0, 1)
        if pairs or rev_only:
            ax.legend(fontsize=7, loc="best")
        fig.savefig(outfile); plt.close(fig)
        print(f"  wrote {outfile}")
    return rows


# =============================================================================
# Self-test: reproduce closed-form reversal states and arc-length fold numbers
# =============================================================================

def selftest():
    print("SELF-TEST  (mapper vs closed forms and independently computed values)")
    print("-" * 66)
    ok = True

    # 1) neo-Hookean reversal (W(I1) class), alpha<1/2 -> closed-form universal state
    for a in (0.20, 1.0 / 3.0, 0.45):
        pts = _trace_one_way(W_neo_hookean, a, np.array([1.0, 1.0]),
                             (0.05, 60.0), h0=0.01, sign=+1)
        c = analyze(W_neo_hookean, pts, a)
        cf = reversal_state_I1(a)
        if c.reversals and cf:
            l1, l2, _ = c.reversals[0]
            err = math.hypot(l1 - cf[0], l2 - cf[1])
            ok &= err < 1e-3
            print(f"  neo-Hookean a={a:.3f}: mapper ({l1:.4f},{l2:.4f}) vs "
                  f"closed ({cf[0]:.4f},{cf[1]:.4f})  err={err:.2e}")
        else:
            ok = False
            print(f"  neo-Hookean a={a:.3f}: reversal not found  FAIL")

    # 2) pure W(I2), alpha=0.7: reversal P~0.692, limit point det H=0 P~2.077
    pts = _trace_one_way(W_I2_constant, 0.7, np.array([1.0, 1.0]),
                         (0.05, 8.0), h0=0.01, sign=+1)
    c = analyze(W_I2_constant, pts, 0.7)
    cf = reversal_state_I2(0.7)
    if c.reversals and cf:
        l1, l2, p = c.reversals[0]
        err = math.hypot(l1 - cf[0], l2 - cf[1])
        ok &= err < 1e-3
        print(f"  W(I2) a=0.700 reversal: mapper ({l1:.4f},{l2:.4f}, P={p:.4f}) "
              f"vs closed ({cf[0]:.4f},{cf[1]:.4f})  err={err:.2e}")
    fold = c.criticals[0] if c.criticals else None
    if fold:
        ok &= abs(fold[2] - 2.077) < 0.02
        print(f"  W(I2) a=0.700 limit point: P={fold[2]:.4f} (arc-length 2.077), "
              f"{fold[4]}, N1+aN2={fold[3]:+.3e}")
    else:
        ok = False
        print("  W(I2) a=0.700 limit point: not found  FAIL")

    print("-" * 66)
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", default="gt", choices=list(MATERIALS))
    ap.add_argument("--alpha", type=float, default=0.20)
    ap.add_argument("--window", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="stretch window LO HI; default 0.1 18 (0.1 35 for gt)")
    ap.add_argument("--ngrid", type=int, default=180)
    ap.add_argument("--linear", action="store_true",
                    help="linear seeding grid (default: log)")
    ap.add_argument("--load", type=float, default=None,
                    help="report all coexisting equilibria at this P "
                         "(non-unique response)")
    ap.add_argument("--plot", action="store_true", help="write the map figure")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--coincidences", default=None, choices=list(MATERIALS),
                    metavar="MATERIAL",
                    help="exploratory: intersect the finite-difference zero contours "
                         "of det H and of the reversal loci for MATERIAL (not "
                         "certified; the exact result is exact_elimination.py)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--conjecture-sweep", default=None, choices=list(MATERIALS),
                    metavar="MATERIAL",
                    help="sweep alpha on the accessible branch and report the "
                         "reversal<->criticality separation")
    ap.add_argument("--alphas", nargs=3, type=float, default=None,
                    metavar=("START", "STOP", "STEP"),
                    help="alpha sampling for --conjecture-sweep as START STOP "
                         "STEP (inclusive of START, exclusive of STOP); "
                         "default 0.05 1.0 0.05")
    args = ap.parse_args()

    # Gent--Thomas needs a wider stretch window to capture the relevant events.
    if args.window is None:
        # Strongly I2-dominated Mooney--Rivlin snaps to a far fibre state at the
        # higher-load fold; widen the window so that destination is captured.
        if args.material in ("mooney1000", "mooney100"):
            args.window = [0.1, 110.0]
        elif args.material == "gt":
            args.window = [0.1, 35.0]
        else:
            args.window = [0.1, 18.0]

    if args.selftest:
        selftest()
        return

    if args.coincidences:
        W, label = MATERIALS[args.coincidences]
        print(f"Material: {label}")
        find_coincidences(W, window=(args.window[0], args.window[1]))
        return

    if args.conjecture_sweep:
        W, label = MATERIALS[args.conjecture_sweep]
        out = (f"{args.outdir}/sweep_{args.conjecture_sweep}.pdf"
               if args.plot else None)
        alphas = None
        if args.alphas is not None:
            start, stop, step = args.alphas
            alphas = np.round(np.arange(start, stop, step), 4)
        conjecture_sweep(W, label, alphas=alphas, outfile=out)
        return

    W, label = MATERIALS[args.material]
    window = (args.window[0], args.window[1])
    comps = enumerate_equilibria(W, args.alpha, window=window,
                                 n_grid=args.ngrid, log_grid=not args.linear)
    report(label, args.alpha, comps)

    if args.load is not None:
        print(f"Coexisting equilibria at P = {args.load:g} "
              f"(Eriksson--Nordmark non-unique response):")
        sols = equilibria_at_load(comps, args.load)
        if not sols:
            print("  none in window")
        for (l1, l2, stab, acc, k) in sols:
            print(f"  l1={l1:.4f} l2={l2:.4f}  {stab:8s} {acc}  (component {k})")
        print()

    if args.plot:
        out = f"{args.outdir}/map_{args.material}_a{args.alpha:.2f}.pdf"
        plot_map(W, label, args.alpha, comps, window, out,
                 logscale=not args.linear,
                 loadlabel=LOAD_LABELS.get(args.material, r"$P$ (normalized)"))


if __name__ == "__main__":
    main()
