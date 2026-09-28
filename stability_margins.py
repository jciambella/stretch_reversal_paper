#!/usr/bin/env python3
r"""
stability_margins.py -- incremental-stability margins for the unequal-biaxial
reversal problem (J. Ciambella, Proc. R. Soc. A).

Builds, from exact symbolic derivatives, the quantities that Remark 2 (Sect. 4) and
Table 1 (Appendix A) of the manuscript report:

  N1  Ogden's instantaneous moduli in principal axes, eqn (4.4) of
      Ogden (1987) = eqn (8) of Haughton (1987).
  N2  strong ellipticity of the incompressible incremental problem.
  N3  the homogeneous dead-load margin det H / Psi_11^2 and the in-plane
      Cauchy stresses t_i = lambda_i Psi_i.
  N4  Haughton's criterion for sinusoidal incremental modes and shear bands,
      eqn (23) of Haughton (1987).
  N5  Ogden's conditions (4.12)-(4.13).  These are derived under
      symmetric loading -- his (3.1) is Psi_1 = Psi_2, i.e. alpha = 1 --
      and are therefore reported only as an alpha -> 1 cross-check on N4,
      never as a criterion at alpha < 1.
  N6  minimisation of every margin along the branch from (1,1) to the
      reversal, which is the question of whether a mode can precede it.

Normalisation by the ground-state shear modulus mu0 = 2(W_1 + W_2)|_(1,1)
follows the physical dimension of each quantity: stresses and the strong-
ellipticity margin (dimension of stress) by mu0; Haughton's I(gamma), which is
quadratic in the moduli, by mu0**2.  Every tabulated entry is therefore
invariant under W -> cW.

Independent of biaxial_equilibrium_mapper.py, which forms Psi_ij
by nested central differences; the moduli need W_ij in three stretches, which
that path cannot supply reliably.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Material library.  Energies in the invariants, c1 = 1 throughout, matching
# MATERIALS in biaxial_equilibrium_mapper.py exactly.
# ---------------------------------------------------------------------------
_i1, _i2 = sp.symbols("i1 i2", positive=True)
_e1, _e2 = _i1 - 3, _i2 - 3

MATERIALS = {
    "neo":        (_e1,                                   r"neo-Hookean"),
    "mooney13":   (_e1 + sp.Rational(1, 3) * _e2,         r"Mooney--Rivlin, $c_2/c_1=1/3$"),
    "mooney02":   (_e1 + sp.Rational(1, 50) * _e2,        r"Mooney--Rivlin, $c_2/c_1=0.02$"),
    "mooney100":  (_e1 + 100 * _e2,                       r"Mooney--Rivlin, $c_2/c_1=100$"),
    "mooney1000": (_e1 + 1000 * _e2,                      r"Mooney--Rivlin, $c_2/c_1=1000$"),
    "gent":       (-sp.Rational(30, 2) * sp.log(1 - _e1 / 30), r"Gent, $J_m=30$"),
    "yeoh":       (_e1 - sp.Rational(5, 100) * _e1**2
                   + sp.Rational(5, 1000) * _e1**3,       r"Yeoh"),
    "gt":         (_e1 + sp.log(_i2 / 3),                 r"Gent--Thomas, $c_2/c_1=1$"),
    "constant":   (_e2,                                   r"$W=c_2(I_2-3)$"),
    "growing":    (_e2 + sp.Rational(1, 10) * _e2**2,     r"$W(I_2)$, growing $W_2$"),
    "decaying":   (20 * sp.log(1 + _e2 / 20),             r"$W(I_2)$, decaying $W_2$"),
}


class Material:
    """Exact derivatives of one energy, lambdified once."""

    def __init__(self, key: str):
        self.key = key
        Wi, self.label = MATERIALS[key]
        self.mu0 = float(2 * (sp.diff(Wi, _i1) + sp.diff(Wi, _i2)).subs({_i1: 3, _i2: 3}))

        l1, l2, l3 = sp.symbols("l1 l2 l3", positive=True)
        I1 = l1**2 + l2**2 + l3**2
        I2 = l1**2 * l2**2 + l2**2 * l3**2 + l3**2 * l1**2
        W3 = Wi.subs({_i1: I1, _i2: I2})
        L = (l1, l2, l3)
        self._W3 = sp.lambdify(L, [[sp.diff(W3, a) for a in L],
                                   [[sp.diff(W3, a, b) for b in L] for a in L]], "numpy")

        # reduced plane-stress energy Psi(l1,l2) = W(l1, l2, 1/(l1 l2))
        Psi = W3.subs(l3, 1 / (l1 * l2))
        self._Psi = sp.lambdify((l1, l2), [sp.diff(Psi, l1), sp.diff(Psi, l2),
                                           sp.diff(Psi, l1, 2), sp.diff(Psi, l1, l2),
                                           sp.diff(Psi, l2, 2)], "numpy")
        self._sym = (W3, Psi, L)

    def psi(self, l1, l2):
        """(Psi_1, Psi_2, Psi_11, Psi_12, Psi_22) -- exact."""
        return [float(v) for v in self._Psi(float(l1), float(l2))]

    def moduli(self, l1, l2, coincide_tol=1e-6):
        """Ogden (4.4) / Haughton (8).  Returns (A4, Aijij, Aijji, sigma)."""
        lam = np.array([l1, l2, 1.0 / (l1 * l2)], float)
        W, Wd = self._W3(*lam)
        W = np.array(W, float)
        Wd = np.array(Wd, float)
        A4 = np.array([[lam[i] * lam[j] * Wd[i][j] for j in range(3)] for i in range(3)])
        lW = lam * W                      # lambda_i W_i
        Aa = np.zeros((3, 3))
        Ab = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if abs(lam[i] - lam[j]) < coincide_tol:
                    Aa[i, j] = 0.5 * (A4[i, i] - A4[i, j] + lW[i])
                else:
                    Aa[i, j] = (lW[i] - lW[j]) * lam[i]**2 / (lam[i]**2 - lam[j]**2)
                Ab[i, j] = Aa[i, j] - lW[i]
        # membrane Cauchy stresses, Haughton (3): sigma_a = lambda_a Psi_a
        p1, p2 = self.psi(l1, l2)[:2]
        sigma = np.array([l1 * p1, l2 * p2, 0.0])
        return A4, Aa, Ab, sigma


# ---------------------------------------------------------------------------
# N2 -- strong ellipticity
# ---------------------------------------------------------------------------
_PLANES = [(0, 1), (1, 2), (2, 0)]
_PLANE_NAME = {(0, 1): "12", (1, 2): "23", (2, 0): "31"}


def se_triple(A4, Aa, Ab, i, j):
    a = Aa[i, j]
    c = Aa[j, i]
    b = A4[i, i] + A4[j, j] - 2 * A4[i, j] - 2 * Ab[i, j]
    return a, b, c


def se_margin(mat: Material, l1, l2):
    """min over the 3 principal planes of min(a, c, b+2 sqrt(ac)) / mu0,
    with the binding inequality recorded."""
    A4, Aa, Ab, _ = mat.moduli(l1, l2)
    best, which = math.inf, ""
    for (i, j) in _PLANES:
        a, b, c = se_triple(A4, Aa, Ab, i, j)
        root = 2 * math.sqrt(a * c) if (a > 0 and c > 0) else float("nan")
        for val, lab in ((a, "a"), (c, "c"), (b + root, "b+2sqrt(ac)")):
            if not math.isnan(val) and val < best:
                best, which = val, f"{lab} [{_PLANE_NAME[(i, j)]}]"
    return best / mat.mu0, which


def _fibonacci_sphere(n):
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    rho = np.sqrt(1.0 - z * z)
    th = np.pi * (3.0 - np.sqrt(5.0)) * k
    return np.column_stack((rho * np.cos(th), rho * np.sin(th), z))


def se_brute(mat: Material, l1, l2, n=20000):
    """Independent Legendre-Hadamard check: least eigenvalue of the acoustic
    tensor Q(n)_ik = A_ijkl n_j n_l restricted to the plane perpendicular to n
    (incompressibility, m . n = 0), minimised over n unit normals spread over
    the whole unit sphere -- not only the principal planes used by se_margin.
    A sampled minimum can only overestimate the true one, so the check is
    se_brute >= se_margin with a small discretisation gap."""
    A4, Aa, Ab, _ = mat.moduli(l1, l2)
    A = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            A[i, i, j, j] = A4[i, j]
            if i != j:
                A[i, j, i, j] = Aa[i, j]
                A[i, j, j, i] = Ab[i, j]
    best = math.inf
    for nv in _fibonacci_sphere(n):
        Q = np.einsum("ijkl,j,l->ik", A, nv, nv)
        e1 = np.cross(nv, [1.0, 0.0, 0.0] if abs(nv[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nv, e1)
        E = np.column_stack((e1, e2))
        B = E.T @ Q @ E
        best = min(best, float(np.linalg.eigvalsh(0.5 * (B + B.T)).min()))
    return best / mat.mu0


# ---------------------------------------------------------------------------
# N4 -- Haughton (1987) eqn (23):  the single criterion for the sinusoidal
# incremental modes (gamma = m/n) and for shear bands (gamma = n1/n2).
#
#   (B_2121 + g^2 l1^2 Psi_11)(g^2 B_1212 + l2^2 Psi_22)
#       - g^2 (l1 l2 Psi_12 + B_1212 - sigma_11)^2 = 0
#
# A mode exists iff this has a real root in g^2 >= 0.
# ---------------------------------------------------------------------------
def haughton_I(mat: Material, l1, l2, g2):
    """LHS of eqn (23) as a function of gamma^2 (accepts arrays)."""
    A4, Aa, Ab, sig = mat.moduli(l1, l2)
    _, _, P11, P12, P22 = mat.psi(l1, l2)
    B1212 = Aa[0, 1]
    B2121 = Aa[1, 0]
    g2 = np.asarray(g2, float)
    return ((B2121 + g2 * l1**2 * P11) * (g2 * B1212 + l2**2 * P22)
            - g2 * (l1 * l2 * P12 + B1212 - sig[0])**2)


def haughton_margin(mat: Material, l1, l2, n=4001, gmax=1e3):
    """min over gamma of eqn (23), normalised by mu0**2 (I is quadratic in the
    moduli), and whether a real root in gamma^2 >= 0 exists.

    I(g2) = A g2^2 + B g2 + C is a quadratic in g2 = gamma^2, so the search is
    exact rather than sampled: C = I(0) is the gamma = 0 shear-band limit, A the
    gamma -> infinity limit (I ~ A g2^2), and the interior minimum, if any, sits
    at g2* = -B/(2A).  The logarithmic sweep is kept only as a cross-check."""
    A4, Aa, Ab, sig = mat.moduli(l1, l2)
    _, _, P11, P12, P22 = mat.psi(l1, l2)
    B1212, B2121 = Aa[0, 1], Aa[1, 0]
    qa = l1**2 * P11 * B1212                                  # gamma -> infinity
    qb = (B2121 * B1212 + l1**2 * l2**2 * P11 * P22
          - (l1 * l2 * P12 + B1212 - sig[0])**2)
    qc = B2121 * l2**2 * P22                                  # gamma = 0
    cands = [qc]
    if qa > 0:
        g2s = -qb / (2 * qa)
        if g2s > 0:
            cands.append(qa * g2s**2 + qb * g2s + qc)
    vmin = min(cands) if qa > 0 else -math.inf               # A < 0: I -> -inf
    disc = qb * qb - 4 * qa * qc
    has_root = (qa <= 0 or qc <= 0
                or (qb < 0 and disc >= 0))                   # real root, g2 >= 0
    # cross-check against the sampled form
    g2 = np.concatenate(([0.0], np.geomspace(gmax**-2, gmax**2, n)))
    v = haughton_I(mat, l1, l2, g2)
    assert float(v.min()) >= vmin - 1e-9 * max(1.0, abs(vmin)), "Haughton quadratic mismatch"
    return float(vmin) / mat.mu0**2, bool(has_root), float(qa) / mat.mu0**2


# ---------------------------------------------------------------------------
# N5 -- Ogden (1987) (4.12)-(4.13), valid only for alpha = 1 (his 3.1).
# ---------------------------------------------------------------------------
def ogden_conditions(mat: Material, l1, l2):
    """(O1, O2) normalised by mu0.  A real inhomogeneous mode requires
    O1 < 0 and O2 >= 0, so the mode is excluded as soon as either fails."""
    P1, _, P11, P12, P22 = mat.psi(l1, l2)
    detH = P11 * P22 - P12**2
    O1 = detH + 2 * P1 * P12 / (l1 + l2)
    O2 = ((l1 + l2) * detH + 2 * P1 * P12)**2 - 4 * P1**2 * P11 * P22
    return O1 / mat.mu0**2, O2 / mat.mu0**4


def ogden_secular(mat: Material, l1, l2, g2):
    """Ogden (4.10) in gamma^2, for the alpha=1 cross-check against (23)."""
    P1, _, P11, P12, P22 = mat.psi(l1, l2)
    g2 = np.asarray(g2, float)
    return (P1 * P11 * g2**2
            + ((l1 + l2) * (P11 * P22 - P12**2) + 2 * P1 * P12) * g2
            + P1 * P22)


# ---------------------------------------------------------------------------
# N3 -- homogeneous dead-load margin and Cauchy stresses
# ---------------------------------------------------------------------------
def dead_load_margin(mat: Material, l1, l2, alpha):
    _, _, P11, P12, P22 = mat.psi(l1, l2)
    return P22 / P11 - alpha**2


def cauchy(mat: Material, l1, l2):
    P1, P2 = mat.psi(l1, l2)[:2]
    return l1 * P1, l2 * P2


# ---------------------------------------------------------------------------
# Equilibrium branch:  g = Psi_2 - alpha Psi_1 = 0, tangent (N1, N2).
# ---------------------------------------------------------------------------
def fields(mat: Material, l1, l2, alpha):
    P1, P2, P11, P12, P22 = mat.psi(l1, l2)
    return dict(g=P2 - alpha * P1,
                N1=P22 - alpha * P12,
                N2=alpha * P11 - P12,
                detH=P11 * P22 - P12**2,
                r=P12 / P11, P=P1)


def _project(mat, p, alpha, iters=12, tol=1e-12):
    """Minimal-norm Newton projection onto g = 0."""
    p = np.array(p, float)
    for _ in range(iters):
        f = fields(mat, p[0], p[1], alpha)
        if abs(f["g"]) < tol:
            break
        grad = np.array([-f["N2"], f["N1"]])
        n2 = grad @ grad
        if n2 < 1e-300:
            break
        p = p - (f["g"] / n2) * grad
    return p


def trace(mat: Material, alpha, h0=0.02, hmax=0.06, hmin=1e-5,
          smax=60.0, l1max=400.0):
    """Arclength continuation of g=0 from the reference state, in the
    direction of increasing lambda_1.  Returns arclength and points."""
    p = _project(mat, (1.0 + 1e-6, 1.0), alpha)
    pts, ss = [p.copy()], [0.0]
    h, s = h0, 0.0
    tprev = None
    for _ in range(20000):
        f = fields(mat, p[0], p[1], alpha)
        t = np.array([f["N1"], f["N2"]])
        nt = np.hypot(*t)
        if nt < 1e-13:
            break
        t = t / nt
        if tprev is not None and t @ tprev < 0:
            t = -t
        if tprev is None and t[0] < 0:
            t = -t
        q = _project(mat, p + h * t, alpha)
        step = np.hypot(*(q - p))
        if step > 3 * h or step < 1e-10:
            h *= 0.5
            if h < hmin:
                break
            continue
        tprev = t
        p = q
        s += step
        pts.append(p.copy())
        ss.append(s)
        h = min(1.1 * h, hmax)
        if s > smax or p[0] > l1max or p[0] < 0.05 or p[1] < 0.02:
            break
    return np.array(ss), np.array(pts)


def _trace_dir(mat, alpha, p0, sign, h0=0.02, hmax=0.06, hmin=1e-5,
               smax=60.0, lmax=400.0, lmin=0.02):
    """One-way continuation from p0; `sign` selects the initial tangent sense."""
    p = _project(mat, p0, alpha)
    pts, ss = [p.copy()], [0.0]
    h, s, tprev = h0, 0.0, None
    for _ in range(20000):
        f = fields(mat, p[0], p[1], alpha)
        t = np.array([f["N1"], f["N2"]])
        nt = np.hypot(*t)
        if nt < 1e-13:
            break
        t /= nt
        if tprev is None:
            t = sign * t
        elif t @ tprev < 0:
            t = -t
        q = _project(mat, p + h * t, alpha)
        step = np.hypot(*(q - p))
        if step > 3 * h or step < 1e-10:
            h *= 0.5
            if h < hmin:
                break
            continue
        tprev, p = t, q
        s += step
        pts.append(p.copy())
        ss.append(s)
        h = min(1.1 * h, hmax)
        if s > smax or not (lmin < p[0] < lmax and lmin < p[1] < lmax):
            break
    return np.array(ss), np.array(pts)


def trace_from(mat, alpha, p0, **kw):
    """Both-ways continuation of g=0 through p0, returned ordered in lambda_1."""
    sa, pa = _trace_dir(mat, alpha, p0, +1.0, **kw)
    sb, pb = _trace_dir(mat, alpha, p0, -1.0, **kw)
    pts = np.vstack([pb[::-1], pa[1:]])
    ss = np.concatenate([-sb[::-1], sa[1:]])
    return ss - ss.min(), pts


def _refine(mat, alpha, pa, pb, key, iters=60):
    """Bisect a sign change of `key` along the curve, reprojecting."""
    a, b = np.array(pa, float), np.array(pb, float)
    fa = fields(mat, a[0], a[1], alpha)[key]
    for _ in range(iters):
        m = _project(mat, 0.5 * (a + b), alpha)
        fm = fields(mat, m[0], m[1], alpha)[key]
        if abs(fm) < 1e-11 or np.hypot(*(b - a)) < 1e-9:
            return m
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def reversals(mat: Material, alpha, **kw):
    """States on the accessible branch where N2 changes sign with det H > 0."""
    ss, pts = trace(mat, alpha, **kw)
    out = []
    prev = None
    for k, p in enumerate(pts):
        f = fields(mat, p[0], p[1], alpha)
        if prev is not None and prev[0] * f["N2"] < 0 and prev[1] > 0 and f["detH"] > 0:
            if np.hypot(p[0] - 1.0, p[1] - 1.0) > 1e-3:
                q = _refine(mat, alpha, pts[k - 1], p, "N2")
                # arclength of the refined reversal (ss[k] belongs to the first
                # trace point past it), so that the branch minimum stops exactly
                # at the reversal.
                s_q = ss[k - 1] + float(np.hypot(*(np.asarray(q) - pts[k - 1])))
                out.append((s_q, q))
        prev = (f["N2"], f["detH"])
    return out, ss, pts


# ---------------------------------------------------------------------------
# Table 1: the twelve tabulated reversal states.
# ---------------------------------------------------------------------------
TABLE_CASES = [("neo", sp.Rational(1, 3)), ("gent", sp.Rational(1, 3)),
               ("yeoh", sp.Rational(1, 3)), ("constant", 0.70), ("growing", 0.75),
               ("decaying", 0.75), ("mooney13", 0.70), ("mooney1000", 0.90),
               ("mooney02", 0.30), ("gt", 0.55)]


@dataclass
class Row:
    key: str
    label: str
    alpha: float
    s: float
    l1: float
    l2: float
    n: int
    dead: float
    tmin: float
    se: float
    se_bind: str
    haug: float
    haug_root: bool
    se_branch: float
    se_branch_s: float
    haug_branch: float
    haug_branch_s: float


def branch_min(mat, alpha, ss, pts, s_star, fn, q=None, nsample=2000):
    """min of fn over the branch from (1,1) to the reversal at arclength s_star
    (closing exactly on the refined reversal state q when given), and where."""
    m = ss < s_star
    if m.sum() < 1:
        return float("nan"), float("nan")
    s_ = ss[m]
    x_, y_ = pts[m, 0], pts[m, 1]
    if q is not None:
        s_ = np.append(s_, s_star)
        x_ = np.append(x_, q[0])
        y_ = np.append(y_, q[1])
    si = np.linspace(0.0, s_star, nsample)
    x = np.interp(si, s_, x_)
    y = np.interp(si, s_, y_)
    best, at = math.inf, 0.0
    for a, b, c in zip(si, x, y):
        v = fn(mat, b, c)
        if v < best:
            best, at = v, a
    return best, at


def build_rows(verbose=True):
    rows = []
    for key, al in TABLE_CASES:
        alpha = float(al)
        mat = Material(key)
        revs, ss, pts = reversals(mat, alpha)
        for n, (s_star, q) in enumerate(revs, 1):
            l1, l2 = float(q[0]), float(q[1])
            se, bind = se_margin(mat, l1, l2)
            hmin, hroot, _ = haughton_margin(mat, l1, l2)
            t1, t2 = cauchy(mat, l1, l2)
            sb, sb_s = branch_min(mat, alpha, ss, pts, s_star,
                                  lambda m, a, b: se_margin(m, a, b)[0], q=q)
            hb, hb_s = branch_min(mat, alpha, ss, pts, s_star,
                                  lambda m, a, b: haughton_margin(m, a, b)[0], q=q)
            rows.append(Row(key, mat.label, alpha, s_star, l1, l2, n,
                            dead_load_margin(mat, l1, l2, alpha),
                            min(t1, t2) / mat.mu0,
                            se, bind, hmin, hroot, sb, sb_s, hb, hb_s))
            if verbose:
                print(f"  {key:<11} a={alpha:.3f} rev{n} ({l1:.4f},{l2:.4f}) "
                      f"detH/Psi11^2={rows[-1].dead:.4f} SE={se:.4f} [{bind}] "
                      f"Haughton={hmin:.4g} root={hroot}")
    return rows


def emit_latex(rows, path):
    """Table 1.  The last two columns are minima over the branch from the
    reference state to the reversal (N6), not values at the reversal."""
    out = [r"\begin{tabular}{lcccccc}", r"\hline",
           r"material & $\alpha$ & $(\lambda_1,\lambda_2)$ at reversal &",
           r"$\dfrac{\Psi_{22}}{\Psi_{11}}-\alpha^2$ & $\min(t_1,t_2)/\mu_0$ &",
           r"$\mathrm{SE}/\mu_0$ & $\min_\gamma\mathcal{I}/\mu_0^2$ \\",
           r"\hline"]
    for r in rows:
        name = r"\quad (second reversal)" if r.n > 1 else r.label
        astr = "$1/3$" if abs(r.alpha - 1 / 3) < 1e-9 else f"${r.alpha:.2f}$"
        t = f"{r.tmin:.4f}"
        h = f"{r.haug_branch:.4f}"
        out.append(f"{name} & {astr} & $({r.l1:.4f},\\,{r.l2:.4f})$ & "
                   f"${r.dead:.4f}$ & ${t}$ & ${r.se_branch:.4f}$ & ${h}$ \\\\")
    out += [r"\hline", r"\end{tabular}"]
    open(path, "w").write("\n".join(out) + "\n")
    return path


def emit_csv(rows, path):
    hdr = ("material,alpha,lambda1,lambda2,reversal_index,arclength,"
           "detH_over_Psi11sq,min_t_over_mu0,SE_over_mu0,SE_binding,haughton_min_over_mu0sq,"
           "haughton_has_root,SE_branch_min,SE_branch_argmin,"
           "haughton_branch_min,haughton_branch_argmin")
    L = [hdr]
    for r in rows:
        L.append(f"{r.key},{r.alpha:.6f},{r.l1:.6f},{r.l2:.6f},{r.n},{r.s:.6f},"
                 f"{r.dead:.6f},{r.tmin:.6f},{r.se:.6f},\"{r.se_bind}\","
                 f"{r.haug:.6e},{r.haug_root},{r.se_branch:.6f},{r.se_branch_s:.6f},"
                 f"{r.haug_branch:.6e},{r.haug_branch_s:.6f}")
    open(path, "w").write("\n".join(L) + "\n")
    return path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def selftest(seed=0, nrand=2000):
    ok = True
    rng = np.random.default_rng(seed)

    print("N1  moduli vs SymPy differentiation (2000 random states, [0.5,5]^2)")
    l1s, l2s, l3s = sp.symbols("l1 l2 l3", positive=True)
    for key in ("neo", "mooney13", "gt", "constant", "yeoh", "gent"):
        mat = Material(key)
        W3 = mat._sym[0]
        f = sp.lambdify((l1s, l2s, l3s),
                        [[sp.diff(W3, a) for a in (l1s, l2s, l3s)],
                         [[sp.diff(W3, a, b) for b in (l1s, l2s, l3s)]
                          for a in (l1s, l2s, l3s)]], "numpy")
        worst = 0.0
        for _ in range(nrand // 6):
            a, b = rng.uniform(0.5, 5.0, 2)
            lam = np.array([a, b, 1 / (a * b)])
            W, Wd = f(*lam)
            W = np.array(W, float); Wd = np.array(Wd, float)
            A4ref = np.array([[lam[i] * lam[j] * Wd[i][j] for j in range(3)]
                              for i in range(3)])
            A4, Aa, Ab, _ = mat.moduli(a, b)
            den = max(1.0, np.abs(A4ref).max())
            worst = max(worst, np.abs(A4 - A4ref).max() / den)
            for i in range(3):
                for j in range(3):
                    if i != j:
                        assert abs(Ab[i, j] - (Aa[i, j] - lam[i] * W[i])) < 1e-8 * den
        print(f"     {key:<11} max relative discrepancy = {worst:.3e}")
        ok &= worst < 1e-10

    print("N2  principal-plane SE vs Legendre-Hadamard over the whole unit sphere")
    for key in ("neo", "mooney1000", "constant", "gt", "yeoh", "growing", "decaying"):
        mat = Material(key)
        a, b = rng.uniform(0.7, 3.0, 2)
        cf = se_margin(mat, a, b)[0]
        bf = se_brute(mat, a, b)
        gap = bf - cf
        print(f"     {key:<11} lam=({a:.3f},{b:.3f})  principal={cf:.6f} "
              f"sphere={bf:.6f}  gap={gap:.2e}")
        ok &= (gap >= -1e-9) and (gap < 1e-3)

    print("N4/N5  Haughton (23) vs Ogden (4.10) at alpha = 1 (Ogden's hypothesis)")
    for key in ("neo", "mooney13", "mooney1000", "gt"):
        mat = Material(key)
        # a symmetric-loading state: Psi_1 = Psi_2 requires lambda_1 = lambda_2
        lam = float(rng.uniform(1.05, 2.0))
        g2 = np.geomspace(1e-3, 1e3, 400)
        h = haughton_I(mat, lam, lam, g2)
        o = ogden_secular(mat, lam, lam, g2)
        # compare root sets via sign patterns
        sh = np.sign(h); so = np.sign(o)
        agree = np.array_equal(sh > 0, so > 0) or np.array_equal(sh > 0, so < 0)
        print(f"     {key:<11} lam={lam:.4f}  sign patterns agree = {agree}")
        ok &= agree

    print("N7  reversal-locus ground truth (Mooney c2=1000, alpha=0.9)")
    mat = Material("mooney1000")
    exp = {0.30: 0.05509, 1.00: 0.08861, 1.50: 3.14455, 5.00: 11.11059}
    for x1, want in exp.items():
        from scipy.optimize import brentq
        fn = lambda y: fields(mat, x1, y, 0.9)["N2"]
        lo, hi = want * 0.9, want * 1.1
        got = brentq(fn, lo, hi, xtol=1e-13)
        print(f"     lambda_1={x1:<6.2f} N2=0 at {got:.5f}  (expected {want:.5f})")
        ok &= abs(got - want) < 1e-5
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if a.table:
        rows = build_rows()
        print("\n", emit_latex(rows, f"{a.outdir}/table1_margins.tex"))
        print(emit_csv(rows, f"{a.outdir}/table1_margins.csv"))
    if not (a.selftest or a.table):
        ap.print_help()


if __name__ == "__main__":
    main()
