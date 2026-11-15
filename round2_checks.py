"""Checks added in the second revision (RSPA-2026-0707.R2).

    python round2_checks.py

1. Large-strain states of the marginal energy W = (I1-3) + 2c(sqrt(I2)-sqrt(3)),
   for which W1 = 1 and W2 = c/sqrt(I2) exactly, so gamma_inf = c:
     c < alpha : spreading state, lambda2/lambda1 -> v_inf      (eq. marginal_ratio)
     c > alpha : lambda2 ~ kappa lambda1^(-1/2), kappa from     (eq. kappa)
     c = alpha : lambda2 ~ A lambda1^(-1/5) (degenerate threshold)
   together with det H > 0 along each branch from the natural state.
2. Class W(I2), the three energies of Fig. 2(b): first criticality on the
   universal trajectory, and monotonicity of the load.
3. Nondegeneracy of the folds reported in the paper: the coefficient
   D3Psi[n,n,n], n = (r,-1), and P'' = D3Psi[n,n,n]/(r - alpha) at each fold.
4. Mooney-Rivlin fold threshold at alpha = 0.90, where the two limit points
   of item 3 coalesce and P'' vanishes (degenerate case of Sect. 7).
All derivatives are exact (SymPy); branches are traced by continuation in
lambda1 with Brent's method, as in make_figures.py. Run time: about 2 min.
"""
import numpy as np
import sympy as sp
from scipy.optimize import brentq, minimize_scalar

l1, l2 = sp.symbols("l1 l2", positive=True)
I1 = l1**2 + l2**2 + 1/(l1**2*l2**2)
I2 = 1/l1**2 + 1/l2**2 + l1**2*l2**2
X = (l1, l2)


def build(W):
    P1, P2 = sp.diff(W, l1), sp.diff(W, l2)
    H = sp.Matrix([[sp.diff(W, a, b) for b in X] for a in X])
    f = lambda e: sp.lambdify(X, e, "numpy")
    D3 = [[[f(sp.diff(W, a, b, c)) for c in X] for b in X] for a in X]
    return dict(P1=f(P1), P2=f(P2), H11=f(H[0, 0]), H12=f(H[0, 1]),
                H22=f(H[1, 1]), D3=D3)


def trace(m, alpha, l1max, n=4000):
    """Branch from the natural state, by continuation in lambda1."""
    rows, g = [], 1.0
    for x in np.geomspace(1 + 1e-6, l1max, n):
        fun = lambda y: m["P2"](x, y) - alpha*m["P1"](x, y)
        lo, hi = g*0.8, g*1.25
        for _ in range(200):
            if fun(lo)*fun(hi) <= 0:
                break
            lo, hi = lo*0.9, hi*1.1
        else:
            break
        y = brentq(fun, lo, hi, xtol=1e-15, rtol=1e-14)
        g = y
        a, b, c = m["H11"](x, y), m["H12"](x, y), m["H22"](x, y)
        rows.append((x, y, m["P1"](x, y), b/a, (a*c - b*b)/a**2))
    return np.array(rows)


def fold_coefficient(m, x, y, alpha):
    a, b = m["H11"](x, y), m["H12"](x, y)
    r = b/a
    nvec = (r, -1.0)
    d3 = sum(m["D3"][i][j][k](x, y)*nvec[i]*nvec[j]*nvec[k]
             for i in range(2) for j in range(2) for k in range(2))
    return r, d3, d3/(r - alpha)


def locate_folds(m, arr, alpha):
    """Refine every sign change of det H along a traced branch."""
    out = []
    d = arr[:, 4]
    for i in np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
        def detx(x):
            fun = lambda y: m["P2"](x, y) - alpha*m["P1"](x, y)
            y = brentq(fun, arr[i, 1]*0.9, arr[i + 1, 1]*1.1, xtol=1e-15)
            a, b, c = m["H11"](x, y), m["H12"](x, y), m["H22"](x, y)
            return a*c - b*b, y
        x = brentq(lambda s: detx(s)[0], arr[i, 0], arr[i + 1, 0], xtol=1e-14)
        y = detx(x)[1]
        out.append((x, y, m["P1"](x, y)) + fold_coefficient(m, x, y, alpha))
    return out


def kappa(q):
    """Root kappa > 1 of (kappa^4 - 1)/(kappa^2 sqrt(kappa^4 + 1)) = q, 0 < q < 1."""
    return ((2 + q*q + q*np.sqrt(q*q + 8))/(2*(1 - q*q)))**0.25


print("1. Marginal energy W = (I1-3) + 2c(sqrt(I2)-sqrt(3)), alpha = 0.55")
alpha = 0.55
for c in (0.3, 0.5, 0.55, 0.6, 0.8, 1.5):
    m = build((I1 - 3) + 2*c*(sp.sqrt(I2) - sp.sqrt(3)))
    arr = trace(m, alpha, 1e6)
    x, y = arr[-1, 0], arr[-1, 1]
    stable = bool((arr[:, 4] > 0).all())
    line = f"   c={c:4.2f}: lambda1_end={x:.1e}, min detH/H11^2={arr[:, 4].min():.3e} (stable: {stable}); "
    if c < alpha:
        line += f"l2/l1={y/x:.5f}, predicted v_inf={(alpha - c)/(1 - alpha*c):.5f}"
    elif c > alpha:
        k = kappa(alpha/c)
        line += (f"l2*sqrt(l1)={y*np.sqrt(x):.5f}, predicted kappa={k:.5f}; "
                 f"l2/l3={y*y*x:.4f}, kappa^2={k*k:.4f}")
    else:
        sel = arr[:, 0] > 1e4
        slope = np.polyfit(np.log(arr[sel, 0]), np.log(arr[sel, 1]), 1)[0]
        A = (3*alpha/(2*(1 - alpha**2)))**0.2
        line += (f"log-slope of l2 over l1>1e4: {slope:.4f} (predicted -0.2); "
                 f"l2*l1^(1/5)={y*x**0.2:.5f}, predicted A={A:.5f}")
    print(line)

print("\n2. Class W(I2), energies of Fig. 2(b), on the universal trajectory")
J = I2 - 3
mats = {"constant W2": J, "growing W2": J + sp.Rational(1, 10)*J**2,
        "decaying W2": 20*sp.log(1 + J/20)}
for name, W in mats.items():
    m = build(W)
    for a in (0.60, 0.75, 0.88):
        v = np.linspace(1 - 1e-6, 2e-4, 200000)
        s = (1 - a*v**3)/(v*(1 - a*v))
        u = s**(1/3)
        x, y = np.sqrt(u/v), np.sqrt(u*v)
        det = m["H11"](x, y)*m["H22"](x, y) - m["H12"](x, y)**2
        P = m["P1"](x, y)
        i = int(np.argmax(det <= 0)) if (det <= 0).any() else None
        where = f"lambda1={x[i]:.3f}" if i is not None else "none"
        print(f"   {name:12s} alpha={a:.2f}: first det H<=0 at {where}; "
              f"P monotone: {bool(np.all(np.diff(P) > 0))}; up to lambda1={x[-1]:.0f}")

print("\n3. Nondegeneracy of the reported folds, P'' = D3Psi[n,n,n]/(r-alpha)")
cases = [("W = c2(I2-3), alpha=0.70", J, 0.70, 10.0, 1.0),
         ("MR c2/c1=1000, alpha=0.90", (I1 - 3) + 1000*(I2 - 3), 0.90, 100.0, 1001.0)]
folds = {}
for label, W, a, lmax, scale in cases:
    m = build(W)
    arr = trace(m, a, lmax, n=20000)
    folds[label] = locate_folds(m, arr, a)
    for (x, y, P, r, d3, Ppp) in folds[label]:
        print(f"   {label}: fold at ({x:.4f},{y:.4f}), P/scale={P/scale:.4f}, r={r:.4f}, "
              f"D3={d3:.4g}, P''={Ppp:.4g} ({'maximum' if Ppp < 0 else 'minimum'})")

print("\n4. Mooney-Rivlin fold threshold at alpha = 0.90 (cusp)")
# The threshold is the smallest c2/c1 (c1 = 1) at which det H, positive along
# the branch from the natural state, first touches zero. The minimum of
# det H/H11^2 along the branch is refined by a bounded scalar minimiser, and
# c2 is bisected until that minimum vanishes. At the touching point the two
# limit points coalesce, so the fold coefficient must vanish: dP/dlambda1 =
# det H/N1 has a double zero there.
a = 0.90
C2 = sp.Symbol("c2", positive=True)
WMR = (I1 - 3) + C2*(I2 - 3)
g = lambda e: sp.lambdify((l1, l2, C2), e, "numpy")
MR = dict(P1=g(sp.diff(WMR, l1)), P2=g(sp.diff(WMR, l2)),
          H11=g(sp.diff(WMR, l1, 2)), H12=g(sp.diff(WMR, l1, l2)),
          H22=g(sp.diff(WMR, l2, 2)),
          D3=[[[g(sp.diff(WMR, p, q, s)) for s in X] for q in X] for p in X])


def mr_at(c2):
    """The dictionary of build() for Mooney-Rivlin at fixed c2."""
    fix = lambda f: (lambda x, y: f(x, y, c2))
    m = {key: fix(MR[key]) for key in ("P1", "P2", "H11", "H12", "H22")}
    m["D3"] = [[[fix(f) for f in row] for row in mat] for mat in MR["D3"]]
    return m


def min_det(c2):
    """Refined minimum of det H/H11^2 along the branch, and where it occurs."""
    m = mr_at(c2)
    arr = trace(m, a, 100.0, n=6000)
    k = int(np.argmin(arr[:, 4]))
    lo_x, hi_x = arr[max(k - 1, 0), 0], arr[min(k + 1, len(arr) - 1), 0]

    def lam2(x):
        fun = lambda y: m["P2"](x, y) - a*m["P1"](x, y)
        return brentq(fun, arr[k, 1]*0.9, arr[k, 1]*1.1, xtol=1e-15)

    def dnorm(x):
        y = lam2(x)
        h11, h12, h22 = m["H11"](x, y), m["H12"](x, y), m["H22"](x, y)
        return (h11*h22 - h12*h12)/h11**2

    res = minimize_scalar(dnorm, bounds=(lo_x, hi_x), method="bounded",
                          options={"xatol": 1e-12})
    return res.fun, res.x, lam2(res.x), m


lo, hi = 30.0, 60.0
for _ in range(40):
    mid = 0.5*(lo + hi)
    if min_det(mid)[0] > 0:
        lo = mid
    else:
        hi = mid
c2s = 0.5*(lo + hi)
dmin, x, y, m = min_det(c2s)
r, d3, Ppp = fold_coefficient(m, x, y, a)
Ppp_ref = folds["MR c2/c1=1000, alpha=0.90"][0][5]
print(f"   threshold c2/c1 = {c2s:.4f}; at the touching point ({x:.4f},{y:.4f}), "
      f"P/(c1+c2)={m['P1'](x, y)/(1 + c2s):.4f}, r={r:.4f}: "
      f"P''/(c1+c2)={Ppp/(1 + c2s):.2e} "
      f"(c2/c1=1000 upper fold: P''/(c1+c2)={Ppp_ref/1001:.4g})")
