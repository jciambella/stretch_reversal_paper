#!/usr/bin/env python3
r"""
exact_elimination.py -- exact separation of the reversal and criticality loci
(J. Ciambella, Proc. R. Soc. A; Sect. 7 and ESM Sect. 2).

Reversal and criticality, with alpha eliminated between them, are the two
plane curves

    g1 = Psi_11 Psi_22 - Psi_12^2          (criticality: det H = 0)
    g2 = Psi_12 Psi_1 - Psi_11 Psi_2       (reversal:  r = alpha = Psi_2/Psi_1)

as in the coincidence system of Sect. 7 of the paper and the ESM.

in x = lambda_1^2, y = lambda_2^2.  The claim of Sect. 7 is that their only
common zero in the open positive quadrant is the equibiaxial Kearsley state
(Mooney--Rivlin) or none (Gent--Thomas).  The resultant Res_y is exact
(rational arithmetic); the lifting of its real roots is done in 100-digit
arithmetic and reports, for every root x*, where the common root y* of
g1(x*,.) and g2(x*,.) lies.

Run:  python3 exact_elimination.py [--material KEY ...]
"""
from __future__ import annotations

import argparse
import time

import sympy as sp

l1, l2 = sp.symbols("l1 l2", positive=True)
_i1, _i2 = sp.symbols("i1 i2", positive=True)

I1 = l1**2 + l2**2 + 1 / (l1**2 * l2**2)
I2 = l1**2 * l2**2 + 1 / l1**2 + 1 / l2**2

CASES = {
    "mooney_1_3":  (_i1 - 3) + sp.Rational(1, 3) * (_i2 - 3),
    "mooney_1":    (_i1 - 3) + (_i2 - 3),
    "mooney_1000": (_i1 - 3) + 1000 * (_i2 - 3),
    "gent_thomas": (_i1 - 3) + sp.log(_i2 / 3),
}


def loci(Wi):
    """g1, g2 as numerators of the two loci, cleared of denominators."""
    Psi = Wi.subs({_i1: I1, _i2: I2})
    P1 = sp.diff(Psi, l1)
    P2 = sp.diff(Psi, l2)
    P11 = sp.diff(Psi, l1, 2)
    P12 = sp.diff(Psi, l1, l2)
    P22 = sp.diff(Psi, l2, 2)
    g1 = sp.together(sp.simplify(P11 * P22 - P12**2))
    g2 = sp.together(sp.simplify(P12 * P1 - P11 * P2))
    return sp.numer(g1), sp.numer(g2)


def run(key, verbose=True):
    t0 = time.time()
    Wi = CASES[key]
    n1, n2 = loci(Wi)
    x, y = sp.symbols("x y", positive=True)
    # substitute lambda_i = sqrt(x), sqrt(y): every exponent is even
    sub = {l1: sp.sqrt(x), l2: sp.sqrt(y)}
    p1 = sp.numer(sp.together(sp.expand(sp.powsimp(n1.subs(sub), force=True))))
    p2 = sp.numer(sp.together(sp.expand(sp.powsimp(n2.subs(sub), force=True))))
    p1 = sp.Poly(sp.expand(p1), y)
    p2 = sp.Poly(sp.expand(p2), y)
    if verbose:
        print(f"[{key}] deg_y g1 = {p1.degree()}, deg_y g2 = {p2.degree()}  "
              f"({time.time()-t0:.1f}s)")
    R = sp.resultant(p1.as_expr(), p2.as_expr(), y)
    R = sp.Poly(sp.expand(R), x)
    if verbose:
        print(f"[{key}] Res_y(g1,g2): degree {R.degree()} in x = lambda_1^2 "
              f"(i.e. {2*R.degree()} in lambda_1)  ({time.time()-t0:.1f}s)")
    fac = sp.factor_list(R.as_expr())
    roots = []
    for f, m in fac[1]:
        pf = sp.Poly(f, x)
        if pf.degree() < 1:
            continue
        for r in sp.real_roots(pf):
            val = float(r.evalf(30))
            if val > 0:
                roots.append((val, r))          # keep the exact root object
    roots.sort(key=lambda t: t[0])
    if verbose:
        print(f"[{key}] positive real roots of the resultant: "
              f"{[round(v,6) for v,_ in roots]}")
    # Lifting test.  At each real root x* of Res_y the two univariate
    # polynomials g1(x*,.) and g2(x*,.) share at least one root y*, real or
    # complex (Res_y vanishes exactly there, leading coefficients permitting).
    # The shared root is located among all roots of g2(x*,.) by a
    # scale-invariant residual,
    #     rho(y) = |g1(x*,y)| / sum_k |c_k| |y|^k ,
    # which, unlike division by max|c_k|, does not depend on the magnitude of
    # y or on the coefficient spread.  The candidate is physical only if the
    # shared root is real and positive (y = lambda_2^2 > 0); otherwise the
    # resultant root is reported together with the location of its common
    # root, which shows it to be non-physical.
    import mpmath as mp
    mp.mp.dps = 100
    lifted = []
    for val, xexact in roots:
        xhi = xexact.evalf(120)
        c1 = [mp.mpf(str(sp.N(c, 110))) for c in sp.Poly(p1.as_expr().subs(x, xhi), y).all_coeffs()]
        c2 = [mp.mpf(str(sp.N(c, 110))) for c in sp.Poly(p2.as_expr().subs(x, xhi), y).all_coeffs()]
        try:
            yr = mp.polyroots(c2, maxsteps=400, extraprec=600)
        except Exception:                                  # noqa: BLE001
            print(f"        x={val:.6f}: root finding failed")
            continue

        def rho(z):
            num = abs(mp.polyval(c1, z))
            den = sum(abs(c) * abs(z)**(len(c1) - 1 - k) for k, c in enumerate(c1))
            return num / den if den else mp.inf

        # shared roots may be double (the diagonal Kearsley root is a tangency),
        # hence accurate only to ~sqrt(eps) = 1e-50; unshared residuals are O(1)
        tol = mp.mpf("1e-35")
        shared = []
        for z in (z for z in yr if rho(z) < tol):       # merge a double root
            if all(abs(z - w) > mp.mpf("1e-30") * max(1, abs(w)) for w in shared):
                shared.append(z)
        if not shared:
            if verbose:
                print(f"        x={val:.6f}: NO shared root found -- inspect "
                      f"(min rho = {mp.nstr(min(rho(z) for z in yr), 3)})")
            continue
        phys = [z for z in shared
                if abs(mp.im(z)) < tol * max(1, abs(z)) and mp.re(z) > 0]
        for z in phys:
            lifted.append((val, float(mp.re(z))))
        if verbose:
            desc = ", ".join(mp.nstr(mp.re(z), 7) if abs(mp.im(z)) < tol * max(1, abs(z))
                             else mp.nstr(z, 5) for z in shared)
            print(f"        x={val:.6f} (lambda_1={val**0.5:.6f}): shared root(s) y* = {desc}"
                  f"  -> {'PHYSICAL (y*>0)' if phys else 'none in the open quadrant'}")
    print(f"[{key}] common zeros in the open positive quadrant: "
          f"{lifted if lifted else 'NONE -> loci are disjoint'}  "
          f"({time.time()-t0:.1f}s total)\n")
    return R.degree(), roots, lifted


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", nargs="*", default=list(CASES))
    a = ap.parse_args()
    for k in a.material:
        try:
            run(k)
        except Exception as exc:                       # noqa: BLE001
            print(f"[{k}] FAILED: {type(exc).__name__}: {exc}\n")
