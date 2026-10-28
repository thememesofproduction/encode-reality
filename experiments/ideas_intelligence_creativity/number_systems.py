"""SPEC.md section 3: polynomials and two-dimensional number systems.

An algebra ``Algebra(alpha, beta)`` is the real plane with basis 1, e, where e*e = alpha + beta*e. The
point ``z = (a, b)`` means ``a + b*e``. Real polynomials are evaluated in the algebra with its own
arithmetic (``poly``), and the residual ``rho(z)`` is the spectral radius of "multiply by p(z)".

The residual's formula is chosen by the algebra's kind (spec 9.1, 9.12a):
- dual: |p(a + b*beta/2)|, because e has the double eigenvalue beta/2;
- split: max |p(a + b*r)| over the two real roots r of t^2 - beta*t - alpha;
- complex: |p(phi(z))|, where phi is the isomorphism onto the complex numbers.
"""

import math

import numpy as np

COEFFS = (4.0, -8.0, -7.0, 20.0, 25.0)   # highest degree first: 4x^4 - 8x^3 - 7x^2 + 20x + 25

DUAL_TOL = 1e-12                          # |delta| <= DUAL_TOL * max(1, beta^2, |4 alpha|) is dual
_IMAG_TOL = 1e-9                          # a critical point with |imag| below this is real


def _horner(coeffs, t):
    """p(t) by Horner's rule, the operations numpy.polyval uses. t may be real, complex or an array."""
    acc = 0.0
    for c in coeffs:
        acc = acc * t + c
    return acc


def p_real(x):
    """p on the real line, vectorised: numpy.polyval(COEFFS, x)."""
    return np.polyval(COEFFS, x)


def real_critical_points(coeffs=COEFFS) -> list:
    """The real critical points of p (real roots of p'), each Newton-polished, sorted."""
    c = np.asarray(coeffs, dtype=float)
    d1 = np.polyder(c)
    d2 = np.polyder(d1)
    points = []
    for z in np.roots(d1):
        if abs(z.imag) >= _IMAG_TOL:
            continue
        x = float(z.real)
        for _ in range(100):
            h = np.polyval(d2, x)
            if h == 0.0:
                break
            step = np.polyval(d1, x) / h
            x -= step
            if abs(step) <= 1e-16 * max(1.0, abs(x)):
                break
        points.append(float(x))
    return sorted(points)


def real_floor(coeffs=COEFFS):
    """(x_star, m): the minimum over the reals of |p(x)|, from the real critical points of p.

    Every minimum of |p| on the real line is at a critical point of p when p has no real root, so the
    smallest |p| over those points is the floor.
    """
    c = np.asarray(coeffs, dtype=float)
    best_x, best_v = None, None
    for x in real_critical_points(c):
        v = abs(float(np.polyval(c, x)))
        if best_v is None or v < best_v:
            best_x, best_v = x, v
    if best_x is None:
        raise ValueError("p has no real critical point")
    return float(best_x), float(best_v)


def complex_roots(coeffs=COEFFS) -> np.ndarray:
    """All complex roots of p, as a complex array."""
    return np.asarray(np.roots(np.asarray(coeffs, dtype=float)), dtype=complex)


class Algebra:
    """The two-dimensional algebra with e*e = alpha + beta*e. See the module docstring."""

    def __init__(self, alpha: float, beta: float):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = self.beta * self.beta + 4.0 * self.alpha
        scale = max(1.0, self.beta * self.beta, abs(4.0 * self.alpha))
        if abs(self.delta) <= DUAL_TOL * scale:
            self.kind = "dual"
        elif self.delta < 0:
            self.kind = "complex"
        else:
            self.kind = "split"
        self._half = self.beta / 2.0                                  # Re(e) in the complex picture
        self._imag = math.sqrt(-self.delta) / 2.0 if self.kind == "complex" else None
        if self.kind == "split":
            s = math.sqrt(self.delta)
            self._roots = ((self.beta + s) / 2.0, (self.beta - s) / 2.0)   # the two real roots of t^2 - beta t - alpha
        else:
            self._roots = None

    def mul(self, z, w):
        """(a + b e)(c + d e) = (a c + b d alpha) + (a d + b c + b d beta) e, as (a', b')."""
        a, b = z
        c, d = w
        return (a * c + b * d * self.alpha, a * d + b * c + b * d * self.beta)

    def poly(self, z, coeffs=COEFFS):
        """p(z) computed with this algebra's arithmetic (Horner), returned as (A, B)."""
        A, B = 0.0, 0.0
        for c in coeffs:
            A, B = self.mul((A, B), z)
            A = A + c
        return A, B

    def rho(self, z, coeffs=COEFFS):
        """The residual: the spectral radius of multiplication by p(z). See the module docstring."""
        a, b = z
        if self.kind == "dual":
            return abs(_horner(coeffs, a + b * self._half))
        if self.kind == "split":
            r1, r2 = self._roots
            u = abs(_horner(coeffs, a + b * r1))
            v = abs(_horner(coeffs, a + b * r2))
            if isinstance(u, float) and isinstance(v, float):
                return max(u, v)
            return np.maximum(u, v)
        return abs(_horner(coeffs, self.phi(z)))

    def coord_norm(self, z, coeffs=COEFFS):
        """sqrt(A^2 + B^2) of p(z) in the basis (1, e). REJECTED as a residual (spec 3): it reports fake
        successes in dual algebras. Kept so the tests can show why."""
        A, B = self.poly(z, coeffs)
        return np.hypot(A, B)

    def phi(self, z):
        """The isomorphism to the complex numbers, for complex algebras: a + b e maps to
        (a + b beta/2) + i b sqrt(-delta)/2. A Python complex for scalars, a complex array otherwise."""
        if self.kind != "complex":
            raise ValueError(f"phi is an isomorphism onto C only for complex algebras, not {self.kind}")
        a, b = z
        return (a + b * self._half) + 1j * (b * self._imag)

    def roots(self, coeffs=COEFFS) -> list:
        """Every z = (a, b) with p(z) = 0, found by inverting phi on each complex root s + it of p.
        Empty unless the algebra is complex."""
        if self.kind != "complex":
            return []
        out = []
        for w in complex_roots(coeffs):
            b = float(w.imag) / self._imag
            a = float(w.real) - b * self._half
            out.append((a, b))
        return out
