"""SPEC.md section 3: number_systems.py -- polynomials and two-dimensional number systems.

References used here are independent of the code under test: the multiplication is the spec's
definition (a + b e)(c + d e) with e*e = alpha + beta*e, and rho is evaluated as max |p(a + b r)|
over the two roots r of t^2 - beta t - alpha (u + v r = p(a + b r), because z -> a + b r is a ring map).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (  # noqa: E402
    dual_kind, load, spec_float, spec_literal, true_floor,
)

COEFFS = spec_literal("COEFFS", 3)
X_TRUE, M_TRUE = true_floor(COEFFS)
DUAL_TOL = spec_float(r"\|delta\| ≤ (1e-\d+)·max", 3)
FLOOR_TOL = 1e-12          # "rho(z) >= m"; the same 1e-12 margin the spec uses in section 4 contract 3
# Section 3, "The polynomial": the complex roots are -1 +- 0.5i and 2 +- i.
SPEC_ROOTS = [complex(-1, 0.5), complex(-1, -0.5), complex(2, 1), complex(2, -1)]


def ns():
    return load("number_systems")


def mul_ref(alpha, beta, z, w):
    a, b = z
    c, d = w
    return (a * c + b * d * alpha, a * d + b * c + b * d * beta)


def poly_ref(alpha, beta, z, coeffs=COEFFS):
    A = np.zeros(np.broadcast(z[0], z[1]).shape)
    B = np.zeros_like(A)
    for c in coeffs:
        A, B = mul_ref(alpha, beta, (A, B), z)
        A = A + c
    return A, B


def rho_ref(alpha, beta, a, b, coeffs=COEFFS):
    """Spec 9.1: dual kind (delta within the tolerance, either sign) -> |p(a + b beta/2)|;
    delta > 0 -> max |p(a + b r)| over the two real roots; complex -> |p(a + b r)| with r complex."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = beta * beta + 4 * alpha
    if dual_kind(alpha, beta, DUAL_TOL) == "dual":
        return np.abs(np.polyval(coeffs, a + b * beta / 2))
    if d >= 0:
        s = math.sqrt(d)
        return np.maximum(np.abs(np.polyval(coeffs, a + b * (beta + s) / 2)),
                          np.abs(np.polyval(coeffs, a + b * (beta - s) / 2)))
    r = complex(beta / 2, math.sqrt(-d) / 2)
    return np.abs(np.polyval(coeffs, a + b * r))


def random_algebras(rng, n, kind):
    out = []
    while len(out) < n:
        beta = float(rng.uniform(-4, 4))
        if kind == "dual":
            out.append((-beta * beta / 4, beta))           # delta == 0.0 exactly
            continue
        if kind in ("dual_neg", "dual_pos"):               # delta within the dual tolerance, nonzero
            beta = float(rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 4.0))
            sign = -1.0 if kind == "dual_neg" else 1.0
            alpha = (-beta * beta + sign * 0.5 * DUAL_TOL * beta * beta) / 4
            d = beta * beta + 4 * alpha
            if dual_kind(alpha, beta, DUAL_TOL) == "dual" and d != 0 and (d < 0) == (sign < 0):
                out.append((alpha, beta))
            continue
        alpha = float(rng.uniform(-5, 5))
        d = beta * beta + 4 * alpha
        if (kind == "complex" and d <= -0.5) or (kind == "split" and d >= 0.5):
            out.append((alpha, beta))
    return out


def all_kinds(rng, n_each=6):
    return [("complex", a) for a in random_algebras(rng, n_each, "complex")] + \
           [("split", a) for a in random_algebras(rng, n_each, "split")] + \
           [("dual", a) for a in random_algebras(rng, n_each, "dual")] + \
           [("dual", a) for a in random_algebras(rng, 2, "dual_neg")] + \
           [("dual", a) for a in random_algebras(rng, 2, "dual_pos")]


def close(x, y, rtol=1e-12, atol=1e-12):
    np.testing.assert_allclose(np.asarray(x, dtype=float), np.asarray(y, dtype=float), rtol=rtol, atol=atol)


# ----------------------------------------------------------------------------- the polynomial

def test_coeffs_constant():
    assert tuple(ns().COEFFS) == tuple(COEFFS)


def test_p_real_is_vectorised_polyval():
    m = ns()
    rng = np.random.default_rng(1)
    for shape in [(100,), (7, 5)]:
        x = rng.uniform(-6, 6, size=shape)
        got = np.asarray(m.p_real(x))
        assert got.shape == shape
        close(got, np.polyval(COEFFS, x), rtol=1e-13)
    assert float(m.p_real(0.0)) == pytest.approx(25.0, rel=1e-15)


def test_p_real_has_the_stated_second_local_minimum():
    x = np.linspace(1.3, 1.8, 200_001)
    v = np.abs(np.asarray(ns().p_real(x)))
    assert abs(x[np.argmin(v)] - 1.54) < 0.01
    assert v.min() == pytest.approx(32.48, abs=0.01)


def test_real_floor_of_p():
    m = ns()
    x_star, floor = m.real_floor()
    assert x_star == pytest.approx(-0.9215, abs=1e-4)
    assert floor == pytest.approx(9.7702, abs=1e-4)
    grid = np.linspace(-5, 5, 1_000_001)
    grid_min = float(np.abs(np.polyval(COEFFS, grid)).min())
    assert abs(floor - grid_min) <= 1e-9 * grid_min          # "agree with a dense grid minimum to 1e-9"
    assert abs(floor - M_TRUE) <= 1e-9 * M_TRUE
    assert abs(abs(np.polyval(COEFFS, x_star)) - floor) <= 1e-12 * floor


def _random_rootless_quartic(seed):
    rng = np.random.default_rng(seed)
    quads = []
    for _ in range(2):
        p = float(rng.uniform(-4, 4))
        q = p * p / 4 + float(rng.uniform(0.3, 3.0))
        quads.append((p, q))
    k = float(rng.uniform(0.5, 3.0)) * (1 if rng.random() < 0.5 else -1)
    coeffs = k * np.polymul([1.0, quads[0][0], quads[0][1]], [1.0, quads[1][0], quads[1][1]])
    roots = []
    for p, q in quads:
        disc = complex(p * p / 4 - q) ** 0.5
        roots += [-p / 2 + disc, -p / 2 - disc]
    return tuple(float(c) for c in coeffs), roots


PERTURBED = [((1.0, 0.0, 1.0), [1j, -1j]),
             ((2.0, -4.0, 7.0), [complex(1, math.sqrt(2.5)), complex(1, -math.sqrt(2.5))]),
             ((-1.0, 2.0, -3.0), [complex(1, math.sqrt(2)), complex(1, -math.sqrt(2))])] + \
            [_random_rootless_quartic(s) for s in range(5)]


@pytest.mark.parametrize("coeffs,roots", PERTURBED, ids=[f"poly{i}" for i in range(len(PERTURBED))])
def test_real_floor_is_computed_for_other_polynomials(coeffs, roots):
    x_star, floor = ns().real_floor(coeffs=coeffs)
    x_ref, m_ref = true_floor(coeffs)
    assert abs(floor - m_ref) <= 1e-9 * m_ref
    assert abs(abs(np.polyval(coeffs, x_star)) - floor) <= 1e-9 * floor
    grid = np.linspace(-8, 8, 1_600_001)
    assert floor <= float(np.abs(np.polyval(coeffs, grid)).min()) * (1 + 1e-9)


@pytest.mark.parametrize("coeffs,roots", PERTURBED, ids=[f"poly{i}" for i in range(len(PERTURBED))])
def test_complex_roots_for_other_polynomials(coeffs, roots):
    got = np.asarray(ns().complex_roots(coeffs=coeffs), dtype=complex)
    assert got.shape == (len(roots),)
    for r in roots:
        assert np.min(np.abs(got - r)) < 1e-8, (r, got)


def test_complex_roots_of_p():
    got = np.asarray(ns().complex_roots(), dtype=complex)
    assert isinstance(ns().complex_roots(), np.ndarray)
    assert got.shape == (4,)
    for r in SPEC_ROOTS:
        assert np.min(np.abs(got - r)) < 1e-9, (r, got)
    assert np.all(np.abs(got.imag) > 0.1)                       # no real roots


# ----------------------------------------------------------------------------- Algebra basics

def test_attributes_and_delta():
    m = ns()
    rng = np.random.default_rng(3)
    for _ in range(50):
        alpha, beta = float(rng.uniform(-5, 5)), float(rng.uniform(-6, 6))
        A = m.Algebra(alpha, beta)
        assert A.alpha == alpha and A.beta == beta
        assert A.delta == pytest.approx(beta ** 2 + 4 * alpha, rel=1e-12, abs=1e-12 * max(1, beta ** 2))


def test_kind_examples_from_the_spec():
    m = ns()
    assert m.Algebra(-2.25, 3).kind == "dual"
    assert m.Algebra(-2.25, 3.001).kind == "split"
    assert m.Algebra(-2.25, 2.999).kind == "complex"
    assert m.Algebra(-1, 0).kind == "complex"
    assert m.Algebra(1, 0).kind == "split"
    assert m.Algebra(0, 0).kind == "dual"


def test_kind_follows_the_dual_tolerance_rule():
    m = ns()
    rng = np.random.default_rng(4)
    cases = []
    for _ in range(300):
        cases.append((float(rng.uniform(-50, 50)), float(rng.uniform(-20, 20))))
    for _ in range(200):
        beta = float(rng.choice([1, -1]) * 10 ** rng.uniform(-3, 3))
        scale = max(1.0, beta * beta)
        for factor in (0.5, -0.5, 2.0, -2.0, 0.0):        # well inside / well outside the tolerance
            cases.append(((-beta * beta + factor * DUAL_TOL * scale) / 4, beta))
    for alpha, beta in cases:
        assert m.Algebra(alpha, beta).kind == dual_kind(alpha, beta, DUAL_TOL), (alpha, beta)


# ----------------------------------------------------------------------------- multiplication

def test_mul_is_the_defining_formula_and_e_squared():
    m = ns()
    rng = np.random.default_rng(5)
    for kind, (alpha, beta) in all_kinds(rng):
        A = m.Algebra(alpha, beta)
        z = (rng.normal(size=40), rng.normal(size=40))
        w = (rng.normal(size=40), rng.normal(size=40))
        got = A.mul(z, w)
        want = mul_ref(alpha, beta, z, w)
        close(got[0], want[0])
        close(got[1], want[1])
        e2 = A.mul((0.0, 1.0), (0.0, 1.0))
        assert float(e2[0]) == pytest.approx(alpha, rel=1e-14, abs=1e-14)
        assert float(e2[1]) == pytest.approx(beta, rel=1e-14, abs=1e-14)


def test_mul_at_minus_one_zero_is_complex_multiplication():
    A = ns().Algebra(-1.0, 0.0)
    rng = np.random.default_rng(6)
    a, b, c, d = rng.normal(size=(4, 100))
    got = A.mul((a, b), (c, d))
    prod = (a + 1j * b) * (c + 1j * d)
    close(got[0], prod.real)
    close(got[1], prod.imag)


def test_mul_is_commutative_associative_with_identity():
    m = ns()
    rng = np.random.default_rng(7)
    for kind, (alpha, beta) in all_kinds(rng, 8):
        A = m.Algebra(alpha, beta)
        z, w, u = [(rng.normal(size=30), rng.normal(size=30)) for _ in range(3)]
        zw, wz = A.mul(z, w), A.mul(w, z)
        close(zw[0], wz[0])
        close(zw[1], wz[1])
        left, right = A.mul(A.mul(z, w), u), A.mul(z, A.mul(w, u))
        close(left[0], right[0], rtol=1e-10, atol=1e-10)
        close(left[1], right[1], rtol=1e-10, atol=1e-10)
        one = A.mul((1.0, 0.0), z)
        close(one[0], z[0])
        close(one[1], z[1])


def test_components_broadcast_elementwise():
    A = ns().Algebra(0.7, -1.3)
    a = np.linspace(-1, 1, 3).reshape(3, 1)
    d = np.linspace(-2, 2, 4).reshape(1, 4)
    got = A.mul((a, 0.5), (1.5, d))
    want = mul_ref(0.7, -1.3, (a, 0.5), (1.5, d))
    assert np.shape(got[0]) == (3, 4) and np.shape(got[1]) == (3, 4)
    close(got[0], want[0])
    close(got[1], want[1])
    r = A.rho((a, d))
    assert np.shape(r) == (3, 4)
    close(r, rho_ref(0.7, -1.3, a, d), rtol=1e-9)


# ----------------------------------------------------------------------------- poly, phi

def test_poly_uses_the_algebras_own_arithmetic():
    m = ns()
    rng = np.random.default_rng(8)
    for kind, (alpha, beta) in all_kinds(rng):
        A = m.Algebra(alpha, beta)
        z = (rng.uniform(-2, 2, 50), rng.uniform(-2, 2, 50))
        got = A.poly(z)
        want = poly_ref(alpha, beta, z)
        close(got[0], want[0], rtol=1e-11, atol=1e-9)
        close(got[1], want[1], rtol=1e-11, atol=1e-9)
        other = A.poly(z, coeffs=(1.0, 0.0, 1.0))                 # coeffs is honoured
        want2 = poly_ref(alpha, beta, z, (1.0, 0.0, 1.0))
        close(other[0], want2[0], rtol=1e-11, atol=1e-9)
        close(other[1], want2[1], rtol=1e-11, atol=1e-9)
        x = rng.uniform(-3, 3, 20)
        on_line = A.poly((x, np.zeros_like(x)))
        close(on_line[0], np.polyval(COEFFS, x), rtol=1e-13)
        close(on_line[1], 0.0, atol=1e-12)


def _phi_e(A):
    return complex(A.phi((0.0, 1.0)))


def test_phi_is_an_isomorphism_onto_c():
    m = ns()
    rng = np.random.default_rng(9)
    for alpha, beta in random_algebras(rng, 12, "complex"):
        A = m.Algebra(alpha, beta)
        e = _phi_e(A)
        assert isinstance(A.phi((1.0, 2.0)), complex)
        assert abs(e.imag) > 1e-9
        assert abs(e * e - (alpha + beta * e)) < 1e-12 * max(1, abs(alpha), abs(beta) ** 2)
        assert complex(A.phi((1.0, 0.0))) == pytest.approx(1.0, abs=1e-15)
        a, b, c, d = rng.normal(size=(4, 25))
        pz = np.asarray(A.phi((a, b)), dtype=complex)
        np.testing.assert_allclose(pz, a + b * e, rtol=1e-12, atol=1e-12)        # R-linear
        prod = A.mul((a, b), (c, d))
        np.testing.assert_allclose(np.asarray(A.phi(prod), dtype=complex),
                                   pz * np.asarray(A.phi((c, d)), dtype=complex), rtol=1e-10, atol=1e-10)
        P = A.poly((a, b))
        np.testing.assert_allclose(np.asarray(A.phi(P), dtype=complex), np.polyval(COEFFS, pz),
                                   rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (-2.25, 3.0), (0.0, 0.0), (2.0, 1.0), (-2.25, 3.001)])
def test_phi_refuses_split_and_dual(alpha, beta):
    A = ns().Algebra(alpha, beta)
    assert A.kind != "complex"
    with pytest.raises(ValueError):
        A.phi((1.0, 1.0))


# ----------------------------------------------------------------------------- rho

def test_rho_on_the_real_line_is_abs_p():
    m = ns()
    rng = np.random.default_rng(10)
    x = rng.uniform(-6, 6, 300)
    for kind, (alpha, beta) in all_kinds(rng):
        r = np.asarray(m.Algebra(alpha, beta).rho((x, np.zeros_like(x))))
        close(r, np.abs(np.polyval(COEFFS, x)), rtol=1e-12, atol=0)
    assert float(m.Algebra(1.0, 0.0).rho((X_TRUE, 0.0))) == pytest.approx(M_TRUE, rel=1e-12)


def test_rho_is_the_spectral_radius_for_every_kind():
    m = ns()
    rng = np.random.default_rng(11)
    for kind, (alpha, beta) in all_kinds(rng, 10):
        A = m.Algebra(alpha, beta)
        a, b = rng.uniform(-3, 3, 400), rng.uniform(-3, 3, 400)
        close(A.rho((a, b)), rho_ref(alpha, beta, a, b), rtol=1e-9, atol=0)
        close(A.rho((a, b), coeffs=(1.0, 0.0, 1.0)), rho_ref(alpha, beta, a, b, (1.0, 0.0, 1.0)), rtol=1e-9, atol=0)


def test_rho_in_complex_algebras_is_abs_p_of_phi():
    m = ns()
    rng = np.random.default_rng(12)
    for alpha, beta in random_algebras(rng, 12, "complex"):
        A = m.Algebra(alpha, beta)
        a, b = rng.uniform(-3, 3, 300), rng.uniform(-3, 3, 300)
        pz = np.asarray(A.phi((a, b)), dtype=complex)
        close(A.rho((a, b)), np.abs(np.polyval(COEFFS, pz)), rtol=1e-9, atol=1e-12)


def test_real_floor_holds_in_every_split_or_dual_algebra():
    """200 random split or dual algebras (spec 9.1: dual kind includes delta just below zero)
    x 10^4 random wide points: rho >= m (1 - 1e-12)."""
    m = ns()
    rng = np.random.default_rng(13)
    algebras = [(1.0, 0.0), (-2.25, 3.0)]
    while len(algebras) < 200:
        beta = float(rng.uniform(-6, 6))
        r = rng.random()
        if r < 0.25:
            alpha = -beta * beta / 4                                       # exactly dual
        elif r < 0.32:
            alpha = (-beta * beta + abs(beta * beta) * 1e-13) / 4          # delta > 0 inside the dual tolerance
        elif r < 0.40:
            alpha = (-beta * beta - abs(beta * beta) * 1e-13) / 4          # delta < 0 inside the dual tolerance
        else:
            alpha = float(rng.uniform(-5, 5))
        if dual_kind(alpha, beta, DUAL_TOL) in ("split", "dual"):
            algebras.append((alpha, beta))
    worst = np.inf
    for alpha, beta in algebras:
        n = 10_000
        uni = rng.uniform(-50, 50, size=(2, n))
        logw = 10 ** rng.uniform(-3, 3, size=(2, n)) * rng.choice([-1.0, 1.0], size=(2, n))
        pick = rng.random(size=(2, n)) < 0.5
        a, b = np.where(pick, uni, logw)
        r = np.asarray(m.Algebra(alpha, beta).rho((a, b)), dtype=float)
        assert r.shape == (n,)
        assert np.all(np.isfinite(r))
        worst = min(worst, float(r.min() / M_TRUE))
        assert r.min() >= M_TRUE * (1 - FLOOR_TOL), (alpha, beta, r.min() / M_TRUE)
    assert worst < 1.5        # the sample does come near the floor (otherwise the check is idle)


def test_real_floor_holds_on_the_lines_that_touch_it():
    """Points with a + b r = x* for a root r: rho is exactly m there in exact arithmetic."""
    m = ns()
    rng = np.random.default_rng(14)
    for kind in ("dual", "split", "dual_neg", "dual_pos"):
        for alpha, beta in random_algebras(rng, 25, kind):
            d = beta * beta + 4 * alpha
            if dual_kind(alpha, beta, DUAL_TOL) == "dual":
                roots = {beta / 2}                                   # spec 9.1: double eigenvalue beta/2
            else:
                roots = {(beta + math.sqrt(d)) / 2, (beta - math.sqrt(d)) / 2}
            for r in roots:
                b = rng.uniform(-3, 3, 2000)
                a = X_TRUE - b * r
                got = np.asarray(m.Algebra(alpha, beta).rho((a, b)))
                assert got.min() >= M_TRUE * (1 - FLOOR_TOL), (kind, alpha, beta, got.min() / M_TRUE)
                close(got, rho_ref(alpha, beta, a, b), rtol=1e-9, atol=0)


def test_rho_is_zero_at_the_roots_of_complex_algebras():
    m = ns()
    rng = np.random.default_rng(15)
    for alpha, beta in [(-1.0, 0.0)] + random_algebras(rng, 10, "complex"):
        A = m.Algebra(alpha, beta)
        roots = A.roots()
        assert isinstance(roots, list) and len(roots) == 4
        for z in roots:
            assert float(A.rho(tuple(z))) < 1e-9 * M_TRUE
            P = A.poly(tuple(z))
            assert abs(float(P[0])) < 1e-8 and abs(float(P[1])) < 1e-8
        images = np.array([complex(A.phi(tuple(z))) for z in roots])
        for r in SPEC_ROOTS:
            assert np.min(np.abs(images - r)) < 1e-9
        assert len(A.roots(coeffs=(1.0, 0.0, 1.0))) == 2


@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (-2.25, 3.0), (2.0, -1.0), (0.0, 0.0)])
def test_no_roots_outside_complex_algebras(alpha, beta):
    assert ns().Algebra(alpha, beta).roots() == []


# ----------------------------------------------------------------------------- coord_norm

def test_coord_norm_is_the_coordinate_norm_of_poly():
    m = ns()
    rng = np.random.default_rng(16)
    for kind, (alpha, beta) in all_kinds(rng):
        A = m.Algebra(alpha, beta)
        a, b = rng.uniform(-3, 3, 200), rng.uniform(-3, 3, 200)
        P = poly_ref(alpha, beta, (a, b))
        close(A.coord_norm((a, b)), np.hypot(P[0], P[1]), rtol=1e-10, atol=1e-9)


def test_coord_norm_reports_fake_successes_in_the_dual_algebra_while_rho_does_not():
    """Negative control for the floor: the rejected residual DOES dip to ~0.555 m at (-2.25, 3)."""
    m = ns()
    A = m.Algebra(-2.25, 3.0)
    assert A.kind == "dual"
    rng = np.random.default_rng(17)
    bounds = [(-10.0, 10.0), (-3.0, 3.0)]

    def cn(p):
        return float(A.coord_norm((float(p[0]), float(p[1]))))

    def rh(p):
        return float(A.rho((float(p[0]), float(p[1]))))

    best_cn, best_cn_x, best_rho = np.inf, None, np.inf
    for _ in range(40):
        x0 = [rng.uniform(-6, 4), rng.uniform(-3, 3)]
        res = minimize(cn, x0, method="Nelder-Mead", bounds=bounds,
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxfev": 4000})
        if res.fun < best_cn:
            best_cn, best_cn_x = res.fun, res.x
        res2 = minimize(rh, x0, method="Nelder-Mead", bounds=bounds,
                        options={"xatol": 1e-10, "fatol": 1e-12, "maxfev": 4000})
        best_rho = min(best_rho, res2.fun)
    assert best_cn < 0.6 * M_TRUE                     # "reaches below 0.6 m"
    assert best_cn > 0.55 * M_TRUE                    # "about 0.555 m"
    assert rh(best_cn_x) >= M_TRUE * (1 - FLOOR_TOL)  # no root there: rho stays on the floor
    assert best_rho >= M_TRUE * (1 - FLOOR_TOL)
    assert best_rho <= M_TRUE * (1 + 1e-6)            # and rho does reach the floor


def test_dual_kind_with_delta_just_below_zero_uses_the_double_eigenvalue():
    """Spec 9.1: kind == "dual" includes a delta within the tolerance but slightly below zero, and rho is
    |p(a + b beta/2)| there. Treating it as complex would dip below the floor on a + b beta/2 = x*."""
    m = ns()
    rng = np.random.default_rng(18)
    algebras = random_algebras(rng, 20, "dual_neg")
    for alpha, beta in algebras:
        A = m.Algebra(alpha, beta)
        assert A.delta < 0 and A.kind == "dual", (alpha, beta, A.delta, A.kind)
        a, b = rng.uniform(-3, 3, 500), rng.uniform(-3, 3, 500)
        close(A.rho((a, b)), np.abs(np.polyval(COEFFS, a + b * beta / 2)), rtol=1e-11, atol=0)
        b_line = np.concatenate([rng.uniform(-3, 3, 2000), [-3.0, 3.0]])
        a_line = X_TRUE - b_line * beta / 2
        on_line = np.asarray(A.rho((a_line, b_line)))
        assert on_line.min() >= M_TRUE * (1 - FLOOR_TOL), (alpha, beta, on_line.min() / M_TRUE - 1)
        wa, wb = rng.uniform(-50, 50, 10_000), rng.uniform(-50, 50, 10_000)
        assert np.asarray(A.rho((wa, wb))).min() >= M_TRUE * (1 - FLOOR_TOL)
        with pytest.raises(ValueError):
            A.phi((1.0, 1.0))
        assert A.roots() == []
