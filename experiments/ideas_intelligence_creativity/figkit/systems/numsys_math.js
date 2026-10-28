// The maths the instruments run (SPEC 10.2). A plain script with no imports: in a page its functions are
// globals; under node it is loaded through module.exports. It mirrors number_systems.py: Horner's rule,
// the same kind test (delta against 1e-12 times the larger of 1, beta^2 and |4 alpha|), and the residual
// formula of each kind (dual: at beta/2; split: the larger of the two real roots; complex: the isomorphism).

// |p(re + i im)| by Horner's rule in complex arithmetic. coeffs are highest degree first.
function absPolyComplex(coeffs, re, im) {
  var ar = 0, ai = 0;
  for (var k = 0; k < coeffs.length; k++) {
    var nr = ar * re - ai * im + coeffs[k];
    ai = ar * im + ai * re;
    ar = nr;
  }
  return Math.hypot(ar, ai);
}

// 'split', 'dual' or 'complex' for e^2 = alpha + beta e.
function kindOf(alpha, beta) {
  var delta = beta * beta + 4 * alpha;
  var scale = Math.max(1, beta * beta, Math.abs(4 * alpha));
  if (Math.abs(delta) <= 1e-12 * scale) return 'dual';
  return delta < 0 ? 'complex' : 'split';
}

// The spectral radius of multiplication by p(z), for z = a + b e in the algebra e^2 = alpha + beta e.
function rho(coeffs, alpha, beta, a, b) {
  var kind = kindOf(alpha, beta);
  var delta = beta * beta + 4 * alpha;
  function real(t) {
    var acc = 0;
    for (var k = 0; k < coeffs.length; k++) acc = acc * t + coeffs[k];
    return Math.abs(acc);
  }
  if (kind === 'dual') return real(a + b * (beta / 2));
  if (kind === 'split') {
    var s = Math.sqrt(delta);
    return Math.max(real(a + b * ((beta + s) / 2)), real(a + b * ((beta - s) / 2)));
  }
  return absPolyComplex(coeffs, a + b * (beta / 2), b * (Math.sqrt(-delta) / 2));
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { absPolyComplex: absPolyComplex, rho: rho, kindOf: kindOf };
}
