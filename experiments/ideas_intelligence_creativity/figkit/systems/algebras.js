// Instrument 2, "Choose a number system" (SPEC 10.3). The background is the recorded map: each cell is an
// algebra e^2 = alpha + beta e, coded floor / below the floor / root. The probe is the chosen algebra. The
// live search is the experiment's (1+1)-ES with the 1/5th rule, started on the real line and scored by rho
// (numsys_math.js), up to child_evals evaluations. The replay draws one recorded Level 2 run.
(function () {
  'use strict';
  var PER_FRAME = 50;                         // evaluations per frame
  var DEBOUNCE = 9;                           // frames to wait after the probe moves before the search restarts
  var SNAP_PX = 6;                            // a pointer this close to the parabola lands on it
  var KEY_STEP = 0.05, KEY_BIG = 0.5;         // model units per arrow key; Shift moves ten times as far
  var REPLAY_GENS = 10;                       // rounds (generations) per second in the replay
  var BOX = { l: 0.1, t: 0.05, r: 0.97, b: 0.88 };
  var KEY_REPLAY = 'dots: systems it tried · line: the ones it kept';

  function clamp(v, range) { return Math.min(range[1], Math.max(range[0], v)); }
  function short(v) {
    var s = v.toFixed(2).replace(/\.?0+$/, '');
    if (s === '-0') s = '0';
    return s.replace('-', '−');
  }
  function parab(beta) { return -beta * beta / 4; }
  var KIND_WORD = {
    split: 'two real numbers in disguise', dual: 'the dual numbers', complex: 'the complex numbers in disguise'
  };
  var KIND_LINE = {                           // where the number system sits on the map, in plain words
    split: 'above the curve: two real numbers in disguise',
    dual: 'on the curve: the dual numbers',
    complex: 'below the curve: the complex numbers in disguise'
  };
  function sup(n) {                           // superscript digits and minus, for 10⁻¹¹ and the like
    return String(n).replace('-', '⁻').replace(/[0-9]/g, function (d) { return '⁰¹²³⁴⁵⁶⁷⁸⁹'.charAt(+d); });
  }
  function ratioText(r) {                     // a ratio to the floor: 0.523, or 6 × 10⁻¹¹ with no e-notation
    if (r <= 0) return '0';
    if (r >= 0.01) return r.toFixed(3);
    var parts = r.toExponential(1).split('e');
    return parts[0].replace(/\.0$/, '') + ' × 10' + sup(parseInt(parts[1], 10));
  }
  function plainText(html) { return html.replace(/<br>/g, '\n').replace(/<[^>]+>/g, ''); }

  FigKit.systems.algebras = function (kit) {
    var data, view, coeffs, m, es, start, cat, alphas, betas, exemplar, seed, schedule, maxGen = 0;
    var probe = { x: 1, y: 2 };                 // x = beta, y = alpha: e^2 = 2 + e, two real numbers in disguise
    var run = null, pending = 0, restarts = 0, rng;
    var grid = null;
    var replay = { on: false, gen: -1, t0: 0 };

    // a label is drawn at a quarter strength while the probe is over it, so text never sits under the probe
    function label(c, text, x, y, align, colour, backed) {
      var C = kit.C, w = c.measureText(text).width;
      var left = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
      var p = kit.toPx(probe.x, probe.y);
      var near = p[0] > left - 12 && p[0] < left + w + 12 && p[1] > y - 22 && p[1] < y + 10;
      c.textAlign = align;
      if (backed) {
        c.globalAlpha = near ? 0.25 : 0.85;
        c.fillStyle = C.band;
        c.fillRect(left - 4, y - 12, w + 8, 16);
      }
      c.globalAlpha = near ? 0.25 : 1;
      c.fillStyle = colour;
      c.fillText(text, x, y);
      c.globalAlpha = 1;
    }
    // the guesses (evaluations) made once the given round is complete: the initial population, then each
    // generation's children and elite top-ups, as experiment3's Level-2 loop spends them
    function guessesAt(round) { return Math.min(schedule.cap, schedule.first + round * schedule.per_round); }
    function replayReadout() {
      var round = Math.floor(replay.gen);
      var html = 'replaying one recorded run · round <b>' + (round + 1) + '</b> of <b>' + (maxGen + 1) +
                 '</b> · <b>' + kit.grouped(guessesAt(round)) + '</b> guesses · first system below the curve at guess <b>' +
                 kit.grouped(exemplar.adoption_eval) + '</b>';
      return { html: html, text: plainText(html), verdict: 'replay' };
    }

    function moved() { run = null; pending = DEBOUNCE; }
    function startSearch() {
      restarts++;
      rng = kit.rng(seed + 7919 * restarts);
      run = { alpha: probe.y, beta: probe.x, cur: null, fcur: Infinity, sigma: es.ES_SIGMA0,
              best: Infinity, evals: 0, curve: [] };
    }
    function searchStep() {
      if (!run || run.evals >= es.child_evals) return;
      for (var k = 0; k < PER_FRAME && run.evals < es.child_evals; k++) {
        var f;
        if (run.cur === null) {
          var z0 = [rng.uniform(start[0], start[1]), 0];
          f = rho(coeffs, run.alpha, run.beta, z0[0], z0[1]);
          run.cur = z0;
          run.fcur = f;
          run.sigma = es.ES_SIGMA0;
        } else if (run.sigma < es.ES_SIGMA_MIN) {
          run.cur = null;
          continue;
        } else {
          var s = run.sigma;
          var z = [clamp(run.cur[0] + s * rng.gauss(), [-data.bound, data.bound]),
                   clamp(run.cur[1] + s * rng.gauss(), [-data.bound, data.bound])];
          f = rho(coeffs, run.alpha, run.beta, z[0], z[1]);
          if (f < run.fcur) {
            run.cur = z;
            run.fcur = f;
            run.sigma = s * es.ES_UP;
          } else {
            run.sigma = s * es.ES_DOWN;
          }
        }
        run.evals++;
        if (f < run.best) run.best = f;
        run.curve.push(run.best);
      }
    }
    function startReplay() {
      replay.on = true;
      replay.gen = 0;
      replay.t0 = window.performance.now();
    }
    function advanceReplay() {
      var g = (window.performance.now() - replay.t0) / 1000 * REPLAY_GENS;
      replay.gen = Math.min(g, maxGen);
      if (g >= maxGen) replay.on = false;
    }

    function buildGrid() {
      var p = kit.plot, w = p.r - p.l, h = p.b - p.t, C = kit.C;
      grid = kit.offscreen(w, h);
      var g = grid.ctx, vx = view.x[1] - view.x[0], vy = view.y[1] - view.y[0];
      var dB = betas.length > 1 ? (betas[1] - betas[0]) : 1, dA = alphas.length > 1 ? (alphas[1] - alphas[0]) : 1;
      var colour = [C.muted, C.pencil, C.pencil], alpha = [0.12, 0.3, 0.95];
      for (var i = 0; i < alphas.length; i++) {
        for (var j = 0; j < betas.length; j++) {
          var code = cat[i][j];
          var x0 = (betas[j] - dB / 2 - view.x[0]) / vx * w, y0 = (view.y[1] - (alphas[i] + dA / 2)) / vy * h;
          g.globalAlpha = alpha[code];
          g.fillStyle = colour[code];
          g.fillRect(x0, y0, dB / vx * w + 0.6, dA / vy * h + 0.6);
        }
      }
      g.globalAlpha = 1;
    }

    function drawParabola(c, C) {
      c.setLineDash([5, 4]);
      c.strokeStyle = C.muted;
      c.lineWidth = 1.2;
      c.beginPath();
      for (var k = 0; k <= 80; k++) {
        var beta = -2 + k / 80 * 4;
        var q = kit.toPx(beta, parab(beta));
        if (k) c.lineTo(q[0], q[1]); else c.moveTo(q[0], q[1]);
      }
      c.stroke();
      c.setLineDash([]);
      c.lineWidth = 1;
    }

    function drawReplay(c, C) {
      if (!exemplar || replay.gen < 0) return;
      var g = replay.gen;
      c.strokeStyle = C.text;
      c.globalAlpha = 0.3;
      c.beginPath();
      var n = 0;
      for (var gi = 0; gi < exemplar.elite_path.length && gi <= g; gi++) {
        var e = exemplar.elite_path[gi][0];
        var q = kit.toPx(e[1], e[0]);
        if (n++) c.lineTo(q[0], q[1]); else c.moveTo(q[0], q[1]);
      }
      c.stroke();
      c.globalAlpha = 1;
      exemplar.proposals.forEach(function (pr) {
        if (pr[3] > g) return;
        var below = pr[2] < 1 - 1e-9, q2 = kit.toPx(pr[1], pr[0]);
        c.fillStyle = below ? C.pencil : C.muted;
        c.beginPath(); c.arc(q2[0], q2[1], below ? 3 : 2.2, 0, 2 * Math.PI); c.fill();
      });
      var a = exemplar.adoption_algebra, s = kit.toPx(a[1], a[0]);
      c.fillStyle = C.text;
      c.beginPath();
      for (var k = 0; k < 10; k++) {
        var r = k % 2 ? 3.2 : 8, ang = -Math.PI / 2 + k * Math.PI / 5;
        if (k) c.lineTo(s[0] + r * Math.cos(ang), s[1] + r * Math.sin(ang));
        else c.moveTo(s[0] + r * Math.cos(ang), s[1] + r * Math.sin(ang));
      }
      c.closePath();
      c.fill();
      // the two lines sit in the clear band above the dome, centred over the star and kept inside the plot
      c.font = '11px "IBM Plex Mono", ui-monospace, monospace';
      var lines = ['one recorded run:', 'first below the curve at guess ' + kit.grouped(exemplar.adoption_eval)];
      var p = kit.plot, half = Math.max(c.measureText(lines[0]).width, c.measureText(lines[1]).width) / 2;
      var cx = Math.min(Math.max(s[0], p.l + 4 + half), p.r - 4 - half), base = kit.toPx(0, 1.3)[1];
      c.strokeStyle = C.text;
      c.globalAlpha = 0.5;
      c.beginPath(); c.moveTo(s[0], base + 4); c.lineTo(s[0], s[1] - 10); c.stroke();
      c.globalAlpha = 1;
      label(c, lines[0], cx, base - 14, 'center', C.text);
      label(c, lines[1], cx, base, 'center', C.text);
    }

    // "complex numbers in disguise" sits on the dome in dark type while the dome is wide enough at the top of
    // the text. On a small stage the dome is too narrow for it, so it moves above the dome with a leader to the arch.
    function drawComplexLabel(c, C) {
      var text = 'complex numbers in disguise', pad = 8, rise = 9;   // rise: the text's height above its baseline, px
      var base = kit.toPx(0, -0.42), top = kit.fromPx(0, base[1] - rise).y;
      var kx = (kit.plot.r - kit.plot.l) / (view.x[1] - view.x[0]);
      var domeWidth = 4 * Math.sqrt(-top) * kx;                  // the arch is beta = ±2 sqrt(-alpha)
      if (domeWidth - c.measureText(text).width >= 2 * pad) {
        label(c, text, base[0], base[1], 'center', C.band);
        return;
      }
      var lx = kit.toPx(-1.2, 0)[0], tag = kit.toPx(-1.2, 0.8), edge = kit.toPx(-1.2, parab(-1.2));
      c.strokeStyle = C.muted;
      c.lineWidth = 1;
      c.beginPath(); c.moveTo(lx, tag[1] + 3); c.lineTo(lx, edge[1]); c.stroke();
      label(c, text, lx, tag[1], 'center', C.muted);
    }

    function drawSpark(c, C, p) {
      if (!run || run.curve.length < 2) return;
      var w = Math.min(190, (p.r - p.l) * 0.42), h = Math.min(86, (p.b - p.t) * 0.3);
      var x0 = p.r - w - 8, y0 = p.t + 8, top = -10, bottom = 2;
      function sy(ratio) {
        var lv = Math.min(bottom, Math.max(top, Math.log10(Math.max(ratio, 1e-300))));
        return y0 + 6 + (bottom - lv) / (bottom - top) * (h - 12);
      }
      c.globalAlpha = 0.92;
      c.fillStyle = C.band;
      c.fillRect(x0, y0, w, h);
      c.globalAlpha = 1;
      c.strokeStyle = C.rule;
      c.strokeRect(x0 + 0.5, y0 + 0.5, w, h);
      var fy = sy(1);
      c.setLineDash([3, 3]);
      c.strokeStyle = C.muted;
      c.beginPath(); c.moveTo(x0 + 4, fy); c.lineTo(x0 + w - 4, fy); c.stroke();
      c.setLineDash([]);
      c.fillStyle = C.muted;
      c.font = '10px "IBM Plex Mono", ui-monospace, monospace';
      c.textAlign = 'right';
      c.fillText('floor', x0 + w - 6, fy - 3);
      c.textAlign = 'left';
      c.fillText('guesses', x0 + 6, y0 + h - 4);
      c.strokeStyle = C.accent;
      c.lineWidth = 1.5;
      c.beginPath();
      run.curve.forEach(function (v, k) {
        var px = x0 + 4 + k / (es.child_evals - 1) * (w - 8), py = sy(v / m);
        if (k) c.lineTo(px, py); else c.moveTo(px, py);
      });
      c.stroke();
      c.lineWidth = 1;
    }

    var system = {
      box: BOX,
      pills: [],
      get action() {
        return exemplar ? { label: 'replay a real run', run: startReplay } : null;
      },
      onPill: function () {},

      init: function (d) {
        data = d;
        coeffs = d.coeffs.slice();
        m = d.m;
        es = Object.assign({}, d.es);
        start = d.start.slice();
        cat = d.cat;
        alphas = d.alpha;
        betas = d.beta;
        exemplar = d.exemplar;
        schedule = d.schedule;
        seed = d.seed;
        view = { x: d.box.beta.slice(), y: d.box.alpha.slice() };
        system.view = view;
        if (exemplar) maxGen = Math.max.apply(null, exemplar.proposals.map(function (pr) { return pr[3]; }));
        startSearch();
      },

      resize: function () { buildGrid(); },

      step: function () {
        if (pending > 0 && --pending === 0) startSearch();
        searchStep();
        if (replay.on) advanceReplay();
      },

      draw: function () {
        var c = kit.ctx, C = kit.C, p = kit.plot;
        c.drawImage(grid.canvas, p.l, p.t, grid.w, grid.h);
        drawParabola(c, C);
        // the landmark: a small dashed ring at e² = −1, the usual i. Its label has a backing, since the cap under
        // it is too narrow to hold the text in dark type.
        var ring = kit.toPx(0, -1);
        c.setLineDash([3, 3]);
        c.lineWidth = 2;
        c.strokeStyle = C.band;
        c.beginPath(); c.arc(ring[0], ring[1], 7, 0, 2 * Math.PI); c.stroke();
        c.setLineDash([]);
        c.lineWidth = 1;
        c.font = '11px "IBM Plex Mono", ui-monospace, monospace';
        label(c, 'e² = −1 (the usual i)', ring[0] + 11, ring[1] - 4, 'left', C.text, true);
        // the labels keep clear of the sparkline inset (top right) and of the arch
        var up = kit.toPx(0, 2.4);
        label(c, 'two real numbers in disguise', up[0], up[1], 'center', C.muted);
        drawComplexLabel(c, C);
        var apex = kit.toPx(0.9, 0.02);
        label(c, 'dual numbers', apex[0], apex[1] - 4, 'left', C.muted);
        drawReplay(c, C);
        drawSpark(c, C, p);
        c.fillStyle = C.muted;
        c.textAlign = 'center';
        for (var t = -4; t <= 4; t += 2) c.fillText(String(t).replace('-', '−'), kit.toPx(t, view.y[0])[0], p.b + 16);
        c.textAlign = 'right';
        c.fillText('β →', p.r, p.b + 30);
        c.textAlign = 'right';
        for (var a = 0; a <= 5; a++) c.fillText(String(a), p.l - 6, kit.toPx(0, a)[1] + 4);
        c.textAlign = 'left';
        c.fillText('α ↑', p.l, p.t - 6);
      },

      probe: {
        get: function () { return { x: probe.x, y: probe.y }; },
        set: function (x, y) {
          probe.x = clamp(x, view.x);
          probe.y = clamp(y, view.y);
          moved();
        },
        pointer: function (px, py) {
          var q = kit.fromPx(px, py), beta = clamp(q.x, view.x), alpha = clamp(q.y, view.y);
          var pxPerY = (kit.plot.b - kit.plot.t) / (view.y[1] - view.y[0]);
          if (parab(beta) >= view.y[0] && Math.abs(alpha - parab(beta)) * pxPerY <= SNAP_PX) alpha = parab(beta);
          probe.x = beta;
          probe.y = alpha;
          moved();
        },
        keyStep: function (dx, dy, big) {
          var s = big ? KEY_BIG : KEY_STEP;
          var b0 = probe.x, a0 = probe.y;
          var b1 = clamp(b0 + dx * s, view.x), a1 = clamp(a0 + dy * s, view.y);
          // a step that crosses the parabola lands on it, so the dual numbers can be reached from the keys
          var side0 = Math.sign(a0 - parab(b0)), side1 = Math.sign(a1 - parab(b1));
          if (side0 * side1 < 0) a1 = parab(b1);
          probe.x = b1;
          probe.y = a1;
          moved();
        },
        aria: function () {
          return { label: 'number system with e squared = α + βe, α = ' + short(probe.y) +
                   ', β = ' + short(probe.x) + ', ' +
                   KIND_WORD[kindOf(probe.y, probe.x)] };
        }
      },

      // while the replay runs, the readout describes it; once it ends, the probe's readout comes back
      readout: function () {
        if (replay.on) return replayReadout();
        var a = probe.y, b = probe.x, kind = kindOf(a, b);
        var ratio = run && run.evals ? run.best / m : null;
        var status = ratio === null ? 'none' : (ratio < 1 - 1e-9 ? 'broke' : 'floor');
        var lastLine = ratio === null
          ? 'best: searching…'
          : status === 'broke'
            ? 'best: <b>' + ratioText(ratio) + '</b> of the floor — <b>broke through</b>'
            : 'best: <b>the floor</b> — stays at the floor';
        var html = 'e² = α + βe with α = <b>' + short(a) + '</b>, β = <b>' + short(b) + '</b><br>' +
                   KIND_LINE[kind] + '<br>' + lastLine;
        return { html: html, text: plainText(html), verdict: kind + ':' + status };
      },

      key: function () { return exemplar && replay.gen >= 0 ? KEY_REPLAY : null; }
    };
    return system;
  };
})();
