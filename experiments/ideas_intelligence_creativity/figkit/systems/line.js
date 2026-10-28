// Instrument 1, "The answer is off the line" (SPEC 10.3). The searchers run the experiment's own rule
// (experiment3.py, hc_fixed): propose x + N(0, sigma) and move only on a strict improvement in |p|. In the
// plane each coordinate gets its own proposal. Everything is in model units; the essay's numbers come
// from the recorded results, which the data carries.
(function () {
  'use strict';
  var STEPS_PER_FRAME = 4;                    // sweeps of every walker per frame
  var TRAIL = 6;                              // frames of trail kept for each walker
  var SOLUTION = 1e-3;                        // "a solution" once the best distance from zero is below this times the floor
  var AT_FLOOR = 1e-4;                        // "the lowest the line allows" once within this relative margin
  var KEY_STEP = 0.05, KEY_BIG = 0.5;         // model units per arrow key; Shift moves ten times as far
  var PLANE = { l: 0.035, t: 0.04, r: 0.965, b: 0.7 };
  var STRIP = { t: 0.77, b: 0.94 };           // the strip along the line, as fractions of the stage height
  var DIP_HEADROOM = 0.4;                     // log10 units the strip rises above the highest recorded dip
  var SAMPLES = 320;
  var FIELD_CELL = 2;                         // CSS px per cell of the cached |p| field, drawn up with smoothing
  var FIELD_GAMMA = 1.5;                      // the plane's glows: a steeper curve, so the roots stand out

  function clamp(v, range) { return Math.min(range[1], Math.max(range[0], v)); }
  function short(v) {
    var s = v.toFixed(2).replace(/\.?0+$/, '');
    if (s === '-0') s = '0';
    return s.replace('-', '−');
  }
  function loc(x, y) { return short(x) + (y < 0 ? ' − ' : ' + ') + short(Math.abs(y)) + 'i'; }
  function valueText(v) {
    if (v < 1e-6) return 'under 0.000001';
    return v < 0.001 ? v.toPrecision(2) : v.toFixed(3);
  }
  var KEY = '<span class="fk-orange">○</span> orange: the four solutions · ' +
            '<span class="fk-white">○</span> white: your probe (drag it)';
  function plainText(html) { return html.replace(/<br>/g, '\n').replace(/<[^>]+>/g, ''); }

  FigKit.systems.line = function (kit) {
    var data, view, coeffs, m, sigma, start, roots, minima, second, N, rng, reseeds = 0;
    var mode = 'line';
    var X = [], Y = [], F = [], TR = [];
    var best = null, bestLine = null;          // {v, x, y}: the lowest |p| seen, and the lowest on the real line
    var guesses = 0;                           // every proposal so far, for the readout
    var probe = { x: 0, y: 0 };
    var logs = [], samples = [], lo = 0, hi = 1, field = null;

    function absP(x, y) { return absPolyComplex(coeffs, x, y); }
    function track(i) {
      var v = F[i];
      if (!best || v < best.v) best = { v: v, x: X[i], y: Y[i] };
      if (Y[i] === 0 && (!bestLine || v < bestLine.v)) bestLine = { v: v, x: X[i], y: 0 };
    }
    function clampProbe() {
      probe.x = clamp(probe.x, view.x);
      probe.y = mode === 'line' ? 0 : clamp(probe.y, view.y);
    }
    function reseed() {
      rng = kit.rng(data.seed + 1000003 * reseeds);
      reseeds++;
      for (var i = 0; i < N; i++) {
        X[i] = rng.uniform(start[0], start[1]);
        Y[i] = 0;
        F[i] = absP(X[i], 0);
        TR[i] = [];
        track(i);
      }
    }
    // the |p| field, cached as a picture. Brighter means closer to zero. It is computed on a grid of
    // FIELD_CELL px cells and drawn up with smoothing, so it reads as a glow rather than a mosaic.
    function buildField() {
      var p = kit.plot, w = p.r - p.l, h = p.b - p.t, C = kit.C;
      var nx = Math.max(24, Math.round(w / FIELD_CELL)), ny = Math.max(16, Math.round(h / FIELD_CELL));
      var gamma = mode === 'plane' ? FIELD_GAMMA : 1;
      var small = document.createElement('canvas');
      small.width = nx;
      small.height = ny;
      var sg = small.getContext('2d');
      sg.fillStyle = C.accent;
      for (var j = 0; j < ny; j++) {
        for (var i = 0; i < nx; i++) {
          var x = view.x[0] + (i + 0.5) / nx * (view.x[1] - view.x[0]);
          var y = view.y[1] - (j + 0.5) / ny * (view.y[1] - view.y[0]);
          var base = clamp((2.2 - Math.log10(absP(x, y) + 1e-12)) / 3.2, [0, 1]);
          sg.globalAlpha = Math.pow(base, gamma);
          sg.fillRect(i, j, 1, 1);
        }
      }
      field = kit.offscreen(w, h);
      field.ctx.imageSmoothingEnabled = true;
      field.ctx.drawImage(small, 0, 0, w, h);
    }
    function setMode(next) {
      if (next === mode) return;
      mode = next;
      if (mode === 'line') {                    // projecting the searchers back onto the line
        for (var i = 0; i < N; i++) {
          Y[i] = 0;
          F[i] = absP(X[i], 0);
          TR[i] = [];
          track(i);
        }
      }
      buildField();
      clampProbe();
    }

    var system = {
      box: PLANE,
      pills: [
        { id: 'line', label: 'the real line', get on() { return mode === 'line'; } },
        { id: 'plane', label: 'allow a second direction', get on() { return mode === 'plane'; } }
      ],
      onPill: function (id) { setMode(id === 'plane' ? 'plane' : 'line'); },
      action: { label: 'release the searchers again', run: reseed },

      init: function (d) {
        data = d;
        view = { x: d.view.re.slice(), y: d.view.im.slice() };
        system.view = view;
        coeffs = d.coeffs.slice();
        m = d.m;
        sigma = d.sigma;
        start = d.start.slice();
        roots = d.roots;
        minima = d.minima;
        var byHeight = minima.slice().sort(function (a, b) { return a[1] - b[1]; });
        second = byHeight.length > 1 ? byHeight[1] : null;     // the lowest dip is the floor; this is the next one
        N = d.walkers;
        probe = { x: d.x_star, y: 0 };
        for (var k = 0; k < SAMPLES; k++) {
          var x = view.x[0] + k / (SAMPLES - 1) * (view.x[1] - view.x[0]);
          samples.push(x);
          logs.push(Math.log10(absP(x, 0)));
        }
        // the strip is log scaled, and it stops just above the highest dip, so the dips differ on screen; the
        // curve is clipped at the top edge where it climbs above that
        lo = Math.log10(m) - 0.25;
        hi = Math.max.apply(null, minima.map(function (mn) { return Math.log10(mn[1]); })) + DIP_HEADROOM;
        reseed();
      },

      resize: function () { buildField(); },

      step: function () {
        guesses += STEPS_PER_FRAME * N * (mode === 'plane' ? 2 : 1);
        for (var s = 0; s < STEPS_PER_FRAME; s++) {
          for (var i = 0; i < N; i++) {
            var cx = X[i] + sigma * rng.gauss(), v = absP(cx, Y[i]);
            if (v < F[i]) { X[i] = cx; F[i] = v; track(i); }
            if (mode === 'plane') {
              var cy = Y[i] + sigma * rng.gauss(), w = absP(X[i], cy);
              if (w < F[i]) { Y[i] = cy; F[i] = w; track(i); }
            }
          }
        }
        for (var k = 0; k < N; k++) {
          TR[k].push([X[k], Y[k]]);
          if (TR[k].length > TRAIL) TR[k].shift();
        }
      },

      draw: function () {
        var c = kit.ctx, C = kit.C, p = kit.plot;
        var y0 = kit.toPx(view.x[0], 0)[1];
        var s0 = kit.H * STRIP.t, s1 = kit.H * STRIP.b;
        function sy(lv) { return s1 - (lv - lo) / (hi - lo) * (s1 - s0); }

        c.globalAlpha = mode === 'line' ? 0.35 : 1;
        c.imageSmoothingEnabled = true;
        c.drawImage(field.canvas, p.l, p.t, field.w, field.h);
        c.globalAlpha = 1;

        c.setLineDash([4, 5]);
        c.strokeStyle = C.muted;
        c.beginPath(); c.moveTo(p.l, y0); c.lineTo(p.r, y0); c.stroke();
        c.setLineDash([]);
        c.fillStyle = C.muted;
        c.font = '11px "IBM Plex Mono", ui-monospace, monospace';
        c.textAlign = 'center';
        for (var t = Math.ceil(view.x[0]); t <= view.x[1]; t++) {
          c.fillText(String(t).replace('-', '−'), kit.toPx(t, 0)[0], y0 + 14);
        }
        c.textAlign = 'right';
        c.fillText('the real line', p.r - 4, y0 - 6);

        c.strokeStyle = C.pencil;
        c.lineWidth = 1.5;
        roots.forEach(function (r) {
          var q = kit.toPx(r[0], r[1]);
          c.beginPath(); c.arc(q[0], q[1], 5, 0, 2 * Math.PI); c.stroke();
        });

        c.lineWidth = 1.4;
        for (var i = 0; i < N; i++) {
          var on = Y[i] === 0, tr = TR[i];
          c.strokeStyle = on ? C.accent : C.pencil;
          c.globalAlpha = 0.5;
          c.beginPath();
          tr.forEach(function (pt, k) {
            var q = kit.toPx(pt[0], pt[1]);
            if (k) c.lineTo(q[0], q[1]); else c.moveTo(q[0], q[1]);
          });
          c.stroke();
          c.globalAlpha = 1;
          var h = kit.toPx(X[i], Y[i]);
          c.fillStyle = on ? C.accent : C.pencil;
          c.beginPath(); c.arc(h[0], h[1], 2.6, 0, 2 * Math.PI); c.fill();
        }
        c.lineWidth = 1;

        c.strokeStyle = C.rule;
        c.beginPath(); c.moveTo(p.l, s1 + 0.5); c.lineTo(p.r, s1 + 0.5); c.stroke();
        c.strokeStyle = C.accent;
        c.beginPath();
        for (var k2 = 0; k2 < SAMPLES; k2++) {
          var sx = kit.toPx(samples[k2], 0)[0], sv = sy(Math.min(logs[k2], hi));
          if (k2) c.lineTo(sx, sv); else c.moveTo(sx, sv);
        }
        c.stroke();
        var fy = sy(Math.log10(m));
        c.setLineDash([5, 4]);
        c.strokeStyle = C.muted;
        c.beginPath(); c.moveTo(p.l, fy); c.lineTo(p.r, fy); c.stroke();
        c.setLineDash([]);
        c.fillStyle = C.muted;
        c.textAlign = 'left';
        c.fillText('floor ' + m.toFixed(2), p.l + 4, fy - 5);
        c.fillStyle = C.text;
        minima.forEach(function (mn) {
          c.beginPath(); c.arc(kit.toPx(mn[0], 0)[0], sy(Math.log10(mn[1])), 3, 0, 2 * Math.PI); c.fill();
        });
        if (second) {
          var dx = kit.toPx(second[0], 0)[0], dy = sy(Math.log10(second[1])) - 9;
          c.font = '11px "IBM Plex Mono", ui-monospace, monospace';
          var dipText = 'second dip ' + second[1].toFixed(1), dw = c.measureText(dipText).width;
          c.textAlign = 'center';
          c.fillStyle = C.band;
          c.globalAlpha = 0.85;
          c.fillRect(dx - dw / 2 - 4, dy - 10, dw + 8, 14);
          c.globalAlpha = 1;
          c.fillStyle = C.text;
          c.fillText(dipText, dx, dy);
        }
        if (mode === 'line') {
          c.fillStyle = C.accent;
          c.globalAlpha = 0.6;
          for (var j = 0; j < N; j++) {
            c.beginPath(); c.arc(kit.toPx(X[j], 0)[0], sy(Math.log10(F[j])), 2, 0, 2 * Math.PI); c.fill();
          }
          c.globalAlpha = 1;
        }
        var px = kit.toPx(probe.x, 0)[0];
        c.setLineDash([2, 4]);
        c.strokeStyle = C.text;
        c.globalAlpha = 0.6;
        c.beginPath(); c.moveTo(px, s0); c.lineTo(px, s1); c.stroke();
        c.setLineDash([]);
        c.globalAlpha = 1;
        c.fillStyle = C.muted;
        c.textAlign = 'left';
        c.fillText('distance from zero, along the line', p.l + 4, s0 - 4);
      },

      probe: {
        get: function () { return { x: probe.x, y: probe.y }; },
        set: function (x, y) { probe.x = x; probe.y = y; clampProbe(); },
        keyStep: function (dx, dy, big) {
          var s = big ? KEY_BIG : KEY_STEP;
          probe.x += dx * s;
          probe.y += dy * s;
          clampProbe();
        },
        aria: function () {
          if (mode === 'line') {
            return { label: 'probe on the real line at x = ' + short(probe.x) + ', distance from zero ' +
                       kit.fmt(absP(probe.x, 0), 2),
                     min: view.x[0], max: view.x[1], now: probe.x };
          }
          return { label: 'probe at ' + loc(probe.x, probe.y) + ', distance from zero ' +
                     kit.fmt(absP(probe.x, probe.y), 2) };
        }
      },

      readout: function () {
        var line = mode === 'line', b = line ? bestLine : best;
        var atFloor = line && b.v <= m * (1 + AT_FLOOR);
        var solved = !line && best.v < SOLUTION * m;
        var where = line ? 'x = ' + short(b.x) : loc(b.x, b.y);
        var probeWhere = line ? 'x = ' + short(probe.x) : loc(probe.x, probe.y);
        // a hill-climber with steps of sigma stops short of an exact root, so a near miss is not called a solution
        var main = atFloor
          ? 'best so far <b>' + valueText(b.v) + '</b> — the lowest the line allows'
          : 'best so far <b>' + valueText(b.v) + '</b> at <b>' + where + '</b> — ' +
            (solved ? 'right next to a solution (as close as steps of ' + short(sigma) + ' get)'
                    : 'still searching');
        var count = 0;
        for (var i = 0; i < N; i++) {
          if (line ? F[i] <= m * (1 + AT_FLOOR) : F[i] < SOLUTION * m) count++;
        }
        var html = main + '<br>probe: <b>' + probeWhere + '</b>, distance from zero <b>' +
                   kit.fmt(absP(probe.x, probe.y), 2) + '</b><br>searchers ' +
                   (line ? 'at the lowest point' : "at a solution's doorstep") + ': <b>' + count + '</b> of ' + N +
                   ', guesses so far <b>' + kit.grouped(guesses) + '</b>';
        return { html: html, text: plainText(html),
                 verdict: (line ? 'line:' : 'plane:') + (atFloor || solved ? 'done' : 'searching') };
      },

      key: function () { return KEY; }
    };
    return system;
  };
})();
