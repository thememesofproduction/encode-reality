// FigKit: the controller for the instrument pages. A page holds one namespace, window.FigKit.
// A system file registers a factory, FigKit.systems.<name> = function (kit) { ... return system; }.
// The kit owns the canvas, the probe button, the readout, the key, the pills and the frame loop. The system owns
// its state in model units and draws it. A resize or a theme change only redraws: state is kept. A system may
// have key(), returning the HTML of the key in the band, or null for no key; the key is read with the readout.
(function (win) {
  'use strict';
  var doc = win.document;
  var FK = win.FigKit || (win.FigKit = {});
  FK.systems = FK.systems || {};

  var DPR_MAX = 2;
  var SETTLE_STEPS = 300;     // steps run before a reduced-motion page shows its settled state
  var READOUT_MS = 1000;      // the readout text is written at most this often, sooner on a verdict change
  var REDUCED = !!(win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches);

  function el(root, name) { return root.querySelector('[data-fk="' + name + '"]'); }
  function now() { return win.performance ? win.performance.now() : Date.now(); }
  function cssVar(name, fallback) {
    var v = win.getComputedStyle(doc.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  // The stage is a dark band in both themes. The fallbacks are the paper values.
  function readColours() {
    return {
      band: cssVar('--er-band', '#1D1F23'),
      text: cssVar('--er-band-text', '#F4F2EC'),
      muted: cssVar('--er-band-muted', '#ADA9A0'),
      pencil: cssVar('--er-band-pencil', '#F0A35E'),
      live: cssVar('--er-band-live', '#7BD3A0'),
      rule: cssVar('--er-band-rule', '#3A3D43'),
      accent: cssVar('--er-band-accent', '#8DB4F0')
    };
  }
  // mulberry32: a small seeded generator, so the first frames are the same on every visit.
  function rngFor(seed) {
    var a = seed >>> 0;
    function u() {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = Math.imul(a ^ (a >>> 15), a | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    return {
      u: u,
      uniform: function (lo, hi) { return lo + (hi - lo) * u(); },
      gauss: function () { var r = u() || 1e-12; return Math.sqrt(-2 * Math.log(r)) * Math.cos(2 * Math.PI * u()); }
    };
  }
  function fmt(v, digits) { return v.toFixed(digits).replace('-', '−'); }
  function grouped(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }

  FK.mount = function mount(root, factory, data) {
    var stage = el(root, 'stage'), canvas = el(root, 'canvas'), probe = el(root, 'probe');
    var readoutEl = el(root, 'readout'), pillsEl = el(root, 'pills'), keyEl = el(root, 'key');
    var actionEl = el(root, 'action'), runEl = el(root, 'run');
    var ctx = canvas.getContext('2d');
    var playing = !REDUCED, visible = true, rafId = 0, destroyed = false;
    var lastText = null, lastVerdict = null, lastWrite = -Infinity, lastKey = null;
    var drag = null, pillList = [];

    var kit = {
      ctx: ctx, W: 0, H: 0, dpr: 1, C: readColours(), plot: { l: 0, t: 0, r: 0, b: 0 },
      reduced: REDUCED, rng: rngFor, fmt: fmt, grouped: grouped,
      toPx: toPx, fromPx: fromPx, offscreen: offscreen
    };
    var sys = factory(kit);

    // model units (sys.view) to stage pixels inside the plot box (kit.plot); y grows upwards
    function toPx(x, y) {
      var p = kit.plot, v = sys.view;
      return [p.l + (x - v.x[0]) / (v.x[1] - v.x[0]) * (p.r - p.l),
              p.b - (y - v.y[0]) / (v.y[1] - v.y[0]) * (p.b - p.t)];
    }
    function fromPx(px, py) {
      var p = kit.plot, v = sys.view;
      return { x: v.x[0] + (px - p.l) / (p.r - p.l) * (v.x[1] - v.x[0]),
               y: v.y[0] + (p.b - py) / (p.b - p.t) * (v.y[1] - v.y[0]) };
    }
    // a pixel cache, drawn with drawImage at CSS size; it is rebuilt on resize and on a theme change
    function offscreen(w, h) {
      var c = doc.createElement('canvas');
      c.width = Math.max(1, Math.round(w * kit.dpr));
      c.height = Math.max(1, Math.round(h * kit.dpr));
      var g = c.getContext('2d');
      g.setTransform(kit.dpr, 0, 0, kit.dpr, 0, 0);
      return { canvas: c, ctx: g, w: w, h: h };
    }

    function measure() {
      var r = stage.getBoundingClientRect(), box = sys.box || { l: 0, t: 0, r: 1, b: 1 };
      kit.dpr = Math.min(DPR_MAX, win.devicePixelRatio || 1);
      kit.W = r.width;
      kit.H = r.height;
      kit.plot = { l: box.l * kit.W, t: box.t * kit.H, r: box.r * kit.W, b: box.b * kit.H };
      canvas.width = Math.max(1, Math.round(kit.W * kit.dpr));
      canvas.height = Math.max(1, Math.round(kit.H * kit.dpr));
    }

    function placeProbe() {
      var p = sys.probe.get(), xy = toPx(p.x, p.y), a = sys.probe.aria();
      probe.style.left = xy[0].toFixed(2) + 'px';
      probe.style.top = xy[1].toFixed(2) + 'px';
      var key = JSON.stringify(a);
      if (key === placeProbe.last) return;
      placeProbe.last = key;
      probe.setAttribute('aria-label', a.label);
      if (a.now !== undefined) {
        probe.setAttribute('role', 'slider');
        probe.setAttribute('aria-valuemin', a.min);
        probe.setAttribute('aria-valuemax', a.max);
        probe.setAttribute('aria-valuenow', a.now);
        probe.setAttribute('aria-valuetext', a.label);
      } else {
        probe.removeAttribute('role');
        ['aria-valuemin', 'aria-valuemax', 'aria-valuenow', 'aria-valuetext'].forEach(function (n) {
          probe.removeAttribute(n);
        });
      }
    }
    placeProbe.last = null;

    function syncControls() {
      pillList.forEach(function (item) {
        item.button.setAttribute('aria-pressed', item.pill.on ? 'true' : 'false');
      });
      var act = sys.action;
      actionEl.hidden = !act;
      if (act && actionEl.textContent !== act.label) actionEl.textContent = act.label;
      runEl.hidden = !(REDUCED && !playing);
    }

    function syncKey() {
      if (!keyEl) return;
      var k = sys.key ? sys.key() : null;
      if (k === lastKey) return;
      lastKey = k;
      keyEl.hidden = !k;
      keyEl.innerHTML = k || '';
    }

    function writeReadout(force) {
      var r = sys.readout(), verdict = r.verdict === undefined ? '' : r.verdict, t = now();
      if (force || (r.text !== lastText && (verdict !== lastVerdict || t - lastWrite >= READOUT_MS))) {
        readoutEl.innerHTML = r.html;
        lastText = r.text;
        lastVerdict = verdict;
        lastWrite = t;
      }
      syncKey();
      syncControls();
    }

    function drawNow() {
      ctx.setTransform(kit.dpr, 0, 0, kit.dpr, 0, 0);
      ctx.clearRect(0, 0, kit.W, kit.H);
      sys.draw();
      placeProbe();
    }

    function settle() { for (var i = 0; i < SETTLE_STEPS; i++) sys.step(); }

    function schedule() {
      if (!rafId && playing && visible && !destroyed) rafId = win.requestAnimationFrame(frame);
    }
    function frame() {
      rafId = 0;
      sys.step();
      drawNow();
      writeReadout(false);
      schedule();
    }

    // a change the visitor made: show it at once
    function afterChange() {
      if (REDUCED && !playing) settle();
      drawNow();
      writeReadout(true);
    }
    function resize() {
      measure();
      if (sys.resize) sys.resize();
      drawNow();
      writeReadout(true);
    }
    function redraw() {
      kit.C = readColours();
      if (sys.resize) sys.resize();
      drawNow();
      writeReadout(true);
    }
    function play() { playing = true; syncControls(); schedule(); }
    function pause() {
      playing = false;
      if (rafId) { win.cancelAnimationFrame(rafId); rafId = 0; }
      syncControls();
    }
    function destroy() {
      destroyed = true;
      pause();
      if (ro) ro.disconnect();
      if (io) io.disconnect();
      if (mo) mo.disconnect();
      if (mq && mq.removeEventListener) mq.removeEventListener('change', redraw);
    }

    // pills, from the system's list; each pill reads its own `on` when the controls are synced
    (sys.pills || []).forEach(function (pill) {
      var b = doc.createElement('button');
      b.type = 'button';
      b.className = 'fk-pill';
      b.textContent = pill.label;
      b.addEventListener('click', function () { sys.onPill(pill.id); afterChange(); });
      pillsEl.appendChild(b);
      pillList.push({ button: b, pill: pill });
    });
    if (actionEl) actionEl.addEventListener('click', function () {
      if (sys.action) { sys.action.run(); afterChange(); }
    });
    runEl.addEventListener('click', play);

    // the probe: a 44px button. Dragging starts only on it; a tap on the stage moves it
    function pointerXY(e) {
      var r = stage.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    }
    function moveProbe(px, py) {
      if (sys.probe.pointer) {
        sys.probe.pointer(px, py);
      } else {
        var m = fromPx(px, py);
        sys.probe.set(m.x, m.y);
      }
      afterChange();
    }
    probe.addEventListener('pointerdown', function (e) {
      e.stopPropagation();
      var pt = pointerXY(e), p0 = sys.probe.get(), xy = toPx(p0.x, p0.y);
      drag = { id: e.pointerId, dx: xy[0] - pt[0], dy: xy[1] - pt[1] };
      probe.setPointerCapture(e.pointerId);
    });
    probe.addEventListener('pointermove', function (e) {
      if (!drag || e.pointerId !== drag.id) return;
      var pt = pointerXY(e);
      moveProbe(pt[0] + drag.dx, pt[1] + drag.dy);
    });
    function endDrag(e) { if (drag && e.pointerId === drag.id) drag = null; }
    probe.addEventListener('pointerup', endDrag);
    probe.addEventListener('pointercancel', endDrag);
    stage.addEventListener('pointerdown', function (e) {
      if (e.target === probe) return;
      var pt = pointerXY(e);
      moveProbe(pt[0], pt[1]);
      probe.focus({ preventScroll: true });
    });
    probe.addEventListener('keydown', function (e) {
      var dx = 0, dy = 0;
      if (e.key === 'ArrowLeft') dx = -1;
      else if (e.key === 'ArrowRight') dx = 1;
      else if (e.key === 'ArrowUp') dy = 1;
      else if (e.key === 'ArrowDown') dy = -1;
      else return;
      e.preventDefault();
      sys.probe.keyStep(dx, dy, e.shiftKey);
      afterChange();
    });

    // the system starts from its data; then the stage is measured, drawn, and (reduced) settled
    sys.init(data);
    measure();
    if (sys.resize) sys.resize();
    if (REDUCED) settle();
    drawNow();
    writeReadout(true);

    var ro = null, io = null, mo = null;
    var mq = win.matchMedia ? win.matchMedia('(prefers-color-scheme: dark)') : null;
    if (win.ResizeObserver) { ro = new win.ResizeObserver(resize); ro.observe(stage); }
    if (win.IntersectionObserver) {
      io = new win.IntersectionObserver(function (entries) {
        visible = entries[entries.length - 1].isIntersecting;
        schedule();
      });
      io.observe(stage);
    }
    if (win.MutationObserver) {
      mo = new win.MutationObserver(redraw);
      mo.observe(doc.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    }
    if (mq && mq.addEventListener) mq.addEventListener('change', redraw);
    schedule();

    return { redraw: redraw, pause: pause, resume: play, destroy: destroy };
  };

  // Called by each page's last script: mounts the system the page was built for.
  FK.start = function start(name) {
    var root = doc.querySelector('[data-fk="root"]');
    var block = doc.querySelector('script[data-fk="data"]');
    var factory = Object.prototype.hasOwnProperty.call(FK.systems, name) ? FK.systems[name] : null;
    if (!root || !block || typeof factory !== 'function') return null;
    return FK.mount(root, factory, JSON.parse(block.textContent));
  };
})(window);
