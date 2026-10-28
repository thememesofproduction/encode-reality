"""SPEC.md 10.1: build_instrument writes one self-contained page per instrument.

The page inlines the kit (kit.js, kit.css), the maths the systems share (systems/numsys_math.js), the
system file and the data. Its only links are the two same-origin design stylesheets. Before the file is
written, the page is checked against the privacy guard, which holds the deploy lint's blocking patterns.
"""

from __future__ import annotations

import html as htmllib
import json
import os
import re
from pathlib import Path

FIGKIT_DIR = Path(__file__).resolve().parent
SYSTEMS_DIR = FIGKIT_DIR / "systems"
SHARED_SYSTEMS = ("numsys_math",)            # maths the other systems call: inlined first, once
STYLESHEETS = ("/assets/er/tokens.css", "/assets/er/fonts.css")
EMBED_NAME, EMBED_CONTENT = "er-embed", "self-themed"
NAME_RE = re.compile(r"[a-z][a-z0-9_]*")

# SPEC 10.1, the privacy guard: literal tokens, then names matched case-insensitively, then patterns.
GUARD_LITERAL = ("/srv/", "/home/", "sourceMappingURL")
GUARD_NAMES = ()   # plus FIGKIT_PRIVATE_NAMES: private names are never written into the repo
GUARD_PATTERNS = {
    "e-mail address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "ISO-8601 time with a numeric UTC offset": re.compile(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?[+-]\d{2}:?\d{2}"),
    "run of 40 hexadecimal characters": re.compile(r"[0-9a-fA-F]{40}"),
}

THEME_BOOT = (
    "(function () {\n"
    "  try {\n"
    "    var t = localStorage.getItem('er-theme');\n"
    "    if (t === 'paper' || t === 'ink') document.documentElement.dataset.theme = t;\n"
    "  } catch (e) {}\n"
    "})();"
)

SKELETON = (
    '<main class="fk fk-{system}" data-fk="root">\n'
    '  <div class="fk-band">\n'
    '    <div class="fk-stage" data-fk="stage" role="group" aria-label="{aria}">\n'
    '      <canvas class="fk-canvas" data-fk="canvas" aria-hidden="true"></canvas>\n'
    '      <button class="fk-probe" data-fk="probe" type="button"></button>\n'
    '    </div>\n'
    '    <p class="fk-key" data-fk="key" hidden></p>\n'
    '    <div class="fk-controls">\n'
    '      <div class="fk-pills" data-fk="pills"></div>\n'
    '      <div class="fk-buttons">\n'
    '        <button class="fk-pill" data-fk="action" type="button" hidden></button>\n'
    '        <button class="fk-pill" data-fk="run" type="button" hidden>Run</button>\n'
    '      </div>\n'
    '    </div>\n'
    '    <div class="fk-readout" data-fk="readout" aria-live="polite"></div>\n'
    '  </div>\n'
    '{caption}'
    '</main>\n'
)


def _inline(source: str) -> str:
    if re.search(r"<!--|</script", source, re.I):
        raise ValueError("an inline script contains a sequence that would hide or close it")
    return "<script>\n" + source.rstrip() + "\n</script>\n"


def _data_block(data) -> str:
    """The data as JSON. '<' becomes \\u003c, so nothing in the data can close the script element."""
    text = json.dumps(data, allow_nan=False, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return '<script type="application/json" data-fk="data">' + text.replace("<", "\\u003c") + "</script>\n"


def _guard(text: str) -> None:
    for token in GUARD_LITERAL:
        if token in text:
            raise ValueError(f"privacy guard: the page contains {token!r}")
    lowered = text.lower()
    private = [n.strip().lower() for n in os.environ.get("FIGKIT_PRIVATE_NAMES", "").split(",") if n.strip()]
    for name in (*GUARD_NAMES, *private):
        if name in lowered:
            raise ValueError(f"privacy guard: the page contains {name!r}")
    for label, pattern in GUARD_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"privacy guard: the page contains a {label}")


def build_instrument(system: str, data: dict, out, *, title: str, aria_label: str,
                     caption: str | None = None) -> Path:
    """Write the instrument page for figkit/systems/<system>.js to `out` and return its Path."""
    if not isinstance(system, str) or not NAME_RE.fullmatch(system) or not (SYSTEMS_DIR / f"{system}.js").is_file():
        raise ValueError(f"unknown instrument system {system!r}: there is no figkit/systems/<name>.js for it")
    out = Path(out)
    kit_css = (FIGKIT_DIR / "kit.css").read_text(encoding="utf-8")
    sources = [(SYSTEMS_DIR / f"{name}.js").read_text(encoding="utf-8")
               for name in SHARED_SYSTEMS if name != system]
    sources.append((FIGKIT_DIR / "kit.js").read_text(encoding="utf-8"))
    sources.append((SYSTEMS_DIR / f"{system}.js").read_text(encoding="utf-8"))

    caption_html = ""
    if caption is not None:
        caption_html = f'  <p class="fk-caption" data-fk="caption">{htmllib.escape(caption)}</p>\n'
    skeleton = SKELETON.format(system=system, aria=htmllib.escape(aria_label, quote=True), caption=caption_html)

    parts = [
        '<!doctype html>\n<html lang="en">\n<head>\n',
        '<meta charset="utf-8">\n',
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
        f'<meta name="{EMBED_NAME}" content="{EMBED_CONTENT}">\n',
        f"<title>{htmllib.escape(title)}</title>\n",
        _inline(THEME_BOOT),
        f'<link rel="stylesheet" href="{STYLESHEETS[0]}">\n',
        f'<link rel="stylesheet" href="{STYLESHEETS[1]}">\n',
        "<style>\n" + kit_css.rstrip() + "\n</style>\n",
        "</head>\n<body>\n",
        skeleton,
        _data_block(data),
    ]
    parts += [_inline(source) for source in sources]
    parts += [_inline(f'FigKit.start("{system}");'), "</body>\n</html>\n"]
    page = "".join(parts)
    _guard(page)
    out.write_text(page, encoding="utf-8")
    return out
