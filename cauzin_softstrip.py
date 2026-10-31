#!/usr/bin/env python3
"""
---ABOUT---

Script Name: cauzin_softstrip.py
Author: Filip Pawłowski
Contact: filippawlowski2012@gmail.com

Cauzin Softstrip encoder / decoder / distortion test bench.

Requires: Python 3.9+, numpy, Pillow.

    python cauzin_softstrip.py                       interactive menu
    python cauzin_softstrip.py encode a.bin b.txt -o out [--page A4]
    python cauzin_softstrip.py decode scan1.png scan2.png -d outdir/
    python cauzin_softstrip.py verify out-1.png      structural check of a generated strip
    python cauzin_softstrip.py stress [--quick]      distortion / scale test battery

FORMAT (as implemented here)
----------------------------
A strip, top to bottom:

  1. Horizontal sync   28 scan steps (0.0635 mm each) tall. The number of black runs
                       along it is 2n-4, which tells the reader n = nibbles per row.
  2. Vertical sync     >= 56 scan steps of ordinary rows that all carry one byte: the row
                       height in 1/16 scan steps. (Tells the reader how tall a row is.)
  3. Data rows         4n bits (n/2 bytes) per row, bytes LSB first, running on across rows.

One row, in bit cells (cell = bit width), w = 8n + 14 cells:

  | start bar | space | checker | L parity | 4n data dibits | R parity | space | rack  |
  |    2      |   1   |    2    |    2     |      8n        |    2     |   2   | 2..3  |

  * dibit: black|white = 0, white|black = 1 (the other two patterns are illegal, which
    makes every dibit self-clocking and lets a reader detect bad cells).
  * checker flips every row, the rack is 2 or 3 cells wide in step with it: together
    they give the reader a row clock on both sides of the strip.
  * L parity = XOR of the odd data dibits (0-based), R parity = XOR of the even ones.

Byte stream of a strip:
  00 00 00 | length(2, LE) | checksum | strip id(6) | seq | type | sw-expansion(2) |
  [first strip: os type, nfiles, per file: ctype, ostype, len(3), name, 00/FF, 00] | data
  length = bytes from the checksum to the end; checksum = add-with-carry of everything
  after it, two's complement; seq bit 7 = "more strips follow".

Decoder design (what makes it tolerant of paper scans)
------------------------------------------------------
No hard-coded scale. Everything is measured from the strip itself:
  tilt        dominant gradient direction mod 90 degrees, then rotate once
  nibbles     horizontal sync run count; cell pitch from a least-squares fit of its edges
  x position  start-bar edge tracked per pixel row (follows residual tilt, shear, curl)
  cell pitch  start-bar edge to rack edge per row (follows anisotropic scale, keystone)
  rows        the checkerboard flips at EVERY row boundary, so zero crossings of
              (cell4 - cell3) are the row boundaries; the rack gives the right side.
              Boundaries are tracked individually, so row-pitch drift is followed
              and a missed/spurious crossing is repaired by pitch interpolation.
  cells       box-averaged on the measured grid (parallelogram per row), dibit decided
              by left-vs-right contrast.
  errors      per-row parity; failing rows are re-sampled on a small grid of offsets,
              then the least-certain dibit of the failing parity group is flipped
              (the same correction the original reader applied). The strip checksum has
              the final say - nothing is accepted on a guess.
"""

__version__ = "01.00.00.00"

import argparse
import io
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
SCAN_MM = 0.0635                      # reader scan step
HSYNC_MM = 28 * SCAN_MM               # horizontal sync height
VSYNC_MM = 56 * SCAN_MM               # minimum vertical sync height
MIN_NIBBLES, MAX_NIBBLES = 4, 12      # reader range (decoder accepts up to 16)
MIN_BIT_MM, MIN_ROW_MM = 0.15, 0.25   # smallest bit the patents / reader specify
MAX_BIT_MM, MAX_ROW_MM = 0.46, 1.00
MIN_STRIP_MM = 15.0                   # shorter strips are padded so pages can be segmented
HEADER_LEN = 16                       # bytes from 00 00 00 through software expansion
PAGES_MM = {"A3": (297, 420), "A4": (210, 297), "A5": (148, 210), "B5": (176, 250),
            "Letter": (215.9, 279.4), "Legal": (215.9, 355.6)}


class StripError(ValueError):
    """Invalid input to the encoder or a structurally invalid strip."""


class StripDecodeError(StripError):
    """The image could not be decoded into a valid strip."""


# --------------------------------------------------------------------------------------
# byte layer
# --------------------------------------------------------------------------------------
def checksum(data: bytes) -> int:
    """Add with carry (end-around), then two's complement."""
    c = 0
    for b in data:
        c = ((c & 0xFF) + b + (c >> 8)) & 0x1FF
    return (0x100 - ((c & 0xFF) + (c >> 8))) & 0xFF


@dataclass
class File:
    name: str
    data: bytes
    cauzin_type: int = 0x00
    os_filetype: int = 0x00
    execute: bool = False


@dataclass
class StripInfo:
    strip_id: bytes
    seq: int
    more: bool
    strip_type: int
    body: bytes
    raw: bytes = b""


def _directory(files, os_type):
    d = bytearray([os_type, len(files)])
    for f in files:
        if not f.name.isascii() or any(ord(c) < 32 for c in f.name):
            raise StripError(f"file name {f.name!r} must be printable ASCII")
        if len(f.data) >= 1 << 24:
            raise StripError("file too large (3-byte length)")
        d += bytes([f.cauzin_type, f.os_filetype]) + len(f.data).to_bytes(3, "little")
        d += f.name.encode("ascii") + bytes([0xFF if f.execute else 0x00, 0x00])
    return bytes(d)


def build_payloads(files, strip_id="STRIP", capacity=1024, os_type=0x00):
    """Split files into strip payloads (each <= capacity bytes)."""
    if not 1 <= len(files) <= 255:
        raise StripError("need 1..255 files")
    if not strip_id.isascii():
        raise StripError("strip id must be ASCII")
    sid = strip_id.encode("ascii")[:6].ljust(6)
    dirent = _directory(files, os_type)
    data = b"".join(f.data for f in files)
    room = capacity - HEADER_LEN
    first = room - len(dirent)
    if first < 0:
        raise StripError(f"file directory ({len(dirent)} B) does not fit on the first strip")
    chunks = [data[:first]] + [data[i:i + room] for i in range(first, len(data), room)]
    if len(chunks) > 127:
        raise StripError(f"{len(chunks)} strips needed, maximum is 127")
    out = []
    for seq, chunk in enumerate(chunks, 1):
        more = 0x80 if seq < len(chunks) else 0
        tail = sid + bytes([seq | more, 0, 0, 0]) + (dirent if seq == 1 else b"") + chunk
        out.append(bytes(3) + (len(tail) + 1).to_bytes(2, "little") + bytes([checksum(tail)]) + tail)
    return out


def parse_payload(p: bytes) -> StripInfo:
    if len(p) < HEADER_LEN or p[:3] != bytes(3):
        raise StripError("bad data sync")
    length = int.from_bytes(p[3:5], "little")
    tail = p[6:5 + length]
    if len(tail) != length - 1 or len(tail) < 10:
        raise StripError("strip truncated")
    if checksum(tail) != p[5]:
        raise StripError("checksum mismatch")
    body = tail[10:]
    if tail[8] & 0x80:           # optional CRC flag (algorithm never published): drop 2 bytes
        body = body[:-2]
    return StripInfo(tail[:6], tail[6] & 0x7F, bool(tail[6] & 0x80), tail[7], body)


def assemble(infos):
    """Strips of ONE sequence (any order) -> (os_type, [File])."""
    strips = sorted(infos, key=lambda s: s.seq)
    if not strips or strips[0].seq != 1:
        raise StripError("strip #1 (with the directory) is missing")
    for a, b in zip(strips, strips[1:]):
        if b.strip_id != a.strip_id:
            raise StripError("strip ids differ inside a sequence")
        if b.seq != a.seq + 1:
            raise StripError(f"strip sequence gap {a.seq} -> {b.seq}")
    if strips[-1].more:
        raise StripError("last strip says more strips follow (missing strips)")
    if any(s.strip_type for s in strips):
        raise StripError("special strip types (key / compressed) are not supported")
    d = strips[0].body
    try:
        os_type, pos, entries = d[0], 2, []
        for _ in range(d[1]):
            ctype, ftype, size = d[pos], d[pos + 1], int.from_bytes(d[pos + 2:pos + 5], "little")
            end = pos + 5
            while d[end] not in (0, 0xFF):
                end += 1
            entries.append((File(d[pos + 5:end].decode("latin-1"), b"", ctype, ftype, d[end] == 0xFF), size))
            pos = end + 2 + d[end + 1]
    except IndexError:
        raise StripError("directory runs past the end of strip 1") from None
    blob = d[pos:] + b"".join(s.body for s in strips[1:])
    if len(blob) < sum(sz for _, sz in entries):
        raise StripError("data short (missing strips?)")
    files, pos = [], 0
    for f, sz in entries:
        f.data = blob[pos:pos + sz]
        pos += sz
        files.append(f)
    return os_type, files


# --------------------------------------------------------------------------------------
# geometry + renderer
# --------------------------------------------------------------------------------------
def to_bits(data):
    a = np.frombuffer(bytes(data), np.uint8)
    return ((a[:, None] >> np.arange(8)) & 1).astype(np.uint8).ravel()   # LSB first


def to_bytes(bits):
    bits = np.asarray(bits, np.uint8)
    bits = bits[: len(bits) // 8 * 8].reshape(-1, 8)
    return (bits << np.arange(8)).sum(1).astype(np.uint8).tobytes()


def hsync_cells(n):
    w = 8 * n + 14
    r = np.zeros(w, bool)
    r[0:2] = r[4:10] = r[w - 12:w - 6] = r[w - 4:w - 2] = True
    for i in range(n - 4):
        r[12 + 4 * i:14 + 4 * i] = True
        r[w - 16 - 4 * i:w - 14 - 4 * i] = True
    return r


def data_rows_cells(bits, first_row_index, n):
    """bits: (R, 4n) -> boolean cell matrix (R, 8n+14), True = ink."""
    R = bits.shape[0]
    w = 8 * n + 14
    cells = np.zeros((R, w), bool)
    k = (np.arange(R) + first_row_index) % 2
    cells[:, 0:2] = True
    cells[:, 3] = k == 0
    cells[:, 4] = k == 1
    lp = bits[:, 1::2].sum(1) & 1
    rp = bits[:, 0::2].sum(1) & 1
    dib = np.concatenate([lp[:, None], bits, rp[:, None]], axis=1).astype(bool)   # (R, 4n+2)
    cells[:, 5:5 + 2 * dib.shape[1]:2] = ~dib
    cells[:, 6:6 + 2 * dib.shape[1]:2] = dib
    cells[:, w - 3:w - 1] = True
    cells[:, w - 1] = cells[:, 4]
    return cells


def height_code(row_mm):
    code = round(row_mm / SCAN_MM * 16)
    if not 0 < code <= 0xFF:
        raise StripError(f"row height {row_mm:.3f} mm is outside the vertical sync range")
    return code


class Layout:
    """Physical strip geometry in whole device pixels (no resampling when printed 1:1)."""

    def __init__(self, nibbles=10, dpi=600, cell_px=4, row_px=6):
        if not MIN_NIBBLES <= nibbles <= 16:
            raise StripError("nibbles must be 4..16")
        if min(dpi, cell_px, row_px) <= 0:
            raise StripError("dpi, cell and row size must be positive")
        self.n, self.dpi, self.cell_px, self.row_px = nibbles, dpi, cell_px, row_px
        self.mm = 25.4 / dpi
        self.cell_mm, self.row_mm = cell_px * self.mm, row_px * self.mm
        self.hsync_px = math.ceil(HSYNC_MM / self.mm)
        self.vsync_rows = math.ceil(VSYNC_MM / self.row_mm)
        self.code = height_code(self.row_mm)
        self.bpr = 4 * nibbles
        self.w_cells = 8 * nibbles + 14

    @property
    def width_mm(self):
        return self.w_cells * self.cell_mm

    def warnings(self):
        w = []
        if self.n > MAX_NIBBLES:
            w.append(f"{self.n} nibbles/row exceeds the original reader's maximum of {MAX_NIBBLES}")
        if self.cell_mm < MIN_BIT_MM or self.row_mm < MIN_ROW_MM:
            w.append(f"bit {self.cell_mm:.3f}x{self.row_mm:.3f} mm is below the specified "
                     f"minimum {MIN_BIT_MM}x{MIN_ROW_MM} mm")
        if self.cell_mm > MAX_BIT_MM or self.row_mm > MAX_ROW_MM:
            w.append("bit is larger than the original reader accepts")
        if not 15.7 <= self.width_mm <= 17.0:
            w.append(f"strip width {self.width_mm:.1f} mm differs from Cauzin's 15.7-17.0 mm")
        return w

    def capacity(self, max_len_mm):
        rows = int((max_len_mm - self.hsync_px * self.mm) / self.row_mm) - self.vsync_rows
        return max(0, rows * self.n // 2)

    def data_row_count(self, payload_len):
        need_min = math.ceil((MIN_STRIP_MM - self.hsync_px * self.mm) / self.row_mm) - self.vsync_rows
        return max(math.ceil(payload_len * 8 / self.bpr), need_min)

    def render(self, payload, margin_px=16, seed=0):
        bits = to_bits(payload)
        rows = self.data_row_count(len(payload))
        fill = np.random.default_rng(seed).integers(0, 2, rows * self.bpr - len(bits), dtype=np.uint8)
        bits = np.concatenate([bits, fill]).reshape(rows, self.bpr)
        vbits = np.tile(to_bits([self.code] * ((self.n + 1) // 2))[: self.bpr], (self.vsync_rows, 1))
        cells = data_rows_cells(np.vstack([vbits, bits]), 0, self.n)
        px = np.vstack([np.repeat(hsync_cells(self.n)[None], self.hsync_px, 0),
                        np.repeat(cells, self.row_px, 0)])
        px = np.repeat(px, self.cell_px, 1)
        px = np.pad(px, margin_px)
        return Image.fromarray(~px)          # mode '1': True = white


def verify_strip_image(img, layout: Layout, margin_px=16):
    """Strict structural check of an *ideal* (undistorted) rendering, by reading pixels back
    on the known grid and testing every format rule. Returns the parsed StripInfo."""
    a = ~np.asarray(img.convert("1"))[margin_px:-margin_px or None, margin_px:-margin_px or None]
    n, w, cp, rp = layout.n, layout.w_cells, layout.cell_px, layout.row_px
    problems = []
    if a.shape[1] != w * cp:
        raise StripError(f"strip is {a.shape[1]} px wide, expected {w * cp}")
    hs = a[layout.hsync_px // 2, cp // 2::cp]
    runs = int(np.sum(hs[1:] & ~hs[:-1]) + hs[0])
    if runs != 2 * n - 4:
        problems.append(f"hsync has {runs} black runs, expected {2 * n - 4}")
    if not np.array_equal(hs, hsync_cells(n)):
        problems.append("hsync pattern differs from the template")
    body = a[layout.hsync_px:]
    nrows = body.shape[0] // rp
    if body.shape[0] % rp:
        problems.append("data area height is not a whole number of rows")
    c = body[rp // 2::rp, cp // 2::cp][:nrows]
    for k in range(nrows):
        r = c[k]
        if not (r[0] and r[1] and not r[2]):
            problems.append(f"row {k}: start bar/space wrong")
        if r[3] == r[4] or r[3] != (k % 2 == 0):
            problems.append(f"row {k}: checkerboard wrong")
        if not (r[w - 3] and r[w - 2] and not r[w - 5] and not r[w - 4]) or r[w - 1] != r[4]:
            problems.append(f"row {k}: rack wrong")
        pairs = r[5:5 + 2 * (4 * n + 2)].reshape(-1, 2)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            problems.append(f"row {k}: illegal dibit")
            continue
        d = pairs[:, 1].astype(np.uint8)
        if d[0] != (d[2:-1:2].sum() & 1) or d[-1] != (d[1:-1:2].sum() & 1):
            problems.append(f"row {k}: parity wrong")
        if len(problems) > 10:
            break
    if not problems:
        d = np.array([[1 if c[k, 6 + 2 * j] else 0 for j in range(1, 4 * n + 1)] for k in range(nrows)], np.uint8)
        want = to_bits([layout.code] * ((n + 1) // 2))[:4 * n]
        if not all(np.array_equal(d[k], want) for k in range(layout.vsync_rows)):
            problems.append("vertical sync rows do not repeat the row-height code")
        stream = to_bytes(d[layout.vsync_rows:].ravel())
        try:
            info = parse_payload(stream)
        except StripError as e:
            problems.append(f"byte stream: {e}")
    if problems:
        raise StripError("; ".join(problems[:6]))
    return info


# --------------------------------------------------------------------------------------
# image analysis primitives
# --------------------------------------------------------------------------------------
def _otsu(g):
    h, _ = np.histogram(np.clip(g, 0, 255).astype(np.uint8), 256, (0, 256))
    h = h.astype(np.float64)
    tot = h.sum()
    if tot == 0:
        return 128.0
    cum, cumx = np.cumsum(h), np.cumsum(h * np.arange(256))
    w0, w1 = cum, tot - cum
    with np.errstate(divide="ignore", invalid="ignore"):
        m0, m1 = cumx / w0, (cumx[-1] - cumx) / w1
        var = w0 * w1 * (m0 - m1) ** 2
    var = np.nan_to_num(var)
    # ties (e.g. a pure 0/255 image: every threshold between them is equally good): take the middle
    return float(np.flatnonzero(var >= var.max() * (1 - 1e-9)).mean())


class _Box:
    """Exact box means at fractional pixel-edge coordinates via an integral image."""

    def __init__(self, a):
        self.H, self.W = a.shape
        self.I = np.zeros((self.H + 1, self.W + 1))
        self.I[1:, 1:] = a.astype(np.float64).cumsum(0).cumsum(1)

    def _at(self, x, y):
        x = np.clip(x, 0, self.W)
        y = np.clip(y, 0, self.H)
        x0 = np.minimum(np.floor(x).astype(np.intp), self.W - 1)
        y0 = np.minimum(np.floor(y).astype(np.intp), self.H - 1)
        fx, fy = x - x0, y - y0
        I = self.I
        return (I[y0, x0] * (1 - fx) * (1 - fy) + I[y0, x0 + 1] * fx * (1 - fy)
                + I[y0 + 1, x0] * (1 - fx) * fy + I[y0 + 1, x0 + 1] * fx * fy)

    def mean(self, x0, y0, x1, y1):
        x0, y0, x1, y1 = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (x0, y0, x1, y1)))
        s = self._at(x1, y1) - self._at(x0, y1) - self._at(x1, y0) + self._at(x0, y0)
        return s / np.maximum((x1 - x0) * (y1 - y0), 1e-9)


def _runs(mask):
    """(start, end) index pairs of True runs, end exclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _longest_run(mask, bridge):
    runs = _runs(mask)
    if not runs:
        return None
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= bridge:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return max(merged, key=lambda r: r[1] - r[0])


def _cell_estimate(ink):
    """Most common black run length along rows ~ cell width in pixels."""
    rows = ink[:: max(1, ink.shape[0] // 300)]
    pad = np.pad(rows.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(pad, axis=1)
    s, e = np.argwhere(d == 1), np.argwhere(d == -1)
    if len(s) == 0:
        return 0.0
    lens = (e[:, 1] - s[:, 1])
    lens = lens[(lens >= 2) & (lens <= 80)]
    if lens.size == 0:
        return 0.0
    return float(np.bincount(lens).argmax())


def _tilt_once(a):
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1] = a[2:] - a[:-2]
    m2 = gx * gx + gy * gy
    z = (m2 * np.exp(4j * np.arctan2(gy, gx))).sum()
    return 0.0 if abs(z) < 1e-9 else float(np.degrees(np.angle(z)) / 4.0)


def estimate_tilt(g, white=None):
    """Degrees to pass to PIL's rotate() so axis-aligned edges become horizontal/vertical.

    Dominant gradient direction modulo 90 degrees (a strip is all right angles). One pass of this
    estimator has a ~12% gain error, so it is iterated on a (downsampled) working copy until the
    residual is negligible; the caller rotates the full-size image once by the returned total."""
    g = np.asarray(g)
    f = int(math.ceil(max(g.shape) / 1800))
    if f > 1:
        h, w = g.shape[0] // f * f, g.shape[1] // f * f
        a = g[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3), dtype=np.float32)
    else:
        a = g.astype(np.float32)
    fill = float(np.percentile(a, 95)) if white is None else white
    total = 0.0
    for _ in range(4):
        step = _tilt_once(_rotate(a, total, fill))
        total += step
        if abs(step) < 0.01:
            break
    return total


def _rotate(g, deg, fill):
    if abs(deg) < 5e-3:
        return g
    im = Image.fromarray(np.ascontiguousarray(g, np.float32), mode="F")
    im = im.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=float(fill))
    return np.asarray(im, np.float32)


def _smooth(a, med, avg):
    """NaN-aware median + moving average with edge handling."""
    a = np.asarray(a, np.float64).copy()
    ok = np.isfinite(a)
    if ok.sum() < 3:
        return a
    idx = np.arange(len(a))
    a[~ok] = np.interp(idx[~ok], idx[ok], a[ok])
    def run(v, k, f):
        k = max(1, k | 1)
        if k == 1:
            return v
        p = np.pad(v, k // 2, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(p, k)
        return f(win, axis=1)
    return run(run(a, med, np.median), avg, np.mean)


def _edge_pass(box, ys, pred, polarity, win_px, half_h, step=0.5):
    """Vectorised: sub-pixel x of the edge crossing 0.5 nearest to pred[i] for every ys[i]."""
    rel = np.arange(-win_px, win_px + 1e-9, step)
    xs = pred[:, None] + rel[None, :]
    v = box.mean(xs - 0.75, ys[:, None] - half_h, xs + 0.75, ys[:, None] + half_h)
    if polarity < 0:
        cond = (v[:, :-1] > 0.5) & (v[:, 1:] <= 0.5)
    else:
        cond = (v[:, :-1] < 0.5) & (v[:, 1:] >= 0.5)
    xc = xs[:, :-1] + step * (v[:, :-1] - 0.5) / (v[:, :-1] - v[:, 1:] + 1e-12)
    dist = np.where(cond, np.abs(xc - pred[:, None]), np.inf)
    j = np.argmin(dist, axis=1)
    out = xc[np.arange(len(ys)), j]
    out[~np.isfinite(dist[np.arange(len(ys)), j])] = np.nan
    return out


def _track_edge(box, ys, x_start, polarity, win_px, half_h=1.0, stride=4):
    """Sub-pixel x of a vertical edge (crossing 0.5 in the normalised image) for every y.
    polarity -1: paper->ink (value falls); +1: ink->paper.
    Coarse sequential tracking every `stride` rows (robust to tilt), then one dense vectorised
    refinement around the interpolated path."""
    ys = np.asarray(ys, np.float64)
    idx = np.arange(0, len(ys), stride)
    cur, coarse = x_start, np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        r = _edge_pass(box, ys[i:i + 1], np.array([cur]), polarity, win_px, half_h * stride / 2 + 0.5)[0]
        if np.isfinite(r):
            cur = coarse[k] = r
    ok = np.isfinite(coarse)
    if ok.sum() < 2:
        return np.full(len(ys), np.nan)
    pred = np.interp(ys, ys[idx][ok], coarse[ok])
    return _edge_pass(box, ys, pred, polarity, max(1.5, 0.6 * win_px), half_h)


def _crossings(y, d, thr, hyst):
    """Zero crossings of d-thr with hysteresis; y = sample positions."""
    s = d - thr
    out, state, last_old = [], 0, -1
    for i, v in enumerate(s):
        new = 1 if v > hyst else (-1 if v < -hyst else 0)
        if state and s[i] * state > 0:
            last_old = i
        if new == 0:
            continue
        if state == 0:
            state, last_old = new, i
            continue
        if new != state:
            j = last_old
            if j < 0 or j + 1 >= len(s):
                j = max(0, i - 1)
            den = s[j + 1] - s[j]
            t = 0.5 if abs(den) < 1e-12 else -s[j] / den
            out.append(y[j] + float(np.clip(t, 0, 1)) * (y[j + 1] - y[j]))
            state, last_old = new, i
    return out


def _regularize(c):
    """Turn raw crossings into one boundary per row: fill missed ones, drop spurious ones."""
    if len(c) < 3:
        return list(c)
    P = float(np.median(np.diff(c)))
    out = [c[0]]
    for v in c[1:]:
        g = v - out[-1]
        m = int(round(g / P))
        if m < 1:
            continue
        for q in range(1, m):
            out.append(out[-1] + g / m)
        out.append(v)
        if m == 1:
            P = 0.8 * P + 0.2 * g
    return out


def _band_end(blk):
    """Row index where rows stop resembling the first rows (None if never)."""
    ref = blk[1:4].mean(0)
    ref = ref - ref.mean()
    rz = blk - blk.mean(1, keepdims=True)
    corr = (rz @ ref) / (np.linalg.norm(rz, axis=1) * np.linalg.norm(ref) + 1e-9)
    corr = np.convolve(corr, np.ones(3) / 3, mode="same")
    drop = np.flatnonzero(corr[4:] < 0.8)
    return None if drop.size == 0 else float(drop[0] + 4)


@dataclass
class StripRead:
    payload: bytes
    nibbles: int
    rows: int
    bad_rows: int = 0         # rows that failed parity at first sampling
    unrepaired_rows: int = 0
    code: int = 0
    cell_px: float = 0.0
    row_px: float = 0.0
    notes: list = field(default_factory=list)


_OFFSETS = sorted(((dx, dy, ds) for dx in (0, -0.12, 0.12, -0.25, 0.25, -0.38, 0.38)
                   for dy in (0, -0.12, 0.12) for ds in (0, -0.012, 0.012)),
                  key=lambda t: abs(t[0]) + abs(t[1]) + 10 * abs(t[2]))


def _decode_upright(gray, refine=True):
    """Decode one strip from an (almost) upright greyscale array. Raises StripDecodeError."""
    g = np.asarray(gray, np.float32)
    t = _otsu(g)
    hi, lo = g[g > t], g[g <= t]
    if hi.size < 50 or lo.size < 50:
        raise StripDecodeError("no ink / no paper in image")
    wm, bm = float(hi.mean()), float(lo.mean())
    if wm - bm < 25:
        raise StripDecodeError("contrast too low")
    nrm = np.pad((g - bm) / (wm - bm), 12, constant_values=1.0)
    box = _Box(nrm)
    ink = nrm < 0.5

    # ---- locate the strip -------------------------------------------------------------
    cellest = _cell_estimate(ink)
    if cellest < 2:
        raise StripDecodeError("cannot estimate cell size")
    bridge = int(5 * cellest)
    rowprof = ink.sum(1)
    if rowprof.max() == 0:
        raise StripDecodeError("empty")
    ry = _longest_run(rowprof > 0.12 * np.percentile(rowprof[rowprof > 0], 90), bridge)
    ys, ye = ry
    colprof = ink[ys:ye].sum(0)
    rx = _longest_run(colprof > 0.12 * np.percentile(colprof[colprof > 0], 90), bridge)
    xs, xe = rx
    H = ye - ys
    if H < 12 or xe - xs < 20:
        raise StripDecodeError("strip too small")

    # start bar: leftmost run of columns that are ink in (almost) every row
    cp = ink[ys:ye, xs:min(xe, xs + int(12 * cellest) + 8)].sum(0)
    bar = None
    for frac in (0.85, 0.65, 0.45):
        r = _runs(cp >= frac * H)
        if r:
            bar = r[0]
            break
    if bar is None:
        raise StripDecodeError("start bar not found")
    # the strip's true vertical extent is the run of rows in which the start bar is inked: text
    # printed just above/below the strip (labels, captions) must not be mistaken for its top
    b0c, b1c = xs + bar[0], xs + bar[1]
    bw_ = max(2, b1c - b0c)
    inner = ink[:, max(0, b0c - bw_ // 2): b1c + bw_ // 2]   # bar +- half its width: tolerates curl/tilt
    vr = _longest_run(inner.mean(1) > 0.35, int(2 * cellest) + 1)
    if vr is None or vr[1] - vr[0] < 12:
        raise StripDecodeError("start bar not found")
    ys, ye = int(vr[0]), int(vr[1])
    H = ye - ys

    # sub-pixel bar edges from the grey-level column profile of the TOP band only (residual tilt
    # has not accumulated there; binary widths are too coarse at 4 px/cell)
    c0 = max(0, xs - 3)
    c1 = min(nrm.shape[1], xs + bar[1] + int(2 * cellest) + 6)
    top = min(ye, ys + max(8, int(30 * cellest)))
    m = nrm[ys:top, c0:c1].mean(0)

    def cross(i, falling):
        a, b = m[i - 1], m[i]
        return c0 + (i - 1) + 0.5 + (a - 0.5) / (a - b + 1e-12) if falling else \
            c0 + (i - 1) + 0.5 + (0.5 - a) / (b - a + 1e-12)

    fi = np.flatnonzero((m[:-1] >= 0.5) & (m[1:] < 0.5))
    if fi.size == 0 or fi[0] + 1 >= len(m):
        raise StripDecodeError("start bar not found")
    xl = cross(fi[0] + 1, True)
    ri = np.flatnonzero((m[:-1] < 0.5) & (m[1:] >= 0.5))
    ri = ri[ri > fi[0]]
    if ri.size == 0:
        raise StripDecodeError("start bar not found")
    xr_ = cross(ri[0] + 1, False)
    A0 = float(xl)
    p0 = max(1.5, (xr_ - xl) / 2.0)

    # ---- start-bar edge per pixel row ----------------------------------------------------
    yy = np.arange(ys, ye) + 0.5
    A_raw = _track_edge(box, yy, A0, -1, max(2.5, 1.5 * p0))
    if np.isfinite(A_raw).mean() < 0.6:
        raise StripDecodeError("start bar edge lost")
    A = _smooth(A_raw, 9, 9)
    if refine:                      # measure residual tilt from the bar edge and re-rotate once
        okm = np.isfinite(A_raw)
        slope = np.polyfit(yy[okm], A_raw[okm], 1)[0]
        phi = -math.degrees(math.atan(slope))
        if abs(phi) > 0.012:
            return _decode_upright(_rotate(g, phi, wm), refine=False)

    def Aat(y):
        return np.interp(y, yy, A)

    # ---- first row boundary (end of horizontal sync) ----------------------------------------
    def dL_signal(pp):
        x3, x4 = A + 3.5 * pp, A + 4.5 * pp
        hw = 0.3 * pp
        v3 = box.mean(x3 - hw, yy - 0.5, x3 + hw, yy + 0.5)
        v4 = box.mean(x4 - hw, yy - 0.5, x4 + hw, yy + 0.5)
        return v4 - v3

    # End of the horizontal-sync band, found WITHOUT knowing the cell pitch (which is still only a
    # rough estimate here): every pixel row of the band is identical, so the band ends where a row
    # stops correlating with the rows at the very top of the strip.
    lim = min(H // 2, 600)
    if lim < 10:
        raise StripDecodeError("horizontal sync not found")
    yl = np.arange(ys, ys + lim) + 0.5
    # only the left ~11 cells (start bar, gap, wide block): there a slanted band edge moves < 1 px
    xsm = np.interp(yl, yy, A)[:, None] + np.arange(-1.0, 11.0 * p0, 1.0)[None, :]
    # rows are aligned on the tracked bar edge first, so shear / residual tilt cannot decorrelate them
    blk = box.mean(xsm - 0.5, yl[:, None] - 0.5, xsm + 0.5, yl[:, None] + 0.5)
    ref = blk[1:4].mean(0)
    ref = ref - ref.mean()
    rz = blk - blk.mean(1, keepdims=True)
    corr = (rz @ ref) / (np.linalg.norm(rz, axis=1) * np.linalg.norm(ref) + 1e-9)
    corr = np.convolve(corr, np.ones(3) / 3, mode="same")
    drop = np.flatnonzero(corr[4:] < 0.8)
    if drop.size == 0:
        raise StripDecodeError("horizontal sync not found")
    hs_h = float(drop[0] + 4)
    b0 = ys + hs_h
    if hs_h < 5:
        raise StripDecodeError("horizontal sync not found")

    # ---- row slant (paper shear / feed skew) -------------------------------------------------
    # The strip's TOP edge is unique (the row pattern itself is periodic) and independent of the
    # slant direction: compare where ink first appears in a window at the left-hand end with one at
    # the right-hand end. Only ink presence matters, so the rough pitch is good enough here.
    Wpx = float(xe - A0)
    up = min(ys, int(0.04 * Wpx) + 3)                      # supports ~2.3 deg; stays inside label clearance
    yl2 = np.arange(ys - up, ys + lim) + 0.5
    Ay2 = np.interp(yl2, yy, A)[:, None]

    def top_edge(urel):
        xw = Ay2 + urel[None, :]
        m = box.mean(xw - 0.5, yl2[:, None] - 0.5, xw + 0.5, yl2[:, None] + 0.5).mean(1)
        i = np.flatnonzero(m < 0.75)
        if i.size == 0 or i[0] == 0:
            return None
        k = i[0]
        return float(yl2[k - 1] + (m[k - 1] - 0.75) / (m[k - 1] - m[k] + 1e-12))

    tl = top_edge(np.arange(-1.0, 11.0 * p0, 1.0))
    tr = top_edge(Wpx + np.arange(-13.0 * p0, -2.0 * p0, 1.0))
    slope = (tr - tl) / max(1.0, Wpx - 12.5 * p0) if (tl is not None and tr is not None) else 0.0
    if abs(slope) * Wpx > 0.4 * lim:
        slope = 0.0

    # ---- horizontal sync -> nibbles and cell pitch -------------------------------------------
    band = np.arange(ys + 0.2 * hs_h, ys + 0.8 * hs_h)
    band = band[band + 0.5 < b0] if band.size else band
    if band.size == 0:
        band = np.array([ys + hs_h / 2.0])
    u = np.arange(-3.0, (xe - xs) + 3.0, 0.5)
    yb = band + 0.5
    xs_ = np.interp(yb, yy, A)[:, None] + u[None, :]
    ys_ = yb[:, None] + slope * u[None, :]                   # sample along the slanted rows
    prof = box.mean(xs_ - 0.5, ys_ - 0.5, xs_ + 0.5, ys_ + 0.5).mean(0)
    edges = []
    for s_, e_ in _runs(prof < 0.5):
        if s_ == 0 or e_ >= len(u):
            continue
        for i, kind in ((s_, "s"), (e_, "e")):
            a_, b_ = prof[i - 1], prof[i]
            edges.append(u[i - 1] + 0.5 * (a_ - 0.5) / (a_ - b_ + 1e-12))
    R = len(edges) // 2
    if len(edges) % 2 or R < 4 or R % 2:
        raise StripDecodeError(f"horizontal sync unreadable ({len(edges)} edges)")
    n = (R + 4) // 2
    if not MIN_NIBBLES <= n <= 16:
        raise StripDecodeError(f"nibble count {n} out of range")
    hc = hsync_cells(n)
    tpl = []
    for s_, e_ in _runs(hc):
        tpl += [s_, e_]
    tpl = np.array(tpl, float)
    meas = np.array(edges)
    sl, ic = np.polyfit(tpl, meas, 1)
    rms = float(np.sqrt(np.mean((meas - (ic + sl * tpl)) ** 2))) / sl
    if rms > 0.55:
        raise StripDecodeError(f"horizontal sync does not match a {n}-nibble template (rms {rms:.2f} cells)")
    p0 = float(sl)
    w = 8 * n + 14
    delta0 = slope * (w - 4.5) * p0       # slant between the two row-clock columns (left cell 4, right rack)
    # ---- pitch per row from rack edge ------------------------------------------------------------
    dL = dL_signal(p0)
    amp = np.percentile(dL, 90) - np.percentile(dL, 10)
    thrL = 0.5 * (np.percentile(dL, 90) + np.percentile(dL, 10))
    cL = _regularize(_crossings(yy, dL, thrL, 0.2 * amp))
    if len(cL) < 3:
        raise StripDecodeError("no row clock (checkerboard) found")
    b0 = cL[0]
    sel = yy > b0
    B_raw = np.full(len(yy), np.nan)
    if sel.sum() > 3:
        B_raw[sel] = _track_edge(box, yy[sel], Aat(yy[sel][0]) + (w - 3) * p0, -1, max(2.5, 1.5 * p0))
    notes = []
    if np.isfinite(B_raw[sel]).mean() < 0.5:
        pit = np.full(len(yy), p0)
        notes.append("rack edge lost: constant pitch assumed")
    else:
        pit = _smooth((_smooth(B_raw, 9, 9) - A) / (w - 3), 31, 31)
        pit = np.where(np.abs(pit / p0 - 1) > 0.25, p0, pit)

    def pat(y):
        return np.interp(y, yy, pit)

    # right-hand row clock: rack cell (w-1) is ink on odd rows only. With shear the right-hand end of
    # the last rows lies BELOW the strip's left-hand end, so search that much further down.
    P0 = float(np.median(np.diff(cL))) if len(cL) > 2 else 10.0
    yyR = np.arange(int(ys), min(box.H - 1, int(ye + max(0.0, delta0) + 3 * P0))) + 0.5
    pR = np.interp(yyR, yy, pit)
    xr = np.interp(yyR, yy, A) + (w - 0.5) * pR
    vR = box.mean(xr - 0.3 * pR, yyR - 0.5, xr + 0.3 * pR, yyR + 0.5)
    selR = yyR > b0 + min(0.0, delta0)
    ampR = np.percentile(vR[selR], 90) - np.percentile(vR[selR], 10)
    thrR = 0.5 * (np.percentile(vR[selR], 90) + np.percentile(vR[selR], 10))
    crR = _crossings(yyR[selR], vR[selR], thrR, 0.2 * ampR) if ampR > 0.2 else []
    cR = _regularize(crR)

    nrows = len(cL)
    ybot = float(ye)
    last_gap = ybot - cL[-1]
    P_est = float(np.median(np.diff(cL))) if nrows > 2 else last_gap
    if not 0.5 * P_est <= last_gap <= 1.5 * P_est:
        ybot = cL[-1] + P_est
    bL = np.array(cL + [ybot])
    bR = np.full(len(bL), np.nan)
    bR[0] = b0 + delta0
    cRa = np.asarray(cR, float)
    if cRa.size:
        for k in range(1, nrows):
            tgt = bL[k] + delta0
            j = int(np.argmin(np.abs(cRa - tgt)))
            if abs(cRa[j] - tgt) < 0.4 * P_est:
                bR[k] = cRa[j]
    good = np.isfinite(bR[:nrows])
    if good.mean() >= 0.9:
        idx = np.arange(nrows)
        bR[:nrows] = np.interp(idx, idx[good], bR[:nrows][good])
        bR[nrows] = ybot + (bR[nrows - 1] - bL[nrows - 1])
    else:
        notes.append("right row clock unreliable: parallel rows assumed")
        bR = bL + delta0
    topL, botL, topR, botR = bL[:-1], bL[1:], bR[:-1], bR[1:]
    cenL, cenR = (topL + botL) / 2, (topR + botR) / 2
    rowh = ((botL - topL) + (botR - topR)) / 2
    cmid = (cenL + cenR) / 2
    rowA, rowp = Aat(cmid), pat(cmid)
    xl_ref, xr_ref = rowA + 4.0 * rowp, rowA + (w - 0.5) * rowp
    J = 4 * n + 2
    jj = np.arange(J)

    def sample(rows, dx=0.0, dy=0.0, ds=0.0):
        A_, p_ = rowA[rows, None], rowp[rows, None] * (1 + ds)
        xL = A_ + dx * p_ + (5 + 2 * jj[None, :] + 0.5) * p_
        xRr = xL + p_
        frac = (xL - xl_ref[rows, None]) / (xr_ref[rows, None] - xl_ref[rows, None])
        yc = cenL[rows, None] + (cenR[rows, None] - cenL[rows, None]) * frac + dy * rowh[rows, None]
        hw_, hh_ = 0.3 * p_, 0.3 * rowh[rows, None]
        vl = box.mean(xL - hw_, yc - hh_, xL + hw_, yc + hh_)
        vr = box.mean(xRr - hw_, yc - hh_, xRr + hw_, yc + hh_)
        return vl - vr

    def parity_ok(bits):
        d = bits[:, 1:-1]
        return (bits[:, 0] == (d[:, 1::2].sum(1) & 1)) & (bits[:, -1] == (d[:, 0::2].sum(1) & 1))

    allrows = np.arange(nrows)
    q = sample(allrows)
    bits = (q > 0).astype(np.uint8)
    ok = parity_ok(bits)
    bad0 = int((~ok).sum())

    # repair (maximum likelihood): for every failing row consider (a) flipping the least certain
    # dibit of each failing parity group - what the original reader did - and (b) re-sampling the
    # row at small offsets. Of all parity-valid candidates keep the one that overrides the LEAST
    # nominal evidence (sum of |contrast| of the dibits it changes). Choosing by contrast alone
    # would let a shifted sample "fix" parity by corrupting a different dibit.
    fail = np.flatnonzero(~ok)
    if fail.size:
        qn, bn = q[fail], bits[fail].copy()
        conf = np.abs(qn)
        best_cost = np.full(fail.size, np.inf)
        best_bits = bn.copy()
        for fi in range(fail.size):
            row, cost = bn[fi].copy(), 0.0
            d = row[1:-1]
            if row[0] != (d[1::2].sum() & 1):
                cand = np.concatenate([[0], 1 + np.arange(1, 4 * n, 2)])
                i = cand[np.argmin(conf[fi][cand])]
                row[i] ^= 1
                cost += conf[fi][i]
            if row[-1] != (d[0::2].sum() & 1):
                cand = np.concatenate([[J - 1], 1 + np.arange(0, 4 * n, 2)])
                i = cand[np.argmin(conf[fi][cand])]
                row[i] ^= 1
                cost += conf[fi][i]
            best_cost[fi], best_bits[fi] = cost, row
        for off in _OFFSETS[1:]:
            bb = (sample(fail, *off) > 0).astype(np.uint8)
            good = parity_ok(bb)
            cost = (conf * (bb != bn)).sum(1)
            take = good & (cost < best_cost)
            best_cost[take] = cost[take]
            best_bits[take] = bb[take]
        bits[fail] = best_bits
    ok = parity_ok(bits)
    data = bits[:, 1:-1]

    # ---- bytes -------------------------------------------------------------------------------
    first_bytes = np.array([int(to_bytes(data[k, :8])[0]) for k in range(nrows)])
    zeros = np.flatnonzero(first_bytes == 0)
    if zeros.size == 0 or zeros[0] == 0:
        raise StripDecodeError("start of data (00 00 00) not found")
    k0 = int(zeros[0])
    code = int(np.bincount(first_bytes[:k0]).argmax())
    stream = to_bytes(data[k0:].ravel())
    if len(stream) < 6 or stream[:3] != bytes(3):
        raise StripDecodeError("data sync bytes are not 00 00 00")
    length = int.from_bytes(stream[3:5], "little")
    if len(stream) < 5 + length:
        raise StripDecodeError(f"strip truncated: needs {5 + length} bytes, read {len(stream)}")
    payload = stream[:5 + length]
    if checksum(payload[6:]) != payload[5]:
        raise StripDecodeError(f"checksum mismatch ({int((~ok).sum())} parity-failed rows)")
    return StripRead(payload, n, nrows, bad0, int((~ok).sum()), code, float(np.median(rowp)),
                     float(np.median(rowh)), notes)


# --------------------------------------------------------------------------------------
# strip finding / orientation / top-level decode
# --------------------------------------------------------------------------------------
def _split_page(gr):
    """Segment an upright greyscale page into candidate strip crops."""
    t = _otsu(gr)
    ink = gr < t
    cellest = _cell_estimate(ink)
    if cellest < 2:
        return []
    gap = max(3, int(6 * cellest))
    col = ink.sum(0)
    if col.max() == 0:
        return []
    mask = col > 0.1 * np.percentile(col[col > 0], 90)
    segs, cur = [], None
    for s, e in _runs(mask):
        if cur and s - cur[1] <= gap:
            cur[1] = e
        else:
            if cur:
                segs.append(cur)
            cur = [s, e]
    if cur:
        segs.append(cur)
    crops = []
    for xs, xe in segs:
        if xe - xs < 12 * cellest:
            continue
        sub = ink[:, xs:xe]
        row = sub.sum(1)
        m2 = row > 0.1 * np.percentile(row[row > 0], 90)
        prev = None
        for s, e in _runs(m2):
            if prev and s - prev[1] <= gap:
                prev[1] = e
            else:
                if prev and prev[1] - prev[0] >= 12 * cellest:
                    crops.append((xs, xe, prev[0], prev[1]))
                prev = [s, e]
        if prev and prev[1] - prev[0] >= 12 * cellest:
            crops.append((xs, xe, prev[0], prev[1]))
    return crops


def _stage(e):
    m = str(e)
    return 3 if ("checksum" in m or "truncated" in m or "data sync" in m or "start of data" in m) else \
        2 if ("row clock" in m or "horizontal sync" in m or "nibble" in m) else 1


def _try_orientations(crop, order=(0, 180, 90, 270)):
    best = None
    for rot in order:
        c = np.rot90(crop, rot // 90) if rot else crop
        try:
            return _decode_upright(np.ascontiguousarray(c))
        except StripDecodeError as e:
            err = e
        except (IndexError, ValueError, FloatingPointError) as e:     # malformed geometry
            err = StripDecodeError(f"internal: {e}")
        if best is None or _stage(err) > _stage(best):
            best = err
    raise best or StripDecodeError("no strip")


def _rotate8(g8, deg, fill):
    """Rotate a uint8 page (kept 8-bit: an A4 scan at 600 dpi is 35 Mpx)."""
    if abs(deg) < 5e-3:
        return g8
    im = Image.fromarray(np.ascontiguousarray(g8)).rotate(deg, resample=Image.BICUBIC, expand=True,
                                                          fillcolor=int(fill))
    return np.asarray(im)


def decode_image(img):
    """Decode every strip found in a PIL image: one strip, or a whole scanned page with many
    strips (any mix of columns / stacked strips; page or single strips may be rotated)."""
    g = np.asarray(img.convert("L"))
    sub = g[:: max(1, g.shape[0] // 500), :: max(1, g.shape[1] // 500)]
    white = int(np.percentile(sub, 95))
    g = np.pad(g, 8, constant_values=white)
    alpha = estimate_tilt(g)
    gr = _rotate8(g, alpha, white)
    del g
    reads, last_err = [], None
    for page_rot in (0, 90):
        gp = np.rot90(gr) if page_rot else gr
        for xs, xe, ys, ye in _split_page(gp):
            m = 6
            crop = gp[max(0, ys - m):ye + m, max(0, xs - m):xe + m].astype(np.float32)
            try:
                reads.append(_try_orientations(crop))
            except StripDecodeError as e:
                last_err = e
        if reads:
            break
    if not reads:           # segmentation can fail on very damaged strips: try the whole image
        try:
            reads.append(_try_orientations(gr.astype(np.float32)))
        except StripDecodeError as e:
            raise last_err or e
    return reads


def load_images(path):
    """Yield (label, PIL image) for an image file or every page of a PDF (needs PyMuPDF)."""
    if not path.lower().endswith(".pdf"):
        with Image.open(path) as im:
            im.load()
            yield path, im.copy()
        return
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            raise StripError("reading PDFs needs PyMuPDF (pip install pymupdf); "
                             "or export the pages as PNG/TIFF") from None
    doc = fitz.open(path)
    for i, page in enumerate(doc, 1):
        im = None
        imgs = page.get_images(full=True)
        if len(imgs) == 1:                 # scanner PDFs hold one full-page image: use it untouched
            try:
                im = Image.open(io.BytesIO(doc.extract_image(imgs[0][0])["image"]))
                im.load()
            except Exception:              # noqa: BLE001 - fall back to rendering
                im = None
        if im is None:
            pix = page.get_pixmap(dpi=600, colorspace=fitz.csGRAY)
            im = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        yield f"{path}[page {i}]", im


def decode_paths(paths):
    """Decode images / PDFs -> list of sequences, each (os_type, [File]); plus messages.
    Strips of one job may be spread over any number of pages and files, in any order."""
    infos, msgs, found = [], [], 0
    for p in paths:
        try:
            for label, im in load_images(p):
                try:
                    reads = decode_image(im)
                except StripError as e:
                    msgs.append(f"{label}: no strip decoded ({e})")
                    continue
                for r in reads:
                    try:
                        info = parse_payload(r.payload)
                    except StripError as e:
                        msgs.append(f"{label}: {e}")
                        continue
                    info.raw = r.payload
                    infos.append(info)
                    found += 1
                    msgs.append(f"{label}: strip id={info.strip_id.decode('latin-1').strip()!r} "
                                f"seq={info.seq}  {r.nibbles} nibbles, {r.rows} rows, "
                                f"parity-repaired rows {r.bad_rows}")
                if len(reads) > 1:
                    msgs.append(f"{label}: {len(reads)} strips on this image")
        except (StripError, OSError) as e:
            msgs.append(f"{p}: {e}")
    groups = {}
    for i in infos:
        slot = groups.setdefault(i.strip_id, {})
        if i.seq in slot and slot[i.seq].raw != i.raw:
            msgs.append(f"warning: id {i.strip_id!r} strip {i.seq} read twice with different content "
                        "(two jobs sharing an id, or a bad read); keeping the first")
            continue
        slot.setdefault(i.seq, i)                   # a strip scanned twice collapses
    seqs = []
    for sid, by_seq in groups.items():
        try:
            seqs.append(assemble(list(by_seq.values())))
        except StripError as e:
            have = sorted(by_seq)
            msgs.append(f"sequence {sid.decode('latin-1').strip()!r}: {e} (have strips {have})")
    return seqs, msgs


def _safe_name(name, outdir):
    base = "".join(c if c.isalnum() or c in "._-" else "_" for c in os.path.basename(name.replace("\\", "/"))) or "file"
    path, i = os.path.join(outdir, base), 1
    stem, ext = os.path.splitext(base)
    while os.path.exists(path):          # never overwrite
        path = os.path.join(outdir, f"{stem}_{i}{ext}")
        i += 1
    return path


# --------------------------------------------------------------------------------------
# encode orchestration
# --------------------------------------------------------------------------------------
def encode_files(files, layout: Layout, strip_id="STRIP", max_len_mm=240.0, os_type=0):
    cap = layout.capacity(max_len_mm)
    if cap <= HEADER_LEN + 8:
        raise StripError("max strip length leaves no room for data")
    payloads = build_payloads(files, strip_id, min(cap, 65000), os_type)
    return payloads


def render_strips(payloads, layout: Layout, margin_px=16):
    return [layout.render(p, margin_px) for p in payloads]


def page_size_mm(page):
    if page in PAGES_MM:
        return PAGES_MM[page]
    try:
        pw, ph = (float(v) for v in str(page).lower().split("x"))
        return pw, ph
    except ValueError:
        raise StripError(f"unknown page size {page!r} (A3 A4 A5 B5 Letter Legal or WxH in mm)") from None


LABEL_MM = 5.0


def layout_pages(strips, layout: Layout, page="A4", margin_mm=12.0, gap_mm=5.0, labels=None):
    """Pack strips onto pages at true size. Strips fill a column top to bottom (several short
    strips stack in one column), then the next column, then the next page. Returns PIL '1' pages
    whose pixel size matches the paper at the layout's dpi."""
    from PIL import ImageDraw, ImageFont
    pw, ph = page_size_mm(page)
    mmpx = layout.mm
    W, H = round(pw / mmpx), round(ph / mmpx)
    mg, gp = round(margin_mm / mmpx), round(gap_mm / mmpx)
    lab = round(LABEL_MM / mmpx) if labels else 0
    pages, notes = [], []                  # notes[page] = [(x, y, text, max_width)]
    cur = None
    x = y = colw = 0
    for i, s in enumerate(strips):
        a = np.asarray(s.convert("1"))
        ink = ~a
        rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
        a = a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        h, w = a.shape
        if h + lab > H - 2 * mg or w > W - 2 * mg:
            raise StripError("strip does not fit on the page; use a smaller --max-length (or --fill)")
        if cur is not None and y + h + lab > H - mg:       # column full -> next column
            x += colw + gp
            y, colw = mg, 0
            if x + w > W - mg:
                cur = None
        if cur is None:
            cur = np.ones((H, W), bool)
            pages.append(cur)
            notes.append([])
            x = y = mg
            colw = 0
        if labels:
            notes[-1].append((x, y, labels[i], w))
        cur[y + lab:y + lab + h, x:x + w] = a
        y += h + lab + gp
        colw = max(colw, w)
    out = []
    for arr, nl in zip(pages, notes):
        im = Image.fromarray(arr)
        if nl:
            d = ImageDraw.Draw(im)
            size = max(8, int(lab * 0.62))
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:
                font = ImageFont.load_default()
            for lx, ly, text, mw in nl:
                while len(text) > 1 and d.textlength(text, font=font) > mw:
                    text = text[:-1]
                d.text((lx, ly + lab // 8), text, fill=0, font=font)
        out.append(im)
    return out


def save_pdf(pages, path, dpi):
    """Ready-to-print PDF: one 1-bit (CCITT G4) image per page at its native resolution, page size
    = pixels / dpi, so a viewer's 'actual size' prints every bit at its designed size."""
    pages[0].save(path, "PDF", resolution=float(dpi), save_all=True, append_images=pages[1:])


def to_text(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") + b"\x1a"


# --------------------------------------------------------------------------------------
# scan simulator (distortion test bench)
# --------------------------------------------------------------------------------------
def _gauss_kernel(sigma):
    if sigma < 0.05:
        return np.array([1.0])
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


def _blur_axis(a, sigma, axis):
    k = _gauss_kernel(sigma)
    if len(k) == 1:
        return a
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="edge")
    n = a.shape[axis]
    out = np.zeros_like(a)
    for i, kv in enumerate(k):
        out += kv * (p[:, i:i + n] if axis == 1 else p[i:i + n])
    return out


def _bilinear(a, x, y):
    H, W = a.shape
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    out = np.zeros(x.shape, np.float32)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            out += np.where(ok, a[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)], 0) * wx * wy
    return out


def simulate_scan(img, layout=None, *, scale=(1.0, 1.0), angle=0.0, shear=0.0, keystone=0.0,
                  drift=0.0, curl=0.0, dot_gain=0.0, blur=0.6, noise=0.0, jpeg=None, contrast=1.0,
                  gamma=1.0, gradient=0.0, specks=0.0, margin=30, seed=0):
    """Turn an ideal strip bitmap into something like a scan of a paper print.

    scale     (sx, sy) scanner px per printer px (anisotropic allowed)
    angle     degrees of rotation of the paper in the scanner
    shear     degrees of skew (paper feed not perpendicular)
    keystone  perspective: relative width change over the strip length
    drift     row-pitch error, as a fraction of one row (peak), sinusoidal over the length
    curl      sideways wobble in cells (peak), e.g. paper curling near a spine
    dot_gain  ink spread in source pixels (negative = ink loss)
    blur      scanner optics, sigma in OUTPUT pixels
    """
    rng = np.random.default_rng(seed)
    ink = 1.0 - np.asarray(img.convert("L"), np.float32) / 255.0
    Hs, Ws = ink.shape
    cell_px = layout.cell_px if layout else 4
    row_px = layout.row_px if layout else 6
    # dot gain: fractional dilation / erosion
    if dot_gain:
        from PIL import ImageFilter
        im = Image.fromarray((ink * 255).astype(np.uint8))
        steps = int(abs(dot_gain))
        frac = abs(dot_gain) - steps
        f = ImageFilter.MaxFilter if dot_gain > 0 else ImageFilter.MinFilter
        cur = im
        for _ in range(steps):
            cur = cur.filter(f(3))
        nxt = cur.filter(f(3))
        ink = (np.asarray(cur, np.float32) * (1 - frac) + np.asarray(nxt, np.float32) * frac) / 255.0
    sx, sy = scale
    if min(sx, sy) < 1.0:                           # prefilter: scanner integrates over its aperture
        ink = _blur_axis(_blur_axis(ink, 0.45 / sx, 1), 0.45 / sy, 0)
    D, C = drift * row_px, curl * cell_px
    th, sh = math.radians(angle), math.tan(math.radians(shear))
    ca, sa = math.cos(th), math.sin(th)

    def fwd(x, y):                                   # source (centred) -> output (centred)
        yy = y + D * np.sin(math.pi * (y + Hs / 2) / Hs)
        xx = x + C * np.sin(math.pi * (y + Hs / 2) / Hs)
        xx = xx + sh * yy
        xx, yy = xx * sx, yy * sy
        xx = xx / (1 + keystone * (yy / (Hs * sy)))
        return xx * ca - yy * sa, xx * sa + yy * ca

    cx, cy = np.array([-Ws / 2, Ws / 2, -Ws / 2, Ws / 2]), np.array([-Hs / 2, -Hs / 2, Hs / 2, Hs / 2])
    ox, oy = fwd(cx, cy)
    Wo, Ho = int(math.ceil(ox.max() - ox.min())) + 2 * margin, int(math.ceil(oy.max() - oy.min())) + 2 * margin
    ux, uy = np.meshgrid(np.arange(Wo) - Wo / 2 + 0.5, np.arange(Ho) - Ho / 2 + 0.5)
    # inverse
    xr = ux * ca + uy * sa
    yr = -ux * sa + uy * ca
    xr = xr * (1 + keystone * (yr / (Hs * sy)))
    xr, yr = xr / sx, yr / sy
    xr = xr - sh * yr
    y = yr.copy()
    for _ in range(5):
        y = yr - D * np.sin(math.pi * (y + Hs / 2) / Hs)
    x = xr - C * np.sin(math.pi * (y + Hs / 2) / Hs)
    out = _bilinear(ink, (x + (Ws - 1) / 2).astype(np.float32), (y + (Hs - 1) / 2).astype(np.float32))
    if blur > 0:
        out = _blur_axis(_blur_axis(out, blur, 1), blur, 0)
    v = 235.0 - (235.0 - 25.0) * out                 # paper 235, ink 25 (not pure black/white)
    v = 128 + (v - 128) * contrast
    v = 255 * np.clip(v / 255, 0, 1) ** gamma
    if gradient:
        v = v * (1 + gradient * ((np.arange(Wo)[None, :] / Wo - 0.5) * 2 + (np.arange(Ho)[:, None] / Ho - 0.5)) / 2)
    if noise:
        v = v + rng.normal(0, 255 * noise, v.shape)
    if specks:
        k = int(specks * v.size / 1e6)
        py, px = rng.integers(0, Ho, k), rng.integers(0, Wo, k)
        v[py, px] = rng.choice([0, 255], k)
    v = np.clip(v, 0, 255).astype(np.uint8)
    im = Image.fromarray(v)
    if jpeg:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(jpeg))
        buf.seek(0)
        im = Image.open(buf).convert("L")
    return im


STRESS_LAYOUTS = [
    dict(nibbles=10, cell_px=4, row_px=6),    # default (0.169 x 0.254 mm @600 dpi)
    dict(nibbles=5, cell_px=8, row_px=14),    # Cauzin STRIPPER-like, odd nibble count
    dict(nibbles=7, cell_px=6, row_px=9),
    dict(nibbles=4, cell_px=5, row_px=8),
    dict(nibbles=12, cell_px=4, row_px=7),
]

STRESS_SCENARIOS = [
    ("clean (identity)", {}),
    ("scale 0.60x", dict(scale=(0.6, 0.6))),
    ("scale 0.75x", dict(scale=(0.75, 0.75))),
    ("scale 0.83x (500 dpi scan of 600 dpi print)", dict(scale=(0.8333, 0.8333))),
    ("scale 1.00x", dict(scale=(1.0, 1.0))),
    ("scale 1.37x", dict(scale=(1.37, 1.37))),
    ("scale 2.00x (1200 dpi scan)", dict(scale=(2.0, 2.0))),
    ("scale 3.17x", dict(scale=(3.17, 3.17))),
    ("anisotropic x1.00 y1.12", dict(scale=(1.0, 1.12))),
    ("anisotropic x0.90 y1.10", dict(scale=(0.9, 1.1))),
    ("anisotropic x1.30 y0.85", dict(scale=(1.3, 0.85))),
    ("rotate +0.5 deg", dict(angle=0.5)),
    ("rotate -1.0 deg", dict(angle=-1.0)),
    ("rotate +2.5 deg", dict(angle=2.5)),
    ("rotate -5 deg", dict(angle=-5.0)),
    ("rotate +10 deg", dict(angle=10.0)),
    ("rotate 90 deg", dict(angle=90.0)),
    ("rotate 180 deg", dict(angle=180.0)),
    ("rotate 270 deg + 1.3 deg", dict(angle=271.3)),
    ("shear 1 deg", dict(shear=1.0)),
    ("keystone 2%", dict(keystone=0.02)),
    ("row-pitch drift 0.4 row", dict(drift=0.4)),
    ("row-pitch drift 0.8 row", dict(drift=0.8)),
    ("curl 1.0 cell", dict(curl=1.0)),
    ("blur 1.0 px", dict(blur=1.0)),
    ("noise 6%", dict(noise=0.06)),
    ("noise 12%", dict(noise=0.12)),
    ("JPEG q40", dict(jpeg=40)),
    ("contrast 45%", dict(contrast=0.45)),
    ("gamma 1.6", dict(gamma=1.6)),
    ("gamma 0.6", dict(gamma=0.6)),
    ("illumination gradient 25%", dict(gradient=0.25)),
    ("dot gain +0.5 px", dict(dot_gain=0.5)),
    ("dot gain -0.5 px (ink loss)", dict(dot_gain=-0.5)),
    ("dust specks (150/Mpx)", dict(specks=150)),
    ("REALISTIC: 0.8 deg, x0.83/y0.86, blur, noise, JPEG, gain, drift, gradient",
     dict(angle=0.8, scale=(0.83, 0.86), blur=0.8, noise=0.05, jpeg=70, dot_gain=0.4,
          drift=0.3, gradient=0.15, keystone=0.004, shear=0.2)),
    ("HARSH: 3 deg, x0.7/y0.75, blur 1, noise 8%, JPEG 50, drift 0.5, curl 0.8",
     dict(angle=3.0, scale=(0.7, 0.75), blur=1.0, noise=0.08, jpeg=50, drift=0.5, curl=0.8,
          dot_gain=0.3, gradient=0.2)),
]


def _random_files(rng, max_size=1400):
    size = int(min(rng.integers(120, 1400), max_size))
    kind = rng.integers(0, 4)
    if kind == 0:
        data = bytes(rng.integers(0, 256, size, dtype=np.uint8))
    elif kind == 1:
        data = bytes(size)                                   # all zero (worst case for clocking)
    elif kind == 2:
        data = b"\xff" * size
    else:
        data = (b"The quick brown fox jumps over the lazy dog. " * 60)[:size]
    return [File("T%03d.BIN" % rng.integers(0, 999), data)]


def run_stress(trials=5, seed=7, quick=False, oracle=True, only=None, out=print):
    rng = np.random.default_rng(seed)
    scenarios = [s for s in STRESS_SCENARIOS if (only is None or only.lower() in s[0].lower())]
    if quick:
        scenarios = [s for i, s in enumerate(scenarios) if i % 2 == 0 or s[0].startswith(("REAL", "HARSH"))]
        trials = min(trials, 3)
    out(f"{'scenario':<78} {'ok':>6}  {'avg px/cell':>11}  {'repairs':>7}")
    tot_ok = tot = 0
    results = []
    for name, par in scenarios:
        ok = 0
        pxc = []
        repairs = 0
        fails = []
        for t in range(trials):
            lay = Layout(**STRESS_LAYOUTS[(t + len(results)) % len(STRESS_LAYOUTS)])
            files = _random_files(rng, lay.capacity(240) - 60)
            payloads = encode_files(files, lay, "TESTID", 240)
            img = lay.render(payloads[0])
            sc = simulate_scan(img, lay, seed=int(rng.integers(1 << 30)), **par)
            pxc.append(lay.cell_px * (par.get("scale", (1, 1))[0]))
            try:
                reads = decode_image(sc)
                good = False
                for r in reads:
                    _, got = assemble([parse_payload(r.payload)])
                    good |= got[0].data == files[0].data and got[0].name == files[0].name
                    repairs += r.bad_rows
                ok += good
                if not good:
                    fails.append("wrong data")
            except StripError as e:
                fails.append(str(e)[:50])
        tot_ok += ok
        tot += trials
        results.append((name, ok, trials))
        out(f"{name:<78} {ok:>2}/{trials:<3}  {np.mean(pxc):>11.2f}  {repairs:>7}" + (f"   <- {fails[0]}" if fails else ""))
    out(f"\nTOTAL {tot_ok}/{tot} strips decoded byte-exactly")
    return results, tot_ok, tot


def run_pagetest(out=print):
    """Multi-strip pages: clean and scanned, packed in stacked columns, with labels, through PDF."""
    rng = np.random.default_rng(21)
    lay = Layout(10, 300, 2, 3)                       # 300 dpi page, same physical 0.169x0.254 mm bit
    files = [File("A.BIN", bytes(rng.integers(0, 256, 9000, dtype=np.uint8))),
             File("B.TXT", bytes(rng.integers(0, 256, 2500, dtype=np.uint8)))]
    pls = encode_files(files, lay, "PAGE01", 70)
    want = [p for p in pls]
    pages = layout_pages(render_strips(pls, lay, 0), lay, "A5", 10, 5, [f"PAGE01 {i + 1}/{len(pls)}" for i in range(len(pls))])
    bad = 0

    def check(name, imgs):
        nonlocal bad
        got = []
        for im in imgs:
            got += [r.payload for r in decode_image(im)]
        good = sorted(got) == sorted(want)
        bad += not good
        out(f"  {'ok  ' if good else 'FAIL'} {name}: {len(got)}/{len(want)} strips on {len(imgs)} page(s)")

    check("clean pages (labels, stacked columns)", pages)
    sc = [simulate_scan(p, lay, scale=(1.6, 1.6), angle=-1.1, blur=0.7, noise=0.04, jpeg=70, dot_gain=0.3,
                        gradient=0.15, drift=0.2, margin=50, seed=i) for i, p in enumerate(pages)]
    check("scanned pages (tilt, 1.6x, JPEG, noise)", sc)
    check("clean pages rotated 90 deg", [p.rotate(90, expand=True) for p in pages])
    check("clean pages rotated 180 deg", [p.rotate(180) for p in pages])
    try:
        import tempfile
        import pymupdf  # noqa: F401
        with tempfile.TemporaryDirectory() as td:
            fn = os.path.join(td, "t.pdf")
            save_pdf(pages, fn, lay.dpi)
            got = []
            for lab, im in load_images(fn):
                got += [r.payload for r in decode_image(im)]
            good = sorted(got) == sorted(want)
            bad += not good
            out(f"  {'ok  ' if good else 'FAIL'} PDF round trip: {len(got)}/{len(want)} strips")
    except ImportError:
        out("  (PyMuPDF not installed: PDF input test skipped)")
    return bad


def run_selfcheck(out=print):
    """Generator validity: every layout renders a strip that passes the strict structural check
    and whose pixel-exact decode returns the original bytes."""
    rng = np.random.default_rng(3)
    bad = 0
    for cfg in STRESS_LAYOUTS + [dict(nibbles=6, cell_px=3, row_px=5), dict(nibbles=9, cell_px=7, row_px=10)]:
        lay = Layout(**cfg)
        for _ in range(3):
            files = _random_files(rng, lay.capacity(240) - 60)
            pl = encode_files(files, lay, "SELF", 240)[0]
            img = lay.render(pl)
            try:
                info = verify_strip_image(img, lay)
                r = decode_image(img)[0]
                assert r.payload == pl and r.bad_rows == 0
                out(f"  ok  {cfg}  {len(pl)} B  {img.size[0]}x{img.size[1]} px")
            except (StripError, AssertionError) as e:
                bad += 1
                out(f"  FAIL {cfg}: {e}")
    # the validator must actually catch corruption (otherwise 'valid' means nothing)
    lay = Layout()
    pl = encode_files([File("N.BIN", bytes(range(200)))], lay, "NEG", 240)[0]
    good_img = lay.render(pl)
    a = np.asarray(good_img.convert("1")).copy()
    m = 16
    def hit(name, mutate):
        b = a.copy()
        mutate(b)
        try:
            verify_strip_image(Image.fromarray(b), lay)
            out(f"  FAIL validator missed: {name}")
            return 1
        except StripError:
            out(f"  ok  validator catches: {name}")
            return 0
    r0 = m + lay.hsync_px + 20 * lay.row_px                       # a data row
    cx = m + 12 * lay.cell_px
    bad += hit("flipped data dibit (parity)", lambda b: b.__setitem__((slice(r0, r0 + lay.row_px), slice(cx, cx + lay.cell_px)), ~b[r0:r0 + lay.row_px, cx:cx + lay.cell_px]))
    bad += hit("start bar cell removed", lambda b: b.__setitem__((slice(r0, r0 + lay.row_px), slice(m, m + lay.cell_px)), True))
    bad += hit("checkerboard not flipping", lambda b: b.__setitem__((slice(r0, r0 + lay.row_px), slice(m + 3 * lay.cell_px, m + 5 * lay.cell_px)), True))
    bad += hit("hsync stripe missing", lambda b: b.__setitem__((slice(m, m + lay.hsync_px), slice(m + 4 * lay.cell_px, m + 10 * lay.cell_px)), True))
    # multi-strip, multi-file round trip
    lay = Layout()
    files = [File("A.TXT", bytes(rng.integers(0, 256, 5000, dtype=np.uint8))),
             File("B.BIN", bytes(rng.integers(0, 256, 3000, dtype=np.uint8)), 1, 2, True)]
    pls = encode_files(files, lay, "MULTI", 60)
    imgs = render_strips(pls, lay)
    infos = [parse_payload(decode_image(i)[0].payload) for i in reversed(imgs)]   # any order
    _, back = assemble(infos)
    good = [(f.name, f.data, f.execute) for f in back] == [(f.name, f.data, f.execute) for f in files]
    out(f"  {'ok ' if good else 'FAIL'} multi-file / {len(imgs)}-strip sequence, strips decoded in reverse order")
    bad += not good
    return bad


# --------------------------------------------------------------------------------------
# optional interoperability check against an independent implementation (stripgen)
# --------------------------------------------------------------------------------------
def run_interop(out=print):
    try:
        from softstrip import format as ofmt, image as oimg
    except ImportError:
        out("  (independent decoder 'stripgen' not installed: pip install git+https://github.com/Lumenman/stripgen)")
        return None
    rng = np.random.default_rng(11)
    bad = 0
    for cfg in STRESS_LAYOUTS:
        lay = Layout(**cfg)
        files = _random_files(rng, lay.capacity(240) - 60)
        mine = encode_files(files, lay, "INTEROP", 240)[0]
        img = lay.render(mine)
        try:
            theirs, _info = oimg.read(img)
            ok1 = bytes(theirs)[:len(mine)] == mine
        except Exception as e:                     # noqa: BLE001
            ok1, theirs = False, repr(e)
        # reverse: their encoder -> my decoder
        og = oimg.Geometry(nibbles=cfg["nibbles"], dpi=600, cell_px=cfg["cell_px"], row_px=cfg["row_px"])
        of = [ofmt.File("X.BIN", files[0].data)]
        op = ofmt.build(of, "INTEROP", og.capacity(240))[0]
        try:
            r = decode_image(og.render(op))[0]
            ok2 = r.payload == op
        except StripError as e:
            ok2 = False
        out(f"  mine->stripgen {'ok ' if ok1 else 'FAIL'}   stripgen->mine {'ok ' if ok2 else 'FAIL'}   {cfg}")
        bad += (not ok1) + (not ok2)
    return bad


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _cmd_encode(a):
    import hashlib
    import base64
    lay = Layout(a.nibbles, a.dpi, a.cell, a.row)
    for w in lay.warnings():
        print("warning:", w)
    want_pdf = a.format in ("pdf", "both")
    page = a.page or ("A4" if want_pdf else None)
    files = []
    for p in a.files:
        with open(p, "rb") as fh:
            data = fh.read()
        if a.text:
            data = to_text(data)
        files.append(File(os.path.basename(p).upper()[:60] if a.upper else os.path.basename(p), data,
                          execute=a.exec))
    strip_id = a.id or base64.b32encode(hashlib.sha1(
        b"".join(f.name.encode() + f.data for f in files)).digest())[:6].decode()
    max_len = a.max_length
    if a.fill and page:
        pw, ph = page_size_mm(page)
        max_len = ph - 2 * a.margin - (LABEL_MM if a.labels else 0)
        print(f"--fill: strips up to {max_len:.0f} mm so a column uses the full page height")
    if max_len > 228.6:
        print("note: strips longer than ~9 in (228.6 mm) do not fit the original hardware reader's "
              "window; scanner/software decoding is unaffected")
    if page and a.gap < 3:
        print("warning: gaps under 3 mm make strips on a tilted scan hard to separate")
    payloads = encode_files(files, lay, strip_id, max_len, a.os)
    if a.verify:
        for im_ in render_strips(payloads, lay):
            verify_strip_image(im_, lay)
        print("verify: all strips pass the structural check")
    labels = [f"{strip_id} {i}/{len(payloads)}" for i in range(1, len(payloads) + 1)] if a.labels else None
    outs = []
    if page:
        imgs = render_strips(payloads, lay, 0)
        pages = layout_pages(imgs, lay, page, a.margin, a.gap, labels)
        if a.format in ("png", "both"):
            for i, pg in enumerate(pages, 1):
                fn = f"{a.output}.png" if len(pages) == 1 else f"{a.output}-page{i}.png"
                pg.save(fn, dpi=(lay.dpi, lay.dpi))
                outs.append(fn)
        if want_pdf:
            fn = f"{a.output}.pdf"
            save_pdf(pages, fn, lay.dpi)
            outs.append(fn)
        print(f"{len(payloads)} strip(s) on {len(pages)} {page} page(s)")
    else:
        for i, im_ in enumerate(render_strips(payloads, lay), 1):
            fn = f"{a.output}.png" if len(payloads) == 1 else f"{a.output}-{i}.png"
            im_.save(fn, dpi=(lay.dpi, lay.dpi))
            outs.append(fn)
    print(f"strip id {strip_id!r}: {len(payloads)} strip(s), {sum(len(p) for p in payloads)} bytes on strips, "
          f"strip width {lay.width_mm:.1f} mm, bit {lay.cell_mm:.3f} x {lay.row_mm:.3f} mm")
    print("PRINT AT 100% / 'ACTUAL SIZE' (never 'fit to page'):", ", ".join(outs))


def _cmd_decode(a):
    seqs, msgs = decode_paths(a.images)
    for m in msgs:
        print(m)
    if not seqs:
        print("nothing decoded")
        return 1
    os.makedirs(a.dir, exist_ok=True)
    for os_type, files in seqs:
        for f in files:
            path = _safe_name(f.name, a.dir)
            with open(path, "wb") as fh:
                fh.write(f.data)
            print(f"wrote {path} ({len(f.data)} bytes)")
    return 0


def _cmd_verify(a):
    bad = 0
    for p in a.images:
        try:
            for label, im in load_images(p):
                reads = decode_image(im)
                print(f"{label}: decodes, {len(reads)} strip(s), parity-failed rows {[r.bad_rows for r in reads]}")
        except StripError as e:
            print(f"{p}: FAIL {e}")
            bad += 1
    return bad


def _cmd_stress(a):
    print("== generator validity ==")
    bad = run_selfcheck()
    print("\n== multi-strip pages ==")
    bad += run_pagetest()
    print("\n== independent decoder interop ==")
    run_interop()
    print("\n== distortion battery ==")
    t0 = time.time()
    _, ok, tot = run_stress(a.trials, a.seed, a.quick, only=a.only)
    print(f"({time.time() - t0:.0f} s)")
    return 0 if (bad == 0 and ok == tot) else 1


def build_parser():
    ap = argparse.ArgumentParser(description=f"Cauzin Softstrip tool v{__version__}")
    sp = ap.add_subparsers(dest="cmd")
    e = sp.add_parser("encode", help="files -> strip image(s)")
    e.add_argument("files", nargs="+")
    e.add_argument("-o", "--output", default="softstrip")
    e.add_argument("--nibbles", type=int, default=10, help="nibbles per row, 4..12 (default 10)")
    e.add_argument("--dpi", type=int, default=600)
    e.add_argument("--cell", type=int, default=4, help="bit width in device px (default 4)")
    e.add_argument("--row", type=int, default=6, help="bit height in device px (default 6)")
    e.add_argument("--max-length", type=float, default=240.0, help="longest strip in mm")
    e.add_argument("--id", default=None, help="strip id, max 6 chars (default: derived from the files, "
                   "so different jobs never share an id)")
    e.add_argument("--os", type=lambda v: int(v, 0), default=0, help="OS type code")
    e.add_argument("--exec", action="store_true", help="mark files executable")
    e.add_argument("--text", action="store_true", help="CRLF line ends + ^Z (Cauzin generic text)")
    e.add_argument("--upper", action="store_true", help="upper-case file names")
    e.add_argument("--page", help="A3/A4/A5/B5/Letter/Legal or WxH mm: pack strips onto pages "
                   "(columns, short strips stack)")
    e.add_argument("--fill", action="store_true", help="with a page: make strips as long as the page allows")
    e.add_argument("--labels", action="store_true", help="print 'id n/total' above every strip")
    e.add_argument("--margin", type=float, default=12.0)
    e.add_argument("--gap", type=float, default=5.0)
    e.add_argument("--format", choices=["png", "pdf", "both"], default="png",
                   help="pdf = ready-to-print PDF at true size (A4 unless --page given)")
    e.add_argument("--verify", action="store_true", help="run the structural check on every strip")
    d = sp.add_parser("decode", help="scans -> files")
    d.add_argument("images", nargs="+")
    d.add_argument("-d", "--dir", default="decoded")
    v = sp.add_parser("verify", help="check that images decode (checksum + parity)")
    v.add_argument("images", nargs="+")
    s = sp.add_parser("stress", help="validity + distortion/scale test battery")
    s.add_argument("--quick", action="store_true")
    s.add_argument("--trials", type=int, default=5)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--only", help="run scenarios whose name contains this text")
    return ap


def interactive():
    print(f"Custom Softstrip Tool - v{__version__}")
    while True:
        choice = input("choose: encode/decode/verify/stress/exit >>> ").strip().lower()
        try:
            if choice == "encode":
                p = input("input file(s), space separated >>> ").split()
                _cmd_encode(build_parser().parse_args(["encode", *p, "-o", "softstrip", "--verify"]))
            elif choice == "decode":
                p = input("scan image path(s) >>> ").split()
                _cmd_decode(build_parser().parse_args(["decode", *p]))
            elif choice == "verify":
                p = input("image path(s) >>> ").split()
                _cmd_verify(build_parser().parse_args(["verify", *p]))
            elif choice == "stress":
                _cmd_stress(build_parser().parse_args(["stress", "--quick"]))
            elif choice in ("exit", "quit", "q"):
                return 0
        except (StripError, OSError) as e:
            print("error:", e)


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd is None:
        return interactive()
    try:
        return {"encode": _cmd_encode, "decode": _cmd_decode, "verify": _cmd_verify,
                "stress": _cmd_stress}[a.cmd](a) or 0
    except (StripError, OSError) as e:
        print("error:", e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
