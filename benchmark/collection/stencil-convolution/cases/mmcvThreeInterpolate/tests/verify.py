#!/usr/bin/env python3
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

OP = 35
OUT_N = 24

def h01(i, seed=123):
    h = (int(i) * 2654435761 + int(seed) * 2246822519) & 0xFFFFFFFF
    h = (h ^ (h >> 15)) & 0xFFFFFFFF
    h = (h * 2246822519) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) & 0xFFFFFFFF
    return np.float32(h & 0xFFFFFF) / np.float32(0x1000000)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def img_hwc(y, x, ch, h, w, c, seed=123):
    y, x = clamp(int(y), 0, h - 1), clamp(int(x), 0, w - 1)
    return h01((y * w + x) * c + ch, seed)

def img_chw(ch, y, x, c, h, w, seed=123):
    y, x = clamp(int(y), 0, h - 1), clamp(int(x), 0, w - 1)
    return h01((ch * h + y) * w + x, seed)

def bilinear_hwc(y, x, ch, h, w, c, seed=123):
    if x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return np.float32(0)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = clamp(x0 + 1, 0, w - 1), clamp(y0 + 1, 0, h - 1)
    wx, wy = np.float32(x - x0), np.float32(y - y0)
    v00 = img_hwc(y0, x0, ch, h, w, c, seed)
    v01 = img_hwc(y0, x1, ch, h, w, c, seed)
    v10 = img_hwc(y1, x0, ch, h, w, c, seed)
    v11 = img_hwc(y1, x1, ch, h, w, c, seed)
    return (np.float32(1) - wy) * ((np.float32(1) - wx) * v00 + wx * v01) + wy * ((np.float32(1) - wx) * v10 + wx * v11)

def make_box(i, seed):
    cx = np.float32(0.12) + np.float32(0.76) * h01(i * 5 + 0, seed)
    cy = np.float32(0.12) + np.float32(0.76) * h01(i * 5 + 1, seed)
    ww = np.float32(0.10) + np.float32(0.25) * h01(i * 5 + 2, seed)
    hh = np.float32(0.10) + np.float32(0.25) * h01(i * 5 + 3, seed)
    return [max(np.float32(0), cx - np.float32(0.5) * ww), max(np.float32(0), cy - np.float32(0.5) * hh),
            min(np.float32(1), cx + np.float32(0.5) * ww), min(np.float32(1), cy + np.float32(0.5) * hh),
            h01(i * 5 + 4, seed)]

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(np.float32(0), ix2 - ix1), max(np.float32(0), iy2 - iy1)
    inter = iw * ih
    aa = max(np.float32(0), a[2] - a[0]) * max(np.float32(0), a[3] - a[1])
    ab = max(np.float32(0), b[2] - b[0]) * max(np.float32(0), b[3] - b[1])
    return inter / (aa + ab - inter + np.float32(1e-6))

def point3(i, seed):
    return np.asarray([h01(i * 3 + 0, seed), h01(i * 3 + 1, seed), h01(i * 3 + 2, seed)], dtype=np.float32)

def dist3(a, sa, b, sb):
    d = point3(a, sa) - point3(b, sb)
    return np.float32(np.sqrt(np.sum(d * d)))

def reference(meta):
    d = meta["input"]["sizes"] + [1] * 8
    out = np.zeros(OUT_N, dtype=np.float32)
    if OP == 1:
        n, h, w, c = d[:4]; means = [0.45, 0.50, 0.55]; stds = [0.20, 0.25, 0.30]
        for idx in range(OUT_N):
            x = idx % w; y = (idx // w) % h; ch = (idx // (w * h)) % c; b = idx // (w * h * c)
            out[idx] = (h01(((b * h + y) * w + x) * c + ch, 123) - np.float32(means[ch])) / np.float32(stds[ch])
    elif OP == 2:
        n, ih, iw, c, oh, ow = d[:6]
        for idx in range(OUT_N):
            ch = idx % c; ox = (idx // c) % ow; oy = (idx // (c * ow)) % oh
            mx = ow - 1 - ox
            sy = (np.float32(oy + 2) + np.float32(0.5)) * ih / (oh + 4) - np.float32(0.5)
            sx = (np.float32(mx + 3) + np.float32(0.5)) * iw / (ow + 6) - np.float32(0.5)
            out[idx] = bilinear_hwc(sy, sx, ch, ih, iw, c)
    elif OP == 3:
        n, ih, iw, c, oh, ow = d[:6]
        for idx in range(OUT_N):
            ch = idx % c; ox = (idx // c) % ow; oy = (idx // (c * ow)) % oh
            sy = np.float32(4.0) + (np.float32(oy) + np.float32(0.5)) * np.float32(16.0) / oh - np.float32(0.5)
            sx = np.float32(5.0) + (np.float32(ox) + np.float32(0.5)) * np.float32(18.0) / ow - np.float32(0.5)
            out[idx] = bilinear_hwc(sy, sx, ch, ih, iw, c)
    elif OP == 4:
        h, w, c, ph, pw = d[:5]; y0, x0 = 7, 5
        for idx in range(OUT_N):
            ch = idx % c; x = (idx // c) % w; y = idx // (c * w)
            v = h01((y * w + x) * c + ch, 123) * np.float32(0.25)
            if y0 <= y < y0 + ph and x0 <= x < x0 + pw:
                v = np.float32(0.75) * h01(((y - y0) * pw + (x - x0)) * c + ch, 321) + np.float32(0.25) * v
            out[idx] = v
    elif OP == 5:
        n, c, h, w = d[:4]
        for idx in range(OUT_N):
            x = idx % w; y = (idx // w) % h; ch = (idx // (w * h)) % c; b = idx // (w * h * c)
            out[idx] = h01(((b * c + ch) * h + y) * 8 + x, 123)
    elif OP == 6:
        n, h, w, c = d[:4]
        for idx in range(OUT_N):
            ch = idx % c; x = (idx // c) % w; y = (idx // (c * w)) % h; b = idx // (c * w * h)
            sx = w - 1 - x; sy = h - 1 - y if b == 1 else y
            out[idx] = h01(((b * h + sy) * w + sx) * c + ch, 123)
    elif OP == 7:
        for idx in range(OUT_N):
            coord = idx % 4; b = idx // 4; box = make_box(b, 123)
            out[idx] = [np.float32(1) - box[2], box[1], np.float32(1) - box[0], box[3]][coord]
    elif OP == 8:
        s, length, feat = d[:3]
        for idx in range(OUT_N):
            f = idx % feat; seq = idx // feat; pos = (2 * seq + 1) % length
            out[idx] = h01((seq * length + pos) * feat + f, 123)
    elif OP == 9:
        for idx in range(OUT_N):
            q = int(np.floor(h01(idx, 123) * np.float32(300.0) - np.float32(20.0) + np.float32(0.5)))
            out[idx] = float(clamp(q, 0, 255))
    elif OP == 10:
        n, h, w, c = d[:4]
        for idx in range(OUT_N):
            ch = idx % c; x = (idx // c) % w; y = (idx // (c * w)) % h
            sx = np.float32(x) + np.float32(1.75) * np.sin(np.float32(0.35) * y)
            sy = np.float32(y) + np.float32(1.25) * np.sin(np.float32(0.27) * x)
            out[idx] = bilinear_hwc(sy, sx, ch, h, w, c)
    elif OP == 11:
        n, bins = d[:2]
        for b in range(bins):
            out[b] = sum(1 for i in range(n) if clamp(int(np.floor(h01(i, 123) * bins)), 0, bins - 1) == b)
    elif OP == 12:
        h, w, bins = d[:3]; tile = 8
        for idx in range(OUT_N):
            x = idx % w; y = idx // w; y0 = (y // tile) * tile; x0 = (x // tile) * tile
            b = clamp(int(np.floor(h01(y * w + x, 123) * bins)), 0, bins - 1)
            cdf = total = 0
            for yy in range(y0, min(y0 + tile, h)):
                for xx in range(x0, min(x0 + tile, w)):
                    bb = clamp(int(np.floor(h01(yy * w + xx, 123) * bins)), 0, bins - 1)
                    if bb <= b: cdf += 1
                    total += 1
            out[idx] = np.float32(cdf) / np.float32(total)
    elif OP == 13:
        h, w = d[:2]
        for idx in range(OUT_N):
            x = idx % w; y = idx // w
            vals = [h01(clamp(y + dy, 0, h - 1) * w + clamp(x + dx, 0, w - 1), 123) for dy in (-1,0,1) for dx in (-1,0,1)]
            out[idx] = sorted(vals)[4]
    elif OP == 14:
        h, w = d[:2]
        for idx in range(OUT_N):
            x = idx % w; y = idx // w
            den = np.float32(0.0015) * x - np.float32(0.0010) * y + np.float32(1.0)
            sx = (np.float32(0.94) * x + np.float32(0.11) * y - np.float32(1.7)) / den
            sy = (-np.float32(0.07) * x + np.float32(1.03) * y + np.float32(1.2)) / den
            out[idx] = bilinear_hwc(sy, sx, 0, h, w, 1)
    elif OP == 15:
        h, w, c = d[:3]
        for idx in range(OUT_N):
            x = idx % w; y = idx // w
            r, g, b = [h01((y * w + x) * 3 + ch, 123) for ch in range(3)]
            out[idx] = np.float32(0.299) * r + np.float32(0.587) * g + np.float32(0.114) * b
    elif OP in (16, 17, 18, 19):
        h, w = d[:2]; kw = [0.25, 0.5, 0.25]
        for idx in range(OUT_N):
            x = idx % w; y = idx // w; acc = np.float32(0)
            if OP == 16:
                k2 = [0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625]; p = 0
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        acc += np.float32(k2[p]) * h01(clamp(y + dy, 0, h - 1) * w + clamp(x + dx, 0, w - 1), 123); p += 1
            elif OP == 17:
                for dx in (-1,0,1): acc += np.float32(kw[dx + 1]) * h01(y * w + clamp(x + dx, 0, w - 1), 123)
            elif OP == 18:
                for dy in (-1,0,1): acc += np.float32(kw[dy + 1]) * h01(clamp(y + dy, 0, h - 1) * w + x, 123)
            else:
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        acc += np.float32(kw[dy + 1] * kw[dx + 1]) * h01(clamp(y + dy, 0, h - 1) * w + clamp(x + dx, 0, w - 1), 123)
            out[idx] = acc
    elif OP == 20:
        h, w = d[:2]
        for idx in range(OUT_N):
            x = idx % w; y = idx // w
            out[idx] = min(h01(clamp(y + dy, 0, h - 1) * w + clamp(x + dx, 0, w - 1), 123) for dy in (-1,0,1) for dx in (-1,0,1))
    elif OP == 21:
        ih, iw, oh, ow = d[:4]
        for idx in range(OUT_N):
            ox = idx % ow; oy = idx // ow
            out[idx] = np.float32(0.25) * sum(h01((oy * 2 + yy) * iw + ox * 2 + xx, 123) for yy in range(2) for xx in range(2))
    elif OP == 22:
        h, w = d[:2]
        for y in range(h):
            out[y] = sum(h01(y * w + x, 123) for x in range(w))
    elif OP == 23:
        n = d[0]; vals = np.asarray([h01(i, 123) for i in range(n)], dtype=np.float32)
        out[:] = (vals - vals.min()) / (vals.max() - vals.min() + np.float32(1e-6))
    elif OP == 24:
        for i in range(OUT_N): out[i] = np.float32(0.65) * h01(i, 123) + np.float32(0.35) * h01(i, 321) + np.float32(0.1)
    elif OP == 25:
        for i in range(OUT_N): out[i] = float(int(np.floor(h01(i, 123) * 256)) & int(np.floor(h01(i, 321) * 256)))
    elif OP == 26:
        for i in range(OUT_N): out[i] = 1.0 if h01(i, 123) > h01(i, 321) else 0.0
    elif OP == 27:
        a_n, b_n = d[:2]
        for idx in range(OUT_N):
            b = idx % b_n; a = idx // b_n
            out[idx] = iou(make_box(a, 123), make_box(b, 321))
    elif OP == 28:
        n = d[0]
        for i in range(n):
            bi = make_box(i, 123); keep = 1
            for j in range(n):
                bj = make_box(j, 123)
                if bj[4] > bi[4] and iou(bi, bj) > np.float32(0.35): keep = 0
            out[i] = keep
    elif OP in (29, 30):
        n, c, h, w, rois, ph, pw = d[:7]
        for idx in range(OUT_N):
            px = idx % pw; py = (idx // pw) % ph; ch = (idx // (pw * ph)) % c; r = idx // (pw * ph * c)
            box = make_box(r, 555); x1, y1, x2, y2 = box[0] * (w - 1), box[1] * (h - 1), box[2] * (w - 1), box[3] * (h - 1)
            if OP == 29:
                sy = y1 + (np.float32(py) + np.float32(0.5)) * (y2 - y1) / ph
                sx = x1 + (np.float32(px) + np.float32(0.5)) * (x2 - x1) / pw
                out[idx] = bilinear_hwc(sy, sx, ch, h, w, c)
            else:
                yy0 = int(np.floor(y1 + py * (y2 - y1) / ph)); yy1 = int(np.ceil(y1 + (py + 1) * (y2 - y1) / ph))
                xx0 = int(np.floor(x1 + px * (x2 - x1) / pw)); xx1 = int(np.ceil(x1 + (px + 1) * (x2 - x1) / pw))
                out[idx] = max(img_hwc(yy, xx, ch, h, w, c) for yy in range(yy0, yy1 + 1) for xx in range(xx0, xx1 + 1))
    elif OP in (31, 32):
        qn, rn = d[:2]; k = 3 if OP == 31 else d[2]
        for q in range(qn):
            ds = sorted(dist3(q, 123, i, 321) for i in range(rn))
            for rank in range(k): out[q * k + rank] = ds[rank]
    elif OP == 33:
        b, c, n, m = d[:4]
        for idx in range(OUT_N):
            mm = idx % m; ch = (idx // m) % c; bb = idx // (m * c); src = (mm * 3 + bb) % n
            out[idx] = h01((bb * c + ch) * n + src, 123)
    elif OP == 34:
        c, n, m, k = d[:4]
        for idx in range(OUT_N):
            kk = idx % k; mm = (idx // k) % m; ch = idx // (k * m); src = (mm * 2 + kk * 3) % n
            out[idx] = h01(ch * n + src, 123)
    elif OP == 35:
        c, n, qn = d[:3]
        for idx in range(OUT_N):
            q = idx % qn; ch = idx // qn; i0, i1, i2 = q % n, (q * 3 + 1) % n, (q * 5 + 2) % n
            out[idx] = np.float32(0.5) * h01(ch * n + i0, 123) + np.float32(0.3) * h01(ch * n + i1, 123) + np.float32(0.2) * h01(ch * n + i2, 123)
    elif OP == 36:
        n, s_n = d[:2]; selected = [0]
        for s in range(1, s_n):
            best, best_i = -1, 0
            for p in range(n):
                nearest = min(dist3(p, 123, q, 123) for q in selected)
                if nearest > best: best, best_i = nearest, p
            selected.append(best_i)
        out[:] = selected
    elif OP == 37:
        qn, rn, k = d[:3]
        for q in range(qn):
            found = [p for p in range(rn) if dist3(q, 123, p, 321) < np.float32(0.55)]
            for kk in range(k): out[q * k + kk] = found[kk] if kk < len(found) else -1
    elif OP == 38:
        p_n, b_n = d[:2]
        for p in range(p_n):
            pt = point3(p, 123)
            for b in range(b_n):
                box = make_box(b, 321)
                out[p * b_n + b] = 1.0 if (box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3] and 0.2 <= pt[2] <= 0.8) else 0.0
    elif OP == 39:
        n, c, h, w = d[:4]; oh, ow = h // 2, w // 2
        for idx in range(OUT_N):
            ox = idx % ow; oy = (idx // ow) % oh; ch = (idx // (ow * oh)) % c; b = idx // (ow * oh * c)
            out[idx] = max(img_chw(ch, oy * 2 + yy, ox * 2 + xx, c, h, w, 123 + b) for yy in range(2) for xx in range(2))
    elif OP == 40:
        n, c, ih, iw, oh, ow = d[:6]
        for idx in range(OUT_N):
            ox = idx % ow; oy = (idx // ow) % oh; ch = (idx // (ow * oh)) % c; b = idx // (ow * oh * c)
            iy = clamp(int(np.floor(np.float32(oy) * ih / oh)), 0, ih - 1)
            ix = clamp(int(np.floor(np.float32(ox) * iw / ow)), 0, iw - 1)
            out[idx] = img_chw(ch, iy, ix, c, ih, iw, 123 + b)
    return out

if __name__ == "__main__":
    V.run(reference)
