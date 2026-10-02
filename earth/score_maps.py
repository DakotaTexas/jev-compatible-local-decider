#!/usr/bin/env python3
"""Maps and scores for blind_earth.py runs, against the real land mask (pip install global-land-mask pillow numpy).

    python3 score_maps.py runs/lp-2.jsonl [runs/jev-2.jsonl ...]

For each run: a PNG map (white = land, black = water, red = no answer) and the scores: accuracy at
0.5 per pixel and weighted by real surface (cos latitude), Brier, log-loss.
"""
import json
import math
import sys

import numpy as np
from global_land_mask import globe
from PIL import Image

lignes = []
for chemin in sys.argv[1:]:
    rows = [json.loads(l) for l in open(chemin)]
    lats = sorted({r["lat"] for r in rows}, reverse=True)
    lons = sorted({r["lon"] for r in rows})
    if len(lats) > 1:
        pas = round(lats[0] - lats[1], 3)
    else:
        pas = 2.0
    H, W = int(round(180 / pas)), int(round(360 / pas))
    img = np.zeros((H, W, 3), dtype=np.uint8); img[:] = (60, 60, 60)
    ok = brier = ll = n = nb = 0
    for r in rows:
        y = int((90 - r["lat"]) / pas); x = int((r["lon"] + 180) / pas)
        if not (0 <= y < H and 0 <= x < W):
            continue
        p = r.get("p")
        if p is None:
            img[y, x] = (200, 30, 30); nb += 1
            continue
        g = int(round(255 * p)); img[y, x] = (g, g, g)
        vrai = 1.0 if globe.is_land(r["lat"], r["lon"]) else 0.0
        ok += (p >= 0.5) == (vrai == 1.0); brier += (p - vrai) ** 2
        q = min(max(p, 1e-4), 1 - 1e-4); ll += -(vrai * math.log(q) + (1 - vrai) * math.log(1 - q)); n += 1
    png = chemin.replace(".jsonl", ".png")
    Image.fromarray(img).resize((W * max(1, int(4 / pas)), H * max(1, int(4 / pas))), Image.NEAREST).save(png)
    jet = [r.get("jetons") for r in rows if r.get("jetons")]
    ws = [(math.cos(math.radians(r["lat"])), (r["p"] >= 0.5) == globe.is_land(r["lat"], r["lon"])) for r in rows if r.get("p") is not None]
    aw = sum(w for w, good in ws if good) / max(sum(w for w, _ in ws), 1e-9)
    lignes.append("%-28s %6d pts · accuracy %.1f %% (by surface %.1f %%) · Brier %.3f · log-loss %.3f · no answer %d%s -> %s" % (
        chemin.split("/")[-1], len(rows), 100 * ok / max(n, 1), 100 * aw, brier / max(n, 1), ll / max(n, 1), nb,
        " · %.0f tokens/reply" % (sum(jet) / len(jet)) if jet else "", png))

# reference: the real mask on the same grid
vrai = np.zeros((90, 180), dtype=np.uint8)
for i in range(90):
    for j in range(180):
        vrai[i, j] = 255 if globe.is_land(89 - 2 * i, -179 + 2 * j) else 0
Image.fromarray(vrai).resize((720, 360), Image.NEAREST).save(sys.argv[1].rsplit("/", 1)[0] + "/truth-2.png")
print("\n".join(lignes))
