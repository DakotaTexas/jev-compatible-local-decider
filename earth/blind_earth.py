#!/usr/bin/env python3
""""How does a blind model see the Earth?" (@arithmoquine, outsidetext.substack.com), on a local LLM and on Jev.

For each point of an equirectangular grid, ask the model whether the place is over land or water,
and read the answer in one of five ways:

  lp        logprobs (Jev-style): 1 token, thinking off, mass of "Land" / ("Land" + "Water")
  sample    no logprobs: 4 samples at temperature 1, thinking off, share of "Land"
  think     thinking on, 1 reply, read the final word (0 or 1)
  think_lp  thinking on + logprobs of the answer token after the reasoning (vLLM)
  jev       the Jev cloud API (/v1/systemone), a 2-option choice; key in JEV_API_KEY

    python3 blind_earth.py <mode> <base_url> <model> [--step 2] [--streams 2] [--priority 100] [--max-points N]

Output: runs/<mode><tag>-<step>.jsonl (resumes where it stopped). Maps and scores: score_maps.py.
"""
import argparse
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = ("If this location is over land, say 'Land'. If this location is over water, say 'Water'. "
          "Do not say anything else. %s")

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["lp", "sample", "think", "think_lp", "jev"])
ap.add_argument("base")
ap.add_argument("modele")
ap.add_argument("--step", dest="pas", type=float, default=2.0)
ap.add_argument("--streams", dest="flux", type=int, default=2)
ap.add_argument("--priority", dest="priorite", default="0")   # vLLM: higher = later; TensorFold: "background"
ap.add_argument("--max-points", type=int, default=0)
ap.add_argument("--tag", default="")
A = ap.parse_args()
PRIO = int(A.priorite) if A.priorite.lstrip("-").isdigit() else A.priorite
OUT = os.path.join(HERE, "runs", "%s%s-%g.jsonl" % (A.mode, A.tag, A.pas))
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def coord(lat, lon):
    return "%.1f° %s, %.1f° %s" % (abs(lat), "N" if lat >= 0 else "S", abs(lon), "E" if lon >= 0 else "W")


def post(body, timeout=600):
    req = urllib.request.Request(A.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    for essai in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError) as e:
            if essai == 3:
                raise
            time.sleep(5 * (essai + 1))


def masse(top):
    land = water = 0.0
    for e in top:
        t = e["token"].strip().strip("'\".").lower()
        if t and "land".startswith(t) and len(t) >= 2 or t == "land":
            land += math.exp(e["logprob"])
        elif t and "water".startswith(t) and len(t) >= 2 or t == "water":
            water += math.exp(e["logprob"])
    return land, water


def mot(texte):
    t = (texte or "").strip().strip("'\".*").lower()
    return 1.0 if t.startswith("land") else 0.0 if t.startswith("water") else None


def jev(lat, lon):
    """Jev (TypeSafe /v1/systemone): one 2-option question, same instruction. Key in JEV_API_KEY."""
    body = {"model": A.modele, "state": coord(lat, lon),
            "questions": {"lieu": {"type": "choice", "instructions": PROMPT % "",
                                   "criteria": {"Land": "The location is over land.",
                                                "Water": "The location is over water."}}}}
    req = urllib.request.Request(A.base.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "Authorization": "Bearer " + os.environ["JEV_API_KEY"]})
    for essai in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                o = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or essai == 3:
                raise RuntimeError("HTTP %s %s" % (e.code, e.read().decode()[:200])) from None
            time.sleep(5 * (essai + 1))
    a = o["answers"]["lieu"]
    return {"p": a["probabilities"].get("Land"), "usage": o.get("usage"), "modele": o.get("model")}


def un_point(lat, lon):
    if A.mode == "jev":
        return jev(lat, lon)
    msg = [{"role": "user", "content": PROMPT % coord(lat, lon)}]
    base = {"model": A.modele, "messages": msg, "priority": PRIO}
    if A.mode == "lp":
        o = post(dict(base, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20,
                      chat_template_kwargs={"enable_thinking": False}))
        land, water = masse(o["choices"][0]["logprobs"]["content"][0]["top_logprobs"])
        p = land / (land + water) if land + water > 0 else None
        return {"p": p, "couverture": land + water}
    if A.mode == "sample":
        votes = []
        for s in range(4):
            o = post(dict(base, max_tokens=3, temperature=1.0, seed=1000 * s + 7,
                          chat_template_kwargs={"enable_thinking": False}))
            votes.append(mot(o["choices"][0]["message"].get("content")))
        ok = [v for v in votes if v is not None]
        return {"p": sum(ok) / len(ok) if ok else None, "votes": votes}
    if A.mode == "think":
        o = post(dict(base, max_tokens=4096, temperature=0.6, top_p=0.95,
                      chat_template_kwargs={"enable_thinking": True}))
        c = o["choices"][0]
        return {"p": mot(c["message"].get("content")), "jetons": o.get("usage", {}).get("completion_tokens"),
                "fin": c.get("finish_reason")}
    # think_lp: thinking on, logprobs of every token; read the answer word after the reasoning
    o = post(dict(base, max_tokens=4096, temperature=0.6, top_p=0.95, logprobs=True, top_logprobs=20,
                  chat_template_kwargs={"enable_thinking": True}))
    c = o["choices"][0]
    toks = (c.get("logprobs") or {}).get("content") or []
    fin = max((i for i, e in enumerate(toks) if "</think>" in e["token"]), default=-1)
    for e in toks[fin + 1:]:
        if e["token"].strip().strip("'\".*"):
            land, water = masse(e["top_logprobs"])
            return {"p": land / (land + water) if land + water > 0 else None, "couverture": land + water,
                    "jetons": o.get("usage", {}).get("completion_tokens"), "lu": e["token"]}
    return {"p": None, "jetons": o.get("usage", {}).get("completion_tokens"), "fin": c.get("finish_reason")}


points = []
lat = 90 - A.pas / 2
while lat > -90:
    lon = -180 + A.pas / 2
    while lon < 180:
        points.append((round(lat, 3), round(lon, 3)))
        lon += A.pas
    lat -= A.pas
if A.max_points:
    points = points[:: max(1, len(points) // A.max_points)][: A.max_points]
fait = set()
if os.path.exists(OUT):
    for l in open(OUT):
        d = json.loads(l)
        fait.add((d["lat"], d["lon"]))
reste = [p for p in points if p not in fait]
print("%s: %d points, %d done, %d to do -> %s" % (A.mode, len(points), len(fait), len(reste), OUT), flush=True)
verrou = threading.Lock()
t0 = time.time(); n = [0]


def tache(pt):
    try:
        r = un_point(*pt)
    except Exception as e:                                           # noqa: BLE001
        r = {"p": None, "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
    r.update(lat=pt[0], lon=pt[1])
    with verrou:
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        n[0] += 1
        if n[0] % 200 == 0:
            v = n[0] / (time.time() - t0)
            print("  %d/%d · %.1f points/s · ~%.0f min left" % (n[0], len(reste), v, (len(reste) - n[0]) / v / 60),
                  flush=True)


with ThreadPoolExecutor(A.flux) as ex:
    list(ex.map(tache, reste))
print("DONE %s in %.0f s" % (A.mode, time.time() - t0))
