#!/usr/bin/env python3
"""A Jev-compatible /v1/systemone server on top of ANY local LLM that returns logprobs.

Each Jev question becomes a short prompt with lettered options (A, B, C...). We ask the model
for ONE token, at temperature 0, with thinking off, and read the top_logprobs of that token.
The answer is the probability mass of each letter, renormalised over the options. No text is
generated and nothing is parsed.

    LLM_URL=http://127.0.0.1:8000 LLM_MODEL=<served model name> python3 systemone_logprobs.py 8021
    curl -s localhost:8021/v1/systemone -d '{"state": "...", "questions": {...}}'

Works with any OpenAI-compatible server that returns `logprobs` + `top_logprobs` on chat
completions (vLLM, SGLang, TensorFold >= 0.6...). Standard library only.

LLM_MAX_INFLIGHT (default 2) caps the requests this server sends to the model at once, across
all callers: each question is one request, so a 6-question call would otherwise take 6 slots
and slow down everybody else on the same box.
"""
import json
import math
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8000").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "default")
# Request priority, if the server supports it (vLLM: an integer, higher = later; TensorFold: "background").
PRIORITY = os.environ.get("LLM_PRIORITY")
MAX_INFLIGHT = int(os.environ.get("LLM_MAX_INFLIGHT", "2"))
SEATS = threading.BoundedSemaphore(MAX_INFLIGHT)
LETTERS = "ABCDEFGHIJKLMNOPQRST"          # top_logprobs is capped at 20 on most servers

SYSTEM = ("You are a decision engine. Read the state, then answer the question by giving the "
          "letter of the single best option. Output the letter only.")


def options_for(q):
    """[(label, description)] in order; the labels are the ones of the Jev contract."""
    t = q["type"]
    crit = q.get("criteria")
    if t == "noul":
        desc = crit if isinstance(crit, dict) else {}
        return [("yes", desc.get("true", "The statement holds.")),
                ("no", desc.get("false", "The statement does not hold."))]
    if t == "score":
        return [(str(i), d) for i, d in enumerate(crit)]
    if isinstance(crit, dict):
        return list(crit.items())
    return [(c, "") for c in crit]


def ask(state, q):
    opts = options_for(q)
    if len(opts) > len(LETTERS):
        raise ValueError("more than %d options" % len(LETTERS))
    lines = ["%s. %s%s" % (LETTERS[i], lab, (" - " + d) if d else "") for i, (lab, d) in enumerate(opts)]
    stmt = q.get("instructions", "")
    if q["type"] == "noul":
        stmt = "Is the following true? " + stmt
    user = "STATE:\n%s\n\nQUESTION:\n%s\n\nOPTIONS:\n%s\n\nAnswer with one letter." % (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False), stmt, "\n".join(lines))
    body = {"model": LLM_MODEL, "temperature": 0, "max_tokens": 1, "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    if PRIORITY:
        body["priority"] = int(PRIORITY) if PRIORITY.lstrip("-").isdigit() else PRIORITY
    req = urllib.request.Request(LLM_URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    with SEATS, urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    mass = [0.0] * len(opts)
    for e in top:
        tok = e["token"].strip().rstrip(".").upper()
        if len(tok) == 1 and tok in LETTERS[:len(opts)]:
            mass[LETTERS.index(tok)] += math.exp(e["logprob"])
    total = sum(mass)
    covered = total
    if total <= 0:
        mass = [1.0 / len(opts)] * len(opts)
        total = 1.0
    probs = {lab: mass[i] / total for i, (lab, _) in enumerate(opts)}
    usage = out.get("usage", {})
    return probs, covered, usage


def answer(q, probs):
    t = q["type"]
    best = max(probs, key=probs.get)
    if t == "noul":
        return {"type": "noul", "noul": probs["yes"], "confidence": max(probs.values())}
    if t == "score":
        exp = sum(int(k) * v for k, v in probs.items())
        return {"type": "score", "score": exp, "choice": best, "probabilities": probs,
                "confidence": probs[best]}
    return {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": "local-logprobs", "object": "model"}]})

    def do_POST(self):
        if self.path not in ("/v1/systemone", "/v1/predict"):
            return self._send(404, {"error": "not found"})
        t0 = time.time()
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            qs = req["questions"]
            with ThreadPoolExecutor(max_workers=len(qs)) as ex:
                res = dict(zip(qs, ex.map(lambda k: ask(req["state"], qs[k]), qs)))
            answers = {k: answer(qs[k], res[k][0]) for k in qs}
            cov = min(v[1] for v in res.values())
            ptok = sum(v[2].get("prompt_tokens", 0) for v in res.values())
            self._send(200, {"model": "local-logprobs", "answers": answers,
                             "usage": {"input_tokens": ptok}, "letter_mass_min": cov,
                             "latency_ms": int((time.time() - t0) * 1000)})
        except Exception as e:                                         # noqa: BLE001
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8021
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
