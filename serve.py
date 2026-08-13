#!/usr/bin/env python3
"""Minimal web UI for the finished system.

    ollama serve &
    .venv/bin/python serve.py          # then open http://localhost:8000

Standard library only, single file, no framework. The corpus and vectors load once at startup;
each request runs retrieval and generation.

**Not part of the study.** Nothing here is measured and no driver imports it. It exists so the
system can be demonstrated on an arbitrary question, and so the effect the study measures can be
watched rather than taken on trust: ask a question in compare mode and the superseded PEPs
appear on the left and vanish on the right.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from speceval.pipeline import DEFAULT_STRENGTH, Answer, Pipeline

PIPELINE: Pipeline | None = None

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>speceval</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { --bg:#fbfbfa; --fg:#1a1a19; --muted:#6b6b68; --line:#e2e2df;
          --card:#fff; --dead:#b3261e; --live:#1b6b3a; --accent:#2a5db0; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16161a; --fg:#e8e8e6; --muted:#9a9a96; --line:#2c2c31;
            --card:#1e1e23; --dead:#ff8a80; --live:#7fd8a0; --accent:#7aa7f0; }
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
         font:15px/1.6 ui-sans-serif,-apple-system,'Segoe UI',system-ui,sans-serif; }
  main { max-width:1080px; margin:0 auto; }
  h1 { font-size:1.35rem; margin:0 0 .25rem; letter-spacing:-.01em; }
  .sub { color:var(--muted); margin:0 0 1.75rem; font-size:.9rem; }
  form { display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:1rem; }
  input[type=text] { flex:1 1 24rem; padding:.7rem .85rem; font-size:1rem; color:var(--fg);
    background:var(--card); border:1px solid var(--line); border-radius:8px; }
  input[type=text]:focus { outline:2px solid var(--accent); outline-offset:-1px; }
  button { padding:.7rem 1.15rem; font-size:.95rem; font-weight:600; cursor:pointer;
    color:#fff; background:var(--accent); border:0; border-radius:8px; }
  button:disabled { opacity:.55; cursor:progress; }
  .opts { display:flex; align-items:center; gap:.4rem; color:var(--muted); font-size:.9rem; }
  .cols { display:grid; gap:1rem; grid-template-columns:1fr; }
  @media (min-width:900px) { .cols.two { grid-template-columns:1fr 1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:1rem 1.1rem; }
  .card h2 { font-size:.78rem; text-transform:uppercase; letter-spacing:.07em;
             color:var(--muted); margin:0 0 .8rem; font-weight:600; }
  .pep { display:flex; gap:.5rem; align-items:baseline; padding:.3rem 0; font-size:.87rem;
         border-bottom:1px solid var(--line); }
  .pep:last-of-type { border-bottom:0; }
  .badge { flex:none; font-size:.68rem; font-weight:700; letter-spacing:.04em;
           padding:.1rem .4rem; border-radius:4px; text-transform:uppercase; }
  .badge.dead { color:var(--dead); border:1px solid var(--dead); }
  .badge.live { color:var(--live); border:1px solid var(--live); }
  .num { flex:none; font-variant-numeric:tabular-nums; font-weight:600; }
  .title { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .answer { margin:1rem 0 .75rem; white-space:pre-wrap; }
  .cites { font-size:.85rem; color:var(--muted); }
  .cites b.dead { color:var(--dead); }
  .cites b.live { color:var(--live); }
  .warn { margin-top:.6rem; padding:.5rem .7rem; border-radius:6px; font-size:.85rem;
          color:var(--dead); border:1px solid var(--dead); }
  .meta { margin-top:.6rem; color:var(--muted); font-size:.78rem; }
  .err { color:var(--dead); }
  .examples { margin-top:1.5rem; font-size:.85rem; color:var(--muted); }
  .examples button { background:none; color:var(--accent); border:0; padding:0 .35rem;
                     font-weight:400; font-size:.85rem; text-decoration:underline; cursor:pointer; }
</style>
<main>
  <h1>speceval</h1>
  <p class="sub">Ask about Python. Answers come from the PEP corpus &mdash; 734 documents,
     36% of them rejected, withdrawn, superseded or deferred.</p>

  <form id="f">
    <input type="text" id="q" placeholder="e.g. how do I define an enumeration type"
           autocomplete="off" required>
    <label class="opts"><input type="checkbox" id="cmp" checked> compare</label>
    <button id="go" type="submit">Ask</button>
  </form>

  <div class="examples">Try:
    <button type="button" data-q="how do I postpone the evaluation of annotations">annotations</button>·
    <button type="button" data-q="which manylinux platform tag should a wheel target">manylinux</button>·
    <button type="button" data-q="how are Python package version numbers compared">versions</button>·
    <button type="button" data-q="is return allowed inside a finally block">finally</button>
  </div>

  <div id="out" class="cols"></div>
</main>
<script>
const out = document.getElementById('out'), go = document.getElementById('go'),
      qin = document.getElementById('q'), cmp = document.getElementById('cmp');

document.querySelectorAll('.examples button').forEach(b =>
  b.onclick = () => { qin.value = b.dataset.q; document.getElementById('f').requestSubmit(); });

function card(a) {
  const peps = a.retrieved.map(c => `<div class="pep">
      <span class="badge ${c.authoritative ? 'live' : 'dead'}">${c.authoritative ? 'live' : 'dead'}</span>
      <span class="num">PEP ${c.number}</span>
      <span class="title" title="${c.title}">${c.title}</span></div>`).join('');
  const cites = a.cited.length
    ? a.cited.map(c => `<b class="${c.authoritative ? 'live' : 'dead'}">PEP ${c.number}</b> [${c.status}]`).join(', ')
    : '(none)';
  const dead = a.cited.filter(c => !c.authoritative).length;
  return `<div class="card">
    <h2>${a.strength === 0 ? 'Without authority reranking' : 'With authority reranking'}</h2>
    ${peps}
    <div class="answer">${a.text}</div>
    <div class="cites">cited: ${cites}</div>
    ${dead ? `<div class="warn">Cites ${dead} non-authoritative PEP${dead > 1 ? 's' : ''}</div>` : ''}
    <div class="meta">${a.elapsed_s.toFixed(1)}s</div></div>`;
}

document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  go.disabled = true; go.textContent = 'Thinking…';
  out.className = 'cols'; out.innerHTML = '<div class="card">Retrieving and generating…</div>';
  try {
    const r = await fetch('/api/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: qin.value, compare: cmp.checked})
    });
    const d = await r.json();
    if (d.error) { out.innerHTML = `<div class="card err">${d.error}</div>`; return; }
    out.className = 'cols' + (d.answers.length > 1 ? ' two' : '');
    out.innerHTML = d.answers.map(card).join('');
  } catch (err) {
    out.innerHTML = `<div class="card err">${err}</div>`;
  } finally { go.disabled = false; go.textContent = 'Ask'; }
};
</script>
"""


def as_dict(answer: Answer) -> dict:
    return {
        "strength": answer.strength,
        "text": html.escape(answer.text),
        "elapsed_s": answer.elapsed_s,
        "retrieved": [
            {
                "number": c.number,
                "title": html.escape(c.title),
                "status": c.status,
                "authoritative": c.is_authoritative,
            }
            for c in answer.retrieved
        ],
        "cited": [
            {"number": c.number, "status": c.status, "authoritative": c.is_authoritative}
            for c in answer.cited
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path not in ("/", "/index.html"):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        self._send(200, PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        assert PIPELINE is not None

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("question must not be empty")
            # Bound the input: this is a demo, and an unbounded prompt is an easy way to
            # make generation hang.
            if len(question) > 400:
                raise ValueError("question too long (max 400 characters)")

            if payload.get("compare"):
                answers = list(PIPELINE.compare(question))
            else:
                answers = [PIPELINE.ask(question, strength=DEFAULT_STRENGTH)]
            body = json.dumps({"answers": [as_dict(a) for a in answers]}).encode()
            self._send(200, body, "application/json")
        except Exception as error:  # a demo server should not die on one bad request
            body = json.dumps({"error": f"{type(error).__name__}: {error}"}).encode()
            self._send(400, body, "application/json")


def main() -> int:
    global PIPELINE
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print("loading corpus and vectors...", file=sys.stderr)
    PIPELINE = Pipeline()
    print(f"{len(PIPELINE.peps)} PEPs, {len(PIPELINE.chunks)} chunks ready", file=sys.stderr)
    print(f"listening on http://{args.host}:{args.port}", file=sys.stderr)

    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
