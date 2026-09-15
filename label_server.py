"""Serve the labelling page and save each verdict straight into labels.csv.

A page opened from file:// cannot write to disk -- the browser will not let it
-- so the choice is between a page that makes you download a CSV at the end and
a tiny local server that saves as you go. This is the second: every click is a
POST that rewrites labels.csv before it answers, so closing the tab, a crash or
a flat battery costs at most the frame on screen. Nothing leaves the machine
and nothing but data/label_sample is touched.

    python label_server.py                 # then open the printed URL
    python label_server.py --port 8899

Keyboard: y = rip, n = no rip, d = doubt, left/right to move, any typing in the
notes box is saved with the verdict.
"""

import argparse
import csv
import io
import json
import os
import posixpath
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT_DIR = "data/label_sample"
IMAGE_DIR = f"{OUT_DIR}/images"
LABEL_CSV = f"{OUT_DIR}/labels.csv"
VERDICTS = {"yes", "no", "doubt", ""}

_lock = threading.Lock()


def read_rows():
    with open(LABEL_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), reader.fieldnames


def write_rows(rows, fields):
    """Rewrite the CSV via a temporary file, so an interrupted save cannot
    truncate the labels already in it."""
    tmp = LABEL_CSV + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, LABEL_CSV)


def save_label(frame_id, verdict, notes):
    """Set one row's verdict. Returns (ok, message, done_count, total)."""
    if verdict not in VERDICTS:
        return False, f"unknown verdict {verdict!r}", 0, 0
    with _lock:
        rows, fields = read_rows()
        hit = None
        for row in rows:
            if row.get("frame_id") == frame_id:
                hit = row
                break
        if hit is None:
            return False, f"no row with frame_id {frame_id!r}", 0, len(rows)
        hit["rip_present"] = verdict
        hit["notes"] = notes
        hit["labeled_at"] = (datetime.now(timezone.utc).isoformat(timespec="seconds")
                             if verdict else "")
        write_rows(rows, fields)
        done = sum(1 for r in rows if r.get("rip_present"))
        return True, "saved", done, len(rows)


PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Walton rip labelling</title>
<style>
  :root{ --bg:#11171c; --panel:#1a232a; --ink:#e6eef2; --muted:#93a6b0;
         --rule:#2b3942; --yes:#3f9e6a; --no:#b5544a; --doubt:#b08a3c; }
  *{ box-sizing:border-box }
  body{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5
        ui-sans-serif,system-ui,-apple-system,sans-serif; }
  header{ display:flex; gap:16px; align-items:baseline; flex-wrap:wrap;
          padding:10px 16px; border-bottom:1px solid var(--rule); }
  header b{ font-size:15px } header span{ color:var(--muted); font-size:13px }
  #bar{ height:3px; background:var(--rule) } #fill{ height:3px; background:var(--yes); width:0 }
  main{ padding:16px; max-width:1100px; margin:0 auto }
  #stage{ position:relative; background:#000; border:1px solid var(--rule);
          border-radius:4px; overflow:hidden; line-height:0 }
  #stage img{ width:100%; height:auto; display:block }
  #boxes{ position:absolute; inset:0; width:100%; height:100% }
  .meta{ display:flex; flex-wrap:wrap; gap:6px 20px; margin:12px 0;
         font:12px ui-monospace,monospace; color:var(--muted) }
  .meta b{ color:var(--ink); font-weight:600 }
  .row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:12px }
  button{ font:inherit; font-weight:600; padding:10px 20px; border-radius:5px;
          border:1px solid var(--rule); background:var(--panel); color:var(--ink);
          cursor:pointer }
  button:focus-visible{ outline:2px solid #6cf; outline-offset:2px }
  #yes{ border-color:var(--yes); color:var(--yes) }
  #no{ border-color:var(--no); color:var(--no) }
  #doubt{ border-color:var(--doubt); color:var(--doubt) }
  #yes.on{ background:var(--yes); color:#06120b }
  #no.on{ background:var(--no); color:#1a0806 }
  #doubt.on{ background:var(--doubt); color:#171003 }
  .nav{ margin-left:auto; display:flex; gap:8px }
  input[type=text]{ flex:1; min-width:200px; font:inherit; padding:9px 11px;
    border-radius:5px; border:1px solid var(--rule); background:var(--panel); color:var(--ink) }
  #note{ color:var(--muted); font-size:13px; min-height:1.4em; margin-top:8px }
  kbd{ font:11px ui-monospace,monospace; border:1px solid var(--rule);
       border-radius:3px; padding:1px 5px; color:var(--muted) }
</style>
<header>
  <b>Walton rip labelling</b>
  <span id="count"></span>
  <span id="strat"></span>
  <span style="margin-left:auto">
    <kbd>y</kbd> rip <kbd>n</kbd> none <kbd>d</kbd> doubt <kbd>&larr;</kbd><kbd>&rarr;</kbd> move
  </span>
</header>
<div id="bar"><div id="fill"></div></div>
<main>
  <div id="stage"><img id="shot" alt=""><svg id="boxes" preserveAspectRatio="none"></svg></div>
  <div class="meta" id="meta"></div>
  <div class="row">
    <button id="yes" type="button">Rip present</button>
    <button id="no" type="button">No rip</button>
    <button id="doubt" type="button">Doubt</button>
    <span class="nav">
      <button id="prev" type="button">&larr; Prev</button>
      <button id="next" type="button">Next &rarr;</button>
      <button id="skipto" type="button">Next unlabelled</button>
    </span>
  </div>
  <div class="row"><input id="notes" type="text" placeholder="notes (saved with the verdict)"></div>
  <div id="note"></div>
</main>
<script>
let rows = [], i = 0;

function esc(s){ return String(s == null ? "" : s); }

function draw(){
  const r = rows[i]; if(!r) return;
  document.getElementById("shot").src = "/images/" + encodeURIComponent(r.image);
  document.getElementById("strat").textContent = r.stratum + (r.booster ? "  ·  " + r.booster : "");
  document.getElementById("count").textContent =
    (i+1) + " / " + rows.length + "  ·  " + rows.filter(x=>x.rip_present).length + " labelled";
  document.getElementById("fill").style.width =
    (100 * rows.filter(x=>x.rip_present).length / rows.length) + "%";

  const n = v => (v === "" || v == null || isNaN(+v)) ? "—" : (+v).toFixed(2);
  document.getElementById("meta").innerHTML =
    "<span>" + esc(r.timestamp) + "</span>" +
    "<span>score <b>" + (r.score_max === "" ? "none" : n(r.score_max)) + "</b></span>" +
    "<span>boxes <b>" + (r.bbox_count === "" ? "0" : esc(r.bbox_count)) + "</b></span>" +
    "<span>Hs <b>" + n(r.mop_wave_height) + " m</b></span>" +
    "<span>cloud <b>" + n(r.cloud_cover) + "%</b></span>" +
    "<span>sun <b>" + n(r.solar_elevation) + "°</b></span>";

  document.getElementById("notes").value = r.notes || "";
  for(const id of ["yes","no","doubt"])
    document.getElementById(id).classList.toggle("on", r.rip_present === id);
  drawBoxes();
}

function drawBoxes(){
  const r = rows[i], svg = document.getElementById("boxes"), img = document.getElementById("shot");
  svg.innerHTML = "";
  let boxes = []; try{ boxes = JSON.parse(r.boxes || "[]"); }catch(e){ boxes = []; }
  if(!boxes.length) return;
  // Boxes are in SOURCE pixels. Set the viewBox to the image's natural size so
  // the browser scales them with the rendered image, at any window width.
  const w = img.naturalWidth || 1, h = img.naturalHeight || 1;
  svg.setAttribute("viewBox", "0 0 " + w + " " + h);
  const stroke = Math.max(2, Math.round(w / 400));
  for(const b of boxes){
    const rect = document.createElementNS("http://www.w3.org/2000/svg","rect");
    rect.setAttribute("x", b.x); rect.setAttribute("y", b.y);
    rect.setAttribute("width", b.w); rect.setAttribute("height", b.h);
    rect.setAttribute("fill", "none");
    rect.setAttribute("stroke", "#39d98a");
    rect.setAttribute("stroke-width", stroke);
    svg.appendChild(rect);
  }
}
document.getElementById("shot").addEventListener("load", drawBoxes);

async function save(verdict){
  const r = rows[i]; if(!r) return;
  r.rip_present = verdict;
  r.notes = document.getElementById("notes").value;
  draw();
  try{
    const resp = await fetch("/save", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({frame_id:r.frame_id, verdict:verdict, notes:r.notes})});
    const data = await resp.json();
    document.getElementById("note").textContent =
      data.ok ? ("saved — " + data.done + " of " + data.total + " labelled")
              : ("NOT SAVED: " + data.message);
  }catch(e){
    document.getElementById("note").textContent = "NOT SAVED: " + e;
  }
}

function go(step){ i = Math.min(rows.length-1, Math.max(0, i+step)); draw(); }
function nextUnlabelled(){
  for(let k=1; k<=rows.length; k++){
    const j = (i+k) % rows.length;
    if(!rows[j].rip_present){ i = j; draw(); return; }
  }
  document.getElementById("note").textContent = "every row has a verdict.";
}

document.getElementById("yes").onclick   = () => { save("yes");   nextUnlabelled(); };
document.getElementById("no").onclick    = () => { save("no");    nextUnlabelled(); };
document.getElementById("doubt").onclick = () => { save("doubt"); nextUnlabelled(); };
document.getElementById("prev").onclick  = () => go(-1);
document.getElementById("next").onclick  = () => go(1);
document.getElementById("skipto").onclick = nextUnlabelled;

addEventListener("keydown", e => {
  if(e.target.tagName === "INPUT") return;
  if(e.key === "y") document.getElementById("yes").click();
  else if(e.key === "n") document.getElementById("no").click();
  else if(e.key === "d") document.getElementById("doubt").click();
  else if(e.key === "ArrowLeft") go(-1);
  else if(e.key === "ArrowRight") go(1);
});

fetch("/rows").then(r => r.json()).then(data => {
  rows = data.rows;
  if(!rows.length){ document.getElementById("note").textContent = "labels.csv is empty."; return; }
  const first = rows.findIndex(r => !r.rip_present);
  i = first < 0 ? 0 : first;
  draw();
});
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/rows":
            rows, _ = read_rows()
            return self._send(200, json.dumps({"rows": rows}).encode(),
                              "application/json")
        if path.startswith("/images/"):
            # Resolve inside IMAGE_DIR and refuse anything that escapes it, so a
            # crafted path cannot read the rest of the disk over this port.
            name = posixpath.basename(path[len("/images/"):])
            from urllib.parse import unquote
            target = os.path.abspath(os.path.join(IMAGE_DIR, unquote(name)))
            if not target.startswith(os.path.abspath(IMAGE_DIR) + os.sep) \
                    or not os.path.isfile(target):
                return self._send(404, b"not found", "text/plain")
            ext = os.path.splitext(target)[1].lower()
            ctype = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                     ".png": "image/png", ".webp": "image/webp"}.get(ext,
                                                                     "application/octet-stream")
            with open(target, "rb") as fh:
                return self._send(200, fh.read(), ctype)
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.split("?")[0] != "/save":
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            return self._send(400, json.dumps({"ok": False, "message": str(exc)}).encode(),
                              "application/json")
        ok, message, done, total = save_label(payload.get("frame_id", ""),
                                              payload.get("verdict", ""),
                                              payload.get("notes", ""))
        return self._send(200 if ok else 400,
                          json.dumps({"ok": ok, "message": message,
                                      "done": done, "total": total}).encode(),
                          "application/json")

    def log_message(self, *args):
        pass  # one line per image request would bury the progress notes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-open", action="store_true",
                        help="do not open a browser window")
    args = parser.parse_args()

    if not os.path.exists(LABEL_CSV):
        sys.exit(f"{LABEL_CSV} does not exist — run build_label_sample.py first.")
    rows, _ = read_rows()
    done = sum(1 for r in rows if r.get("rip_present"))
    missing = [r["image"] for r in rows
               if not os.path.isfile(os.path.join(IMAGE_DIR, r.get("image", "")))]

    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  {len(rows)} rows, {done} already labelled")
    if missing:
        print(f"  WARNING: {len(missing)} rows name an image that is not in {IMAGE_DIR}/")
        print(f"           first missing: {missing[0]}")
    print(f"  serving {url}   (ctrl-C to stop; every click is saved before it answers)\n")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        rows, _ = read_rows()
        done = sum(1 for r in rows if r.get("rip_present"))
        print(f"\n  stopped. {done} of {len(rows)} labelled, saved in {LABEL_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
