"""Serve the labelling page and save each verdict straight into labels.csv.

A page opened from file:// cannot write to disk -- the browser will not let it
-- so the choice is between a page that makes you download a CSV at the end and
a tiny local server that saves as you go. This is the second: every click is a
POST that rewrites labels.csv before it answers, so closing the tab, a crash or
a flat battery costs at most the frame on screen. Nothing leaves the machine
and nothing but data/label_sample is touched.

The failure this was rewritten to kill: the page used to set rip_present
locally BEFORE the POST and never put it back if the POST failed. A failed save
therefore looked labelled in the browser, "next unlabelled" skipped it, and the
work went nowhere -- the browser showing progress while the disk stayed empty.
Now the local row is only updated after the server confirms the write, a
failure is shown in red and STOPS the advance, and every save is logged to the
terminal with a running count so the two can be compared without guessing.

    python label_server.py                      # then open the printed URL
    python label_server.py --port 8899
    python label_server.py --import dump.json   # merge labels from elsewhere

Keyboard: y = rip, n = no rip, d = doubt, u = unusable, left/right to move.
Boxes can be marked individually with 1-9 (cycles true -> false -> unset).
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

# "unusable" is not a fourth shade of doubt. Doubt means the image is readable
# and the answer is genuinely unclear; unusable means the frame cannot be
# judged at all -- lens water, total dark, a test card. They have to be
# separable downstream: doubt belongs in the precision bracket, unusable
# belongs out of the denominator entirely.
VERDICTS = {"yes", "no", "doubt", "unusable", ""}
BOX_COLUMN = "box_labels"
EXTRA_COLUMNS = [BOX_COLUMN]

_lock = threading.Lock()
_saves = 0


def read_rows():
    with open(LABEL_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for column in EXTRA_COLUMNS:
        if column not in fields:
            fields.append(column)
            for row in rows:
                row.setdefault(column, "")
    for row in rows:
        for column in EXTRA_COLUMNS:
            if row.get(column) is None:
                row[column] = ""
    return rows, fields


def write_rows(rows, fields):
    """Rewrite the CSV via a temporary file, so an interrupted save cannot
    truncate the labels already in it."""
    tmp = LABEL_CSV + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, LABEL_CSV)


def count_done(rows):
    return sum(1 for r in rows if (r.get("rip_present") or "").strip())


def clean_box_labels(value):
    """{"0": true, "3": false} from whatever the page sent, or "" for nothing.

    Stored as JSON in one column rather than as a column per box: the number of
    boxes varies by frame, and a wide table would have to be rebuilt every time
    a frame with more boxes than any before it turned up.
    """
    if value in (None, "", {}):
        return ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return ""
    if not isinstance(value, dict):
        return ""
    out = {}
    for key, flag in value.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if index >= 0 and isinstance(flag, bool):
            out[str(index)] = flag
    return json.dumps(out, sort_keys=True) if out else ""


def save_label(frame_id, verdict, notes, box_labels=None, log=False):
    """Set one row's verdict. Returns (ok, message, done_count, total).

    Keyed by frame_id, so relabelling a frame overwrites its row rather than
    appending a second one.
    """
    global _saves
    if verdict not in VERDICTS:
        return False, f"unknown verdict {verdict!r}", 0, 0
    with _lock:
        rows, fields = read_rows()
        hit = next((r for r in rows if r.get("frame_id") == frame_id), None)
        if hit is None:
            return False, f"no row with frame_id {frame_id!r}", 0, len(rows)
        hit["rip_present"] = verdict
        hit["notes"] = notes or ""
        hit[BOX_COLUMN] = clean_box_labels(box_labels)
        hit["labeled_at"] = (datetime.now(timezone.utc).isoformat(timespec="seconds")
                             if verdict else "")
        write_rows(rows, fields)
        done = count_done(rows)
        _saves += 1
        if log:
            stamp = datetime.now().strftime("%H:%M:%S")
            boxes = hit[BOX_COLUMN]
            print(f"  [{stamp}] {frame_id}  {verdict or 'cleared':<9}"
                  f"{done:>4} of {len(rows)} labelled"
                  + (f"   boxes {boxes}" if boxes else ""), flush=True)
        return True, "saved", done, len(rows)


def import_labels(path, overwrite=False):
    """Merge labels from a JSON dump into labels.csv.

    Accepts either {frame_id: {...}} or a list of objects carrying frame_id.
    A verdict already in the CSV is kept unless --import-overwrite is passed:
    an import is a recovery, and a recovery that silently replaces newer work
    with older is not one.
    """
    # A CSV as readily as JSON: the rebuild's own backup of labels.csv is the
    # most likely thing anyone imports, and telling someone to convert their
    # recovered labels to JSON first is a step that exists only because the
    # reader was narrow.
    if os.path.splitext(path)[1].lower() == ".csv":
        with open(path, newline="") as fh:
            payload = [row for row in csv.DictReader(fh)
                       if (row.get("rip_present") or "").strip()]
    else:
        with open(path) as fh:
            payload = json.load(fh)

    if isinstance(payload, dict):
        entries = [dict(value, frame_id=key) if isinstance(value, dict)
                   else {"frame_id": key, "rip_present": value}
                   for key, value in payload.items()]
    elif isinstance(payload, list):
        entries = [e for e in payload if isinstance(e, dict)]
    else:
        return 0, 0, [f"unsupported JSON: {type(payload).__name__}"]

    rows, fields = read_rows()
    by_id = {r.get("frame_id"): r for r in rows}
    applied, skipped, problems = 0, 0, []
    for entry in entries:
        frame_id = str(entry.get("frame_id") or entry.get("id") or "").strip()
        verdict = str(entry.get("rip_present") or entry.get("verdict") or "").strip()
        if verdict == "unsure":
            verdict = "doubt"
        row = by_id.get(frame_id)
        if row is None:
            problems.append(f"no row for frame_id {frame_id!r}")
            continue
        if verdict not in VERDICTS:
            problems.append(f"{frame_id}: unknown verdict {verdict!r}")
            continue
        if (row.get("rip_present") or "").strip() and not overwrite:
            skipped += 1
            continue
        row["rip_present"] = verdict
        if entry.get("notes"):
            row["notes"] = str(entry["notes"])
        if entry.get(BOX_COLUMN) is not None:
            row[BOX_COLUMN] = clean_box_labels(entry[BOX_COLUMN])
        row["labeled_at"] = str(entry.get("labeled_at") or
                                datetime.now(timezone.utc)
                                .isoformat(timespec="seconds"))
        applied += 1
    if applied:
        write_rows(rows, fields)
    return applied, skipped, problems


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
<style>
  #note{ margin-top:12px; font:13px ui-monospace,monospace; min-height:20px }
  #note.good{ color:#39d98a } #note.bad{ color:#ff8a7a; font-weight:700 }
  .boxcap{ color:var(--muted); font:12px ui-monospace,monospace }
  .boxbtn{ padding:5px 12px; font:12px ui-monospace,monospace; font-weight:600 }
  .boxbtn.ok{ border-color:var(--yes); color:#8fe3b4 }
  .boxbtn.bad{ border-color:var(--no); color:#ffb3a9 }
</style>

<header>
  <b>Walton rip labelling</b>
  <span id="count"></span>
  <span id="strat"></span>
  <span style="margin-left:auto">
    <kbd>y</kbd> rip <kbd>n</kbd> none <kbd>d</kbd> doubt <kbd>u</kbd> unusable
    <kbd>1</kbd>-<kbd>9</kbd> box <kbd>&larr;</kbd><kbd>&rarr;</kbd> move
  </span>
</header>
<div id="bar"><div id="fill"></div></div>
<main>
  <div id="stage"><img id="shot" alt=""><svg id="boxes" preserveAspectRatio="none"></svg></div>
  <div class="meta" id="meta"></div>
  <div class="row" id="boxrow"></div>
  <div class="row">
    <button id="yes" type="button">Rip present</button>
    <button id="no" type="button">No rip</button>
    <button id="doubt" type="button">Doubt</button>
    <button id="unusable" type="button">Unusable</button>
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
let rows = [], i = 0, busy = false;
const VERDICTS = ["yes","no","doubt","unusable"];
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function status(text, bad){
  const el = document.getElementById("note");
  el.textContent = text;
  el.className = bad ? "bad" : "good";
}

function boxLabels(r){
  try { return JSON.parse(r.box_labels || "{}") || {}; } catch(e){ return {}; }
}

function boxCount(r){
  try { return (JSON.parse(r.boxes || "[]") || []).length; } catch(e){ return 0; }
}

function drawBoxRow(){
  const r = rows[i], host = document.getElementById("boxrow");
  host.innerHTML = "";
  const count = boxCount(r);
  if(!count){ host.style.display = "none"; return; }
  host.style.display = "flex";
  const marks = boxLabels(r);
  const caption = document.createElement("span");
  caption.className = "boxcap";
  caption.textContent = count + (count === 1 ? " box:" : " boxes:");
  host.appendChild(caption);
  for(let b = 0; b < count; b++){
    const mark = marks[String(b)];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "boxbtn" + (mark === true ? " ok" : mark === false ? " bad" : "");
    button.textContent = (b+1) + ": " + (mark === true ? "true" : mark === false ? "false" : "—");
    button.onclick = () => cycleBox(b);
    host.appendChild(button);
  }
}

function cycleBox(b){
  const r = rows[i], marks = boxLabels(r), key = String(b);
  const now = marks[key];
  if(now === undefined) marks[key] = true;
  else if(now === true) marks[key] = false;
  else delete marks[key];
  r.box_labels = JSON.stringify(marks);
  drawBoxRow();
  if((r.rip_present || "").trim()) save(r.rip_present, false);
}

function draw(){
  const r = rows[i]; if(!r) return;
  document.getElementById("shot").src = "/images/" + encodeURIComponent(r.image);
  document.getElementById("strat").textContent =
    r.stratum + (r.booster ? "  ·  " + r.booster : "");
  const done = rows.filter(x => (x.rip_present || "").trim()).length;
  document.getElementById("count").textContent =
    (i+1) + " / " + rows.length + "  ·  " + done + " labelled";
  document.getElementById("fill").style.width = (100 * done / rows.length) + "%";

  const n = v => (v === "" || v == null || isNaN(+v)) ? "—" : (+v).toFixed(2);
  document.getElementById("meta").innerHTML =
    "<span>" + esc(r.timestamp) + "</span>" +
    "<span>score <b>" + (r.score_max === "" ? "none" : n(r.score_max)) + "</b></span>" +
    "<span>boxes <b>" + (r.bbox_count === "" ? "0" : esc(r.bbox_count)) + "</b></span>" +
    "<span>Hs <b>" + n(r.mop_wave_height) + " m</b></span>" +
    "<span>cloud <b>" + n(r.cloud_cover) + "%</b></span>" +
    "<span>sun <b>" + n(r.solar_elevation) + "°</b></span>";

  document.getElementById("notes").value = r.notes || "";
  for(const id of VERDICTS)
    document.getElementById(id).classList.toggle("on", r.rip_present === id);
  drawBoxRow();
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

// The local row is updated only AFTER the server confirms the write. The old
// version set it first, so a failed POST left the page showing a verdict that
// was never on disk and "next unlabelled" walked straight past it.
async function save(verdict, advance){
  const r = rows[i];
  if(!r || busy) return false;
  busy = true;
  status("saving…", false);
  const notes = document.getElementById("notes").value;
  const boxes = r.box_labels || "";
  try{
    const resp = await fetch("/save", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({frame_id: r.frame_id, verdict: verdict,
                            notes: notes, box_labels: boxes})});
    if(!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    if(!data.ok){ status("NOT SAVED: " + data.message, true); return false; }
    r.rip_present = verdict;
    r.notes = notes;
    status("saved " + esc(r.frame_id) + " — " + data.done + " of " + data.total
           + " labelled", false);
    draw();
    if(advance) nextUnlabelled();
    return true;
  }catch(e){
    status("NOT SAVED (" + e.message + ") — this frame is NOT on disk. "
           + "Check the terminal; nothing has advanced.", true);
    return false;
  }finally{
    busy = false;
  }
}

function go(step){ i = Math.min(rows.length-1, Math.max(0, i+step)); draw(); }
function nextUnlabelled(){
  for(let k=1; k<=rows.length; k++){
    const j = (i+k) % rows.length;
    if(!(rows[j].rip_present || "").trim()){ i = j; draw(); return; }
  }
  status("every row has a verdict.", false);
}

for(const id of VERDICTS)
  document.getElementById(id).onclick = () => save(id, true);
document.getElementById("prev").onclick  = () => go(-1);
document.getElementById("next").onclick  = () => go(1);
document.getElementById("skipto").onclick = nextUnlabelled;

addEventListener("keydown", e => {
  if(e.target.tagName === "INPUT") return;
  if(e.key === "y") save("yes", true);
  else if(e.key === "n") save("no", true);
  else if(e.key === "d") save("doubt", true);
  else if(e.key === "u") save("unusable", true);
  else if(e.key >= "1" && e.key <= "9") cycleBox(+e.key - 1);
  else if(e.key === "ArrowLeft") go(-1);
  else if(e.key === "ArrowRight") go(1);
});

fetch("/rows").then(r => r.json()).then(data => {
  rows = data.rows;
  if(!rows.length){ status("labels.csv is empty.", true); return; }
  // Resume where the work stopped, so a restart does not start at frame 1.
  const first = rows.findIndex(r => !(r.rip_present || "").trim());
  i = first < 0 ? 0 : first;
  const done = rows.filter(x => (x.rip_present || "").trim()).length;
  status(done ? ("resumed at frame " + (i+1) + " — " + done + " already on disk")
              : "ready.", false);
  draw();
}).catch(e => status("could not load labels.csv: " + e.message, true));
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
                                              payload.get("notes", ""),
                                              payload.get("box_labels"),
                                              log=True)
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
    parser.add_argument("--import", dest="import_path", default=None,
                        metavar="FILE",
                        help="merge labels from a JSON dump and exit")
    parser.add_argument("--import-overwrite", action="store_true",
                        help="let the import replace verdicts already in the CSV")
    args = parser.parse_args()

    if not os.path.exists(LABEL_CSV):
        sys.exit(f"{LABEL_CSV} does not exist — run build_label_sample.py first.")

    if args.import_path:
        applied, skipped, problems = import_labels(args.import_path,
                                                   args.import_overwrite)
        print(f"\n  imported {applied} label(s) from {args.import_path}")
        if skipped:
            print(f"  kept {skipped} verdict(s) already in the CSV "
                  "(pass --import-overwrite to replace them)")
        for problem in problems[:20]:
            print(f"  skipped: {problem}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        rows, _ = read_rows()
        print(f"  {count_done(rows)} of {len(rows)} rows now carry a verdict")
        return 0

    rows, _ = read_rows()
    done = count_done(rows)
    missing = [r["image"] for r in rows
               if not os.path.isfile(os.path.join(IMAGE_DIR, r.get("image", "")))]

    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  {len(rows)} rows, {done} already labelled")
    if missing:
        print(f"  WARNING: {len(missing)} rows name an image that is not in {IMAGE_DIR}/")
        print(f"           first missing: {missing[0]}")
    print(f"  serving {url}   (ctrl-C to stop)")
    print("  every save is logged below with a running count; if the browser "
          "says saved\n  and no line appears here, the write did not happen.\n")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        # Re-read rather than trusting the count this process started with: the
        # old version printed the STARTUP figure on the way out, so a session
        # that saved forty labels still said "0 labelled" at the end and looked
        # like total data loss when nothing had been lost at all.
        rows, _ = read_rows()
        print(f"\n  stopped. {_saves} save(s) this session; "
              f"{count_done(rows)} of {len(rows)} rows carry a verdict "
              f"in {LABEL_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
