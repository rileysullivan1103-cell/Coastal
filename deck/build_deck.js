const pptxgen = require("pptxgenjs");

const INK="12212B", PAPER="F7F6F3", TEAL="3E7C8C", ACC="E4572E",
      MUTE="7C8F99", LINE="D9D6CE", GOOD="2C6E49";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.3 x 7.5
pres.author = "Coastal monitoring";
pres.title  = "Rips, Runoff and What Replicates";

const W = 13.3, M = 0.7, CW = W - 2*M;

function dark(s){ s.background = { color: INK }; }
function light(s){ s.background = { color: PAPER }; }

// eyebrow + title block used on every content slide
function head(s, eyebrow, title){
  s.addText(eyebrow.toUpperCase(), { x:M, y:0.42, w:CW, h:0.24, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:11, bold:true, charSpacing:2.2, color:ACC });
  s.addText(title, { x:M, y:0.70, w:CW, h:0.72, isTextBox:true, margin:0,
    fontFace:"Cambria", fontSize:31, bold:true, color:INK });
}

// "n = 96" style chip — the recurring motif; sample size is what carries this deck
function chip(s, x, y, text, fill){
  s.addShape(pres.ShapeType.roundRect, { x, y, w:1.15, h:0.30, rectRadius:0.15,
    fill:{ color: fill || TEAL } });
  s.addText(text, { x, y, w:1.15, h:0.30, isTextBox:true, margin:0, align:"center",
    fontFace:"Calibri", fontSize:10.5, bold:true, color:"FFFFFF" });
}

function note(s, text, y){
  s.addText(text, { x:M, y:y, w:CW, h:0.9, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:11.5, color:MUTE, lineSpacing:16 });
}

/* ---------------------------------------------------------------- 1 title */
let s = pres.addSlide(); dark(s);
s.addText("Findings brief   ·   updated with the controls applied", { x:M, y:1.5, w:CW, h:0.3,
  isTextBox:true, margin:0, fontFace:"Calibri", fontSize:12.5, charSpacing:1.6, color:TEAL });
s.addText("Rips, Runoff, and\nWhat Actually Replicates", { x:M, y:1.95, w:9.6, h:1.9,
  isTextBox:true, margin:0, fontFace:"Cambria", fontSize:47, bold:true, color:"FFFFFF",
  lineSpacing:52 });
s.addText("Four beach cameras, eight bacteria stations and 2,815 hand-annotated European frames, run against measured and modelled ocean conditions.",
  { x:M, y:4.05, w:8.4, h:0.9, isTextBox:true, margin:0, fontFace:"Calibri", fontSize:15,
    color:"C9D4DA", lineSpacing:23 });
s.addShape(pres.ShapeType.rect, { x:M, y:5.25, w:1.5, h:0.035, fill:{ color:ACC } });
s.addText("One association survives every control and repeats on two coasts.\nSeveral that looked strong were measuring which beach the sample came from.",
  { x:M, y:5.55, w:8.4, h:0.9, isTextBox:true, margin:0, fontFace:"Cambria", fontSize:14.5,
    italic:true, color:"FFFFFF", lineSpacing:24 });
s.addNotes("Every number here is a Spearman rank correlation after removing per-month means, or per-site means where stated. n is on every claim.");

/* ------------------------------------------------------------- 2 what's in */
s = pres.addSlide(); light(s);
head(s, "What is in the analysis", "Four independent bodies of evidence");
const src = [
  { t:"Camera rip detection", a:"WebCOOS \u00b7 YOLOv8", n:"7,393 hours",
    b:"Walton Lighthouse (CA) and Virginia Beach. Hours with no detection were reconstructed from the stills feed, so absence means \u2018looked and saw nothing\u2019." },
  { t:"Beach bacteria", a:"CA CKAN + national WQP", n:"750 samples",
    b:"Eight stations, four analytes, 2023\u20132026. Enterococcus, E. coli, total and fecal coliform, as log10 counts." },
  { t:"Hand-annotated rips", a:"RipAID v1.0.0 \u00b7 Balearics", n:"2,815 frames",
    b:"Thirteen years, eight cameras at Cala Millor and Son Bou. 1,959 rips drawn by people, plus 915 marked \u2018doubt\u2019." },
  { t:"Rip casualties", a:"NOAA Storm Events", n:"151 days", f:ACC,
    b:"New Hanover NC and Palm Beach FL, 2000\u20132026. A day a rip killed or injured somebody \u2014 the only outcome here not produced by our own instrument." },
];
src.forEach((c,i)=>{
  const colw = CW/4 - 0.30;
  const x = M + i*(CW/4);
  s.addShape(pres.ShapeType.rect, { x:x, y:1.85, w:colw, h:0.028, fill:{ color: c.f || TEAL } });
  s.addText(c.t, { x:x, y:2.02, w:colw, h:0.4, isTextBox:true, margin:0,
    fontFace:"Cambria", fontSize:16.5, bold:true, color:INK });
  s.addText(c.a, { x:x, y:2.44, w:colw, h:0.28, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:10.5, color:MUTE });
  chip(s, x, 2.80, c.n, c.f);
  s.addText(c.b, { x:x, y:3.30, w:colw, h:2.3, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:11.5, color:INK, lineSpacing:17 });
});
note(s, "The cameras and the bacteria stations were matched by a scan of all 86 WebCOOS cameras: 68 have a bacteria station within 2 km, and 58 also have a buoy, a tide gauge and a rain gauge. The casualty record is filed against a stretch of county coast, not a point, so it is never joined to a camera \u2014 only compared against one.", 5.85);

/* --------------------------------------------------------- 3 THE HEADLINE */
s = pres.addSlide(); light(s);
head(s, "The one result that repeats", "Rain drives bacteria — at some beaches");
const stats = [
  { v:"0.47", l:"Enterococcus", w:"rain, 72 h", site:"Santa Cruz Wharf", n:"n = 44", p:"p = 0.001", c:GOOD },
  { v:"0.45", l:"Total coliform", w:"rain, 72 h", site:"Santa Cruz Wharf", n:"n = 45", p:"p = 0.002", c:GOOD },
  { v:"0.43", l:"Fecal coliform", w:"rain, 72 h", site:"Santa Cruz Wharf", n:"n = 45", p:"p = 0.003", c:GOOD },
  { v:"0.32", l:"Enterococcus", w:"rain, 24 h", site:"Virginia Beach", n:"n = 96", p:"p = 0.001", c:ACC },
];
stats.forEach((d,i)=>{
  const x = M + i*(CW/4);
  s.addText(d.v, { x:x, y:1.85, w:CW/4-0.3, h:1.0, isTextBox:true, margin:0,
    fontFace:"Cambria", fontSize:60, bold:true, color:d.c });
  s.addText(d.l, { x:x, y:2.95, w:CW/4-0.3, h:0.3, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:14, bold:true, color:INK });
  s.addText(d.w + "   ·   " + d.site, { x:x, y:3.25, w:CW/4-0.3, h:0.3, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:11.5, color:MUTE });
  s.addText(d.n + "     " + d.p, { x:x, y:3.55, w:CW/4-0.3, h:0.3, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:11, color:TEAL });
});
s.addShape(pres.ShapeType.rect, { x:M, y:4.25, w:CW, h:0.02, fill:{ color:LINE } });
s.addText("Three analytes at one Californian site, and a fourth at a site 4,500 km away on the Atlantic. Same sign, same control, independent sampling programmes.",
  { x:M, y:4.5, w:7.4, h:0.9, isTextBox:true, margin:0, fontFace:"Cambria", fontSize:16,
    color:INK, lineSpacing:26 });
s.addShape(pres.ShapeType.roundRect, { x:8.35, y:4.4, w:CW-7.65, h:1.75, rectRadius:0.06,
  fill:{ color:"FFFFFF" }, line:{ color:LINE, width:1 } });
s.addText("…and it is not a coastal law", { x:8.65, y:4.6, w:4.1, h:0.3, isTextBox:true,
  margin:0, fontFace:"Calibri", fontSize:12, bold:true, color:ACC });
s.addText("Carpinteria State Beach runs the other way: total coliform vs 48-72 h rain is −0.32 (n = 42, p = 0.036), same control. Two sites, two signs. This is a fact about a particular outfall, not about rain.",
  { x:8.65, y:4.95, w:4.1, h:1.1, isTextBox:true, margin:0, fontFace:"Calibri",
    fontSize:11.5, color:INK, lineSpacing:17 });
note(s, "All four figures are after removing per-month means from both sides, because agencies sample in swim season and California rain is seasonal.", 6.35);

/* ----------------------------------------------------------- 4 correction */
s = pres.addSlide(); light(s);
head(s, "What the controls overturned", "Half the pooled signals were naming a beach");
s.addText("Enterococcus, 326 samples across 8 sites", { x:M, y:1.62, w:CW, h:0.3,
  isTextBox:true, margin:0, fontFace:"Calibri", fontSize:12.5, color:MUTE });
const rows = [
  ["Predictor","Pooled, month-controlled","Within site","Verdict"],
  ["Wave height (buoy)","−0.37","−0.005","between-beach artefact"],
  ["Water temperature","+0.25","+0.07","between-beach artefact"],
  ["Tide level","−0.12","+0.01","between-beach artefact"],
  ["Rain, 72 h","+0.18","+0.15","survives"],
  ["Swell period","+0.11","+0.19","survives"],
];
const colw = [3.6, 3.3, 2.4, 2.6];
rows.forEach((r,ri)=>{
  const y = 2.1 + ri*0.52;
  if (ri === 0){
    s.addShape(pres.ShapeType.rect, { x:M, y:y+0.42, w:CW, h:0.02, fill:{ color:INK } });
  } else if (ri < rows.length){
    s.addShape(pres.ShapeType.rect, { x:M, y:y+0.44, w:CW, h:0.01, fill:{ color:LINE } });
  }
  let x = M;
  r.forEach((cell,ci)=>{
    const isHead = ri === 0;
    const survives = r[3] === "survives";
    let col = INK, bold = isHead;
    if (!isHead && ci === 2) { col = survives ? GOOD : ACC; bold = true; }
    if (!isHead && ci === 3) { col = survives ? GOOD : MUTE; }
    s.addText(cell, { x:x, y:y, w:colw[ci]-0.15, h:0.42, isTextBox:true, margin:0,
      fontFace: isHead ? "Calibri" : (ci===0 ? "Calibri" : "Consolas"),
      fontSize: isHead ? 11.5 : 14, bold:bold, color:col, valign:"middle",
      charSpacing: isHead ? 1.2 : 0 });
    x += colw[ci];
  });
});
s.addText("A predictor that is strong pooled and zero within a beach was ranking beaches, not conditions: the sheltered sites have both higher counts and smaller waves.",
  { x:M, y:5.35, w:CW, h:0.6, isTextBox:true, margin:0, fontFace:"Cambria", fontSize:15,
    color:INK, lineSpacing:24 });
note(s, "For E. coli the collapse is total — the strongest within-site correlation across 167 samples is 0.09, and the run says so in as many words. One caution the tables carry: wave height is only measured at one of the eight sites, so its within-site column is doing no work.", 6.05);

/* ------------------------------------------------------------ 5 rip: 2 cams */
s = pres.addSlide(); light(s);
head(s, "Camera rip detection", "Two cameras, two very different answers");
const cams = [
  { name:"Virginia Beach", sub:"Apr–Aug 2026 · 1,924 hours analysed · 1,194 observed zeros",
    col:ACC,
    rows:[["wave height (model)","0.40"],["air temperature","−0.36"],["tide level","0.20"],
          ["onshore wind","0.19"],["onshore swell (model)","0.15"]],
    foot:"R² = 0.24 across all predictors" },
  { name:"Walton Lighthouse", sub:"Sep 2025–Aug 2026 · 4,843 hours analysed · 2,295 observed zeros",
    col:TEAL,
    rows:[["precipitation","0.15"],["tide level","0.14"],["rain, 48 h","0.11"],
          ["onshore swell (buoy)","0.08"],["wave height (buoy)","0.04"]],
    foot:"“strongest |rho| is 0.15 — nothing here is a strong driver”" },
];
cams.forEach((c,i)=>{
  const x = M + i*(CW/2 + 0.15);
  const w = CW/2 - 0.15;
  s.addShape(pres.ShapeType.rect, { x:x, y:1.72, w:w, h:0.035, fill:{ color:c.col } });
  s.addText(c.name, { x:x, y:1.88, w:w, h:0.4, isTextBox:true, margin:0,
    fontFace:"Cambria", fontSize:20, bold:true, color:INK });
  s.addText(c.sub, { x:x, y:2.30, w:w, h:0.3, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:11, color:MUTE });
  c.rows.forEach((r,j)=>{
    const y = 2.78 + j*0.45;
    s.addText(r[0], { x:x, y:y, w:w-1.5, h:0.36, isTextBox:true, margin:0,
      fontFace:"Calibri", fontSize:13.5, color:INK, valign:"middle" });
    s.addText(r[1], { x:x+w-1.5, y:y, w:1.5, h:0.36, isTextBox:true, margin:0, align:"right",
      fontFace:"Consolas", fontSize:15, bold:j===0, color:j===0 ? c.col : INK, valign:"middle" });
    s.addShape(pres.ShapeType.rect, { x:x, y:y+0.38, w:w, h:0.008, fill:{ color:LINE } });
  });
  s.addText(c.foot, { x:x, y:5.12, w:w, h:0.4, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:11.5, italic:true, color:MUTE });
});
s.addText("Detection rate vs conditions, after removing hour-of-day and month. Tide is the one driver that agrees across both coasts.",
  { x:M, y:5.75, w:CW, h:0.4, isTextBox:true, margin:0, fontFace:"Cambria", fontSize:14.5, color:INK });
note(s, "Both use a stills-feed denominator so hours with imagery and no detection count as observed zeros.", 6.25);

/* ------------------------------------------------------- 6 the wave puzzle */
s = pres.addSlide(); light(s);
head(s, "Reading the disagreement", "The wave contradiction is about instruments, not physics");
s.addShape(pres.ShapeType.roundRect, { x:M, y:1.8, w:5.9, h:2.6, rectRadius:0.06,
  fill:{ color:"FFFFFF" }, line:{ color:LINE, width:1 } });
s.addText("Virginia Beach", { x:M+0.3, y:2.0, w:5.3, h:0.35, isTextBox:true, margin:0,
  fontFace:"Cambria", fontSize:17, bold:true, color:INK });
s.addText("0.40", { x:M+0.3, y:2.42, w:2.0, h:0.75, isTextBox:true, margin:0,
  fontFace:"Cambria", fontSize:44, bold:true, color:ACC });
s.addText("Wave height is the top driver of all four rip targets. The source is an Open-Meteo model cell directly offshore, present in 100% of analysed hours.",
  { x:M+2.4, y:2.42, w:3.2, h:1.6, isTextBox:true, margin:0, fontFace:"Calibri",
    fontSize:12.5, color:INK, lineSpacing:18 });

s.addShape(pres.ShapeType.roundRect, { x:M+6.2, y:1.8, w:5.9, h:2.6, rectRadius:0.06,
  fill:{ color:"FFFFFF" }, line:{ color:LINE, width:1 } });
s.addText("Walton Lighthouse", { x:M+6.5, y:2.0, w:5.3, h:0.35, isTextBox:true, margin:0,
  fontFace:"Cambria", fontSize:17, bold:true, color:INK });
s.addText("0.04", { x:M+6.5, y:2.42, w:2.0, h:0.75, isTextBox:true, margin:0,
  fontFace:"Cambria", fontSize:44, bold:true, color:MUTE });
s.addText("Wave height does nothing (p = 0.20). The source is a buoy 23 km away that reports a wave height in only 43% of the analysed hours.",
  { x:M+8.6, y:2.42, w:3.2, h:1.6, isTextBox:true, margin:0, fontFace:"Calibri",
    fontSize:12.5, color:INK, lineSpacing:18 });

s.addShape(pres.ShapeType.rect, { x:M, y:4.75, w:0.05, h:1.35, fill:{ color:ACC } });
s.addText("Walton's wave null is most likely a measurement failure, not evidence that waves do not drive rips.",
  { x:M+0.35, y:4.72, w:CW-0.35, h:0.45, isTextBox:true, margin:0, fontFace:"Cambria",
    fontSize:17, bold:true, color:INK });
s.addText("I previously reported that null as a finding. With a second camera in hand it reads differently: the site with the closer, more complete wave record is the site where waves matter. Before Walton is used to argue anything about waves, it needs a wave source that covers its hours.",
  { x:M+0.35, y:5.20, w:CW-1.2, h:0.95, isTextBox:true, margin:0, fontFace:"Calibri",
    fontSize:13, color:INK, lineSpacing:20 });
note(s, "Naming convention throughout: ALL-CAPS columns are measured by an instrument, lower_snake_case are model reanalysis.", 6.35);

/* ------------------------------------------------------------- 7 RipAID bias */
s = pres.addSlide(); dark(s);
s.addText("RIPAID · 2,815 FRAMES · 2011–2024", { x:M, y:1.25, w:CW, h:0.28, isTextBox:true,
  margin:0, fontFace:"Calibri", fontSize:11, bold:true, charSpacing:2.2, color:ACC });
s.addText("The question this dataset cannot answer", { x:M, y:1.62, w:10.5, h:0.85,
  isTextBox:true, margin:0, fontFace:"Cambria", fontSize:33, bold:true, color:"FFFFFF" });
s.addText("“After visual inspection, most of the images that did not show rip currents were removed.”",
  { x:M+0.35, y:2.75, w:9.6, h:0.75, isTextBox:true, margin:0, fontFace:"Cambria",
    fontSize:19, italic:true, color:"FFFFFF", lineSpacing:28 });
s.addShape(pres.ShapeType.rect, { x:M, y:2.78, w:0.05, h:0.68, fill:{ color:ACC } });
s.addText("RipAID v1.0.0 README, section 2.1", { x:M+0.35, y:3.52, w:9.6, h:0.28,
  isTextBox:true, margin:0, fontFace:"Calibri", fontSize:11, color:TEAL });

const bias = [
  ["Frames were pulled at lifeguard rip sightings, ±2 h.",
   "Every hour in the set is an hour a lifeguard was on duty and reported a rip. Staffing and rip occurrence cannot be separated."],
  ["Most no-rip frames were then deleted by hand.",
   "The 1,288 frames without a rip are the residue of a deletion, not a control group. They are not the hours when no rip happened."],
];
bias.forEach((b,i)=>{
  const x = M + i*(CW/2 + 0.2);
  s.addText(String(i+1), { x:x, y:4.15, w:0.4, h:0.4, isTextBox:true, margin:0,
    fontFace:"Cambria", fontSize:22, bold:true, color:ACC });
  s.addText(b[0], { x:x+0.5, y:4.15, w:CW/2-0.7, h:0.5, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:14, bold:true, color:"FFFFFF", lineSpacing:20 });
  s.addText(b[1], { x:x+0.5, y:4.72, w:CW/2-0.7, h:1.1, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:12.5, color:"C9D4DA", lineSpacing:19 });
});
s.addText("So “what conditions predict a rip being present” would measure the curator's selection at least as much as the ocean. This is not a flaw — the set was built to train detectors, and the enrichment is a feature for that.",
  { x:M, y:6.1, w:CW, h:0.7, isTextBox:true, margin:0, fontFace:"Calibri", fontSize:12.5,
    italic:true, color:"8FA3AD", lineSpacing:19 });

/* -------------------------------------------------------- 8 RipAID result */
s = pres.addSlide(); light(s);
head(s, "RipAID, asked properly", "Within the annotated rips: essentially nothing");
s.addText("Restricting to hours that contain a hand-drawn rip leaves a question the selection does not destroy — given a rip is there, does its size or orientation track the conditions? Across the four best-sampled cameras, almost nothing survives.",
  { x:M, y:1.62, w:CW, h:0.65, isTextBox:true, margin:0, fontFace:"Calibri", fontSize:13.5,
    color:INK, lineSpacing:21 });
const rip = [
  ["Camera","Rip hours","Strongest driver of box area","ρ","Waves usable in"],
  ["snb-s-01","387","air temperature","−0.28","108 hrs"],
  ["snb-s-03","339","rain, 24 h","0.13","74 hrs"],
  ["clm-s-01","317","onshore swell (model)","0.19","75 hrs"],
  ["clm-s-02","244","swell height (model)","0.20","41 hrs"],
];
const rw = [2.3, 1.9, 4.4, 1.5, 2.0];
rip.forEach((r,ri)=>{
  const y = 2.5 + ri*0.5;
  if (ri===0) s.addShape(pres.ShapeType.rect, { x:M, y:y+0.4, w:CW, h:0.02, fill:{ color:INK } });
  else s.addShape(pres.ShapeType.rect, { x:M, y:y+0.42, w:CW, h:0.01, fill:{ color:LINE } });
  let x = M;
  r.forEach((cell,ci)=>{
    s.addText(cell, { x:x, y:y, w:rw[ci]-0.15, h:0.4, isTextBox:true, margin:0,
      fontFace: ri===0 ? "Calibri" : (ci===0||ci===2 ? "Calibri" : "Consolas"),
      fontSize: ri===0 ? 11 : 13.5, bold: ri===0, valign:"middle",
      color: ri===0 ? INK : (ci===3 ? (Math.abs(parseFloat(cell))>=0.25 ? ACC : MUTE) : INK),
      charSpacing: ri===0 ? 1.2 : 0 });
    x += rw[ci];
  });
});
s.addShape(pres.ShapeType.roundRect, { x:M, y:5.15, w:CW, h:1.35, rectRadius:0.06,
  fill:{ color:"FFFFFF" }, line:{ color:LINE, width:1 } });
s.addText("Two things limit this before the numbers do.", { x:M+0.3, y:5.3, w:CW-0.6, h:0.3,
  isTextBox:true, margin:0, fontFace:"Calibri", fontSize:12.5, bold:true, color:ACC });
s.addText("Box area is in pixels, and cross-shore resolution runs 0.2–15 m across these cameras, so areas are comparable within a camera and meaningless pooled. And the wave reanalysis is populated for only 41–108 of each camera's rip hours, so every wave figure above rests on roughly a quarter of the data — check that before believing any of them.",
  { x:M+0.3, y:5.62, w:CW-0.6, h:0.85, isTextBox:true, margin:0, fontFace:"Calibri",
    fontSize:12, color:INK, lineSpacing:18 });

/* --------------------------------------------- 9 storm: one coastline */
s = pres.addSlide(); light(s);
head(s, "An outcome nobody here measured", "One coastline, opposite answers");
s.addText("NOAA logs a rip-current event when one hurts somebody. It has real zeros and it was written by people who never saw our cameras \u2014 so it can check them. These are two Atlantic beaches 800 km apart, not two oceans. Both are ranked within month and weekday, because Saturday alone holds 31 of New Hanover\u2019s 72 events.",
  { x:M, y:1.58, w:CW, h:0.62, isTextBox:true, margin:0, fontFace:"Calibri",
    fontSize:12.5, color:INK, lineSpacing:19 });

const zCX = [M, M+4.35, M+6.55, M+8.75];
const zCWID = [4.15, 2.0, 2.0, CW-8.75];
const zHdr = ["", "New Hanover, NC", "Palm Beach, FL", "they disagree by"];
zHdr.forEach((h,i)=>{
  if(!h) return;
  s.addText(h, { x:zCX[i], y:2.42, w:zCWID[i], h:0.3, isTextBox:true, margin:0,
    align: i===3 ? "left" : "right",
    fontFace:"Calibri", fontSize:10.5, bold:true, charSpacing:1.2, color:MUTE });
});
chip(s, zCX[1]+zCWID[1]-1.15, 2.74, "n = 72");
chip(s, zCX[2]+zCWID[2]-1.15, 2.74, "n = 79", ACC);

const zRows = [
  ["wind speed",       "\u22120.042", "+0.069", "z = 6.5"],
  ["wave height",      "\u22120.008", "+0.074", "z = 4.8"],
  ["wave height, peak","\u22120.024", "+0.068", "z = 5.5"],
  ["wave period",      "+0.092", "+0.005", "z = 5.1"],
];
zRows.forEach((r,i)=>{
  const y = 3.28 + i*0.53;
  if(i % 2 === 0){
    s.addShape(pres.ShapeType.rect, { x:M-0.12, y:y-0.06, w:CW+0.24, h:0.48,
      fill:{ color:"EFEDE6" } });
  }
  s.addText(r[0], { x:zCX[0], y:y, w:zCWID[0], h:0.36, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:13.5, color:INK });
  s.addText(r[1], { x:zCX[1], y:y, w:zCWID[1], h:0.36, isTextBox:true, margin:0, align:"right",
    fontFace:"Consolas", fontSize:14, bold:true,
    color: r[1].charAt(0) === "+" ? ACC : TEAL });
  s.addText(r[2], { x:zCX[2], y:y, w:zCWID[2], h:0.36, isTextBox:true, margin:0, align:"right",
    fontFace:"Consolas", fontSize:14, bold:true,
    color: r[2].charAt(0) === "+" ? ACC : TEAL });
  s.addText(r[3], { x:zCX[3], y:y, w:zCWID[3], h:0.36, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:12, color:MUTE });
});

s.addText("A casualty day in New Hanover is long swell under light wind \u2014 calm-looking. In Palm Beach it is a windy day with a short, steep sea. The seasons agree: July\u2013August in Carolina, April\u2013May in Florida.",
  { x:M, y:5.62, w:CW, h:0.5, isTextBox:true, margin:0, fontFace:"Calibri",
    fontSize:13, color:INK, lineSpacing:20 });
note(s, "Neither zone has a strong driver: the largest controlled correlation in either is 0.09. What is significant is that they point opposite ways on one shoreline \u2014 the wind term is not merely absent at the second beach, it is reversed.", 6.25);

/* ----------------------------------------------------------- 10 bottom line */
s = pres.addSlide(); dark(s);
s.addText("THE BOTTOM LINE", { x:M, y:1.15, w:CW, h:0.3, isTextBox:true, margin:0,
  fontFace:"Calibri", fontSize:11, bold:true, charSpacing:2.2, color:ACC });
const bl = [
  ["What replicates","Rain against bacteria at Santa Cruz Wharf: three analytes, same sign, \u03c1 \u2248 0.45 after the season control \u2014 and it repeats at Virginia Beach, an independent sampling programme on the opposite coast.", GOOD],
  ["What does not transfer","The sign flips between beaches. Rain against enterococcus is +0.47 at Santa Cruz and \u22120.32 at Carpinteria; wind against a rip casualty is \u22120.04 in New Hanover and +0.07 in Palm Beach, two Atlantic beaches 800 km apart. Unrelated outcomes, unrelated programmes, and the sign reverses within a single coastline as readily as between oceans: a driver fitted at one beach should not be deployed at another.", ACC],
  ["Worth acting on","Nothing yet. The strongest effect anywhere in this project explains under a quarter of the variance in one target at one camera, and no rip driver survives at two beaches at once.", MUTE],
];
bl.forEach((b,i)=>{
  const y = 1.70 + i*1.62;
  s.addShape(pres.ShapeType.rect, { x:M, y:y+0.05, w:0.05, h:1.30, fill:{ color:b[2] } });
  s.addText(b[0], { x:M+0.35, y:y, w:3.0, h:0.4, isTextBox:true, margin:0,
    fontFace:"Cambria", fontSize:19, bold:true, color:"FFFFFF" });
  s.addText(b[1], { x:M+3.6, y:y, w:CW-3.6, h:1.45, isTextBox:true, margin:0,
    fontFace:"Calibri", fontSize:13, color:"C9D4DA", lineSpacing:20 });
});
s.addShape(pres.ShapeType.rect, { x:M, y:6.5, w:CW, h:0.015, fill:{ color:"2C4652" } });
s.addText("Every figure is reproducible from the repository: analyze_drivers.py --target wq  ·  --target rip  ·  --positives-only for RipAID  ·  analyze_storm.py --zone --camera",
  { x:M, y:6.72, w:CW, h:0.4, isTextBox:true, margin:0, fontFace:"Calibri", fontSize:11,
    color:"7C8F99" });

const OUT = require("path").join(__dirname, "Rips-and-Runoff.pptx");
pres.writeFile({ fileName: OUT })
  .then(f => console.log("wrote", f));
