import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Build the eClinicalMedicine submission manuscript (Lancet house style).

Inputs : src_v4.md + src_back_v4.md (placeholders {key} filled from facts.json; citations [@Key]),
         refs_v4.json (Crossref-verified metadata), results/table1.csv, results/table2.csv,
         figs/output_v7/FigureN/FigureN_composite.png
Outputs: output/eClinicalMedicine_manuscript.docx / .md, output/word_counts.txt, output/reference_list.csv
"""
import json, re, csv, os, html
from PIL import Image
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output"); os.makedirs(OUT, exist_ok=True)
RES = _os.path.join(HB_ROOT, "results")
FIG = _os.path.join(HB_ROOT, "figs/output_v7")
REFS = json.load(open(os.path.join(HERE, "refs_v4.json")))
FACTS = json.load(open(os.path.join(HERE, "facts.json")))
FONT = "Times New Roman"

# ----------------------------------------------------------------------------- Lancet / Vancouver references
JABBR = {
    "The Lancet Neurology": "Lancet Neurol", "Circulation": "Circulation", "European Heart Journal": "Eur Heart J",
    "Nature Reviews Cardiology": "Nat Rev Cardiol", "JAMA Neurology": "JAMA Neurol", "Science": "Science",
    "Nature Communications": "Nat Commun", "Brain": "Brain", "PLOS Medicine": "PLoS Med",
    "Journal of the American College of Cardiology": "J Am Coll Cardiol", "Nature Medicine": "Nat Med",
    "Nature": "Nature", "European Journal of Preventive Cardiology": "Eur J Prev Cardiol", "SoftwareX": "SoftwareX",
    "Nature Reviews Neuroscience": "Nat Rev Neurosci",
    "American Journal of Respiratory and Critical Care Medicine": "Am J Respir Crit Care Med", "BMJ": "BMJ",
    "NeuroImage": "Neuroimage", "Medical Decision Making": "Med Decis Making",
    "American Journal of Epidemiology": "Am J Epidemiol",
}
GROUP_AUTHOR = {"GBD_stroke_2024": "GBD 2021 Stroke Risk Factor Collaborators",
                "SCORE2_2021": "SCORE2 working group and ESC Cardiovascular risk collaboration",
                "SCORE2OP_2021": "SCORE2-OP working group and ESC Cardiovascular risk collaboration"}
OVERRIDE = {"Martin_AHA_2025": {"page": "e41-e660"}, "Debette_JAMANeurol_2019": {"page": "81-94"},
            "Masci_EHJ_2025": {"online": "Oct 9"}}


def initials(given):
    given = given.replace(".", " ").strip()
    out = []
    for part in given.split():
        sub = [p for p in part.split("-") if p]
        out.append("-".join(p[0].upper() for p in sub))
    return "".join(out)


def author_str(key, r):
    if key in GROUP_AUTHOR:
        return GROUP_AUTHOR[key]
    au = [a for a in r["authors"] if a[0]]
    n = r.get("n_authors", len(au))
    fmt = lambda a: f"{a[0]} {initials(a[1])}".strip() if a[1] else a[0]
    if n <= 6 and len(au) >= n:
        return ", ".join(fmt(a) for a in au[:n])
    return ", ".join(fmt(a) for a in au[:3]) + ", et al"


def clean_title(t):
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    t = re.sub(r"\s+", " ", t).strip().replace("\u2010", "-")
    t = re.sub(r"\s+:", ":", t)
    return t.rstrip(".")


def ref_text(key):
    r = dict(REFS[key]); r.update(OVERRIDE.get(key, {}))
    au = author_str(key, r)
    title = clean_title(r["title"])
    j = JABBR.get(html.unescape(r["journal"]), r["journal"])
    vol, page, yr = r.get("volume"), r.get("page"), r["year"]
    tend = "" if title.endswith(("?", "!")) else "."
    if r.get("online"):
        tail = f"{yr}; published online {r['online']}. https://doi.org/{r['doi']}."
    elif vol and page:
        tail = f"{yr}; {vol}: {page.replace('-', '–')}."
    elif vol:
        tail = f"{yr}; {vol}. https://doi.org/{r['doi']}."
    else:
        tail = f"{yr}. https://doi.org/{r['doi']}."
    return f"{au}. {title}{tend} ", j, f" {tail}"


# ----------------------------------------------------------------------------- source
src = open(os.path.join(HERE, "src_v4.md")).read() + "\n" + open(os.path.join(HERE, "src_back_v4.md")).read()
missing = sorted(set(k for k in re.findall(r"\{([A-Za-z][A-Za-z0-9_]*)\}", src) if k not in FACTS))
if missing:
    raise SystemExit(f"unresolved placeholders: {missing}")
src = re.sub(r"\{([A-Za-z][A-Za-z0-9_]*)\}", lambda m: str(FACTS[m.group(1)]), src)


def bracket_nested(text):
    """Lancet/eCM style: parentheses inside parentheses become square brackets (depth tracked per line)."""
    out = []
    for line in text.split("\n"):
        depth, buf = 0, []
        for ch in line:
            if ch == "(":
                depth += 1; buf.append("(" if depth == 1 else "[")
            elif ch == ")" and depth > 0:
                buf.append(")" if depth == 1 else "]"); depth -= 1
            else:
                buf.append(ch)
        out.append("".join(buf))
    return "\n".join(out)


src = bracket_nested(src)

CITE_RE = re.compile(r"\s?\[@([^\]]+)\]")
order = []


def compress(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 1 and j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


def _cite(m):
    nums = []
    for k in [k.strip().lstrip("@") for k in m.group(1).split(";")]:
        if k not in REFS:
            raise KeyError(f"citation key not verified: {k}")
        if k not in order:
            order.append(k)
        nums.append(order.index(k) + 1)
    return "«SUP:" + compress(nums) + "»"


blocks = []
tag = None
for line in src.splitlines():
    if line.startswith("%% "):
        tag = line[3:].strip(); blocks.append([tag, []]); continue
    if line.strip():
        blocks[-1][1].append(line.strip())
B = {t: lines for t, lines in blocks}

# citations are numbered in order of first appearance in the reading order of the manuscript
READING = [t for t, _ in blocks if not t.startswith(("SUMMARY", "RIC", "TITLE"))]
for t in READING:
    B[t] = [CITE_RE.sub(_cite, p) for p in B[t]]
for t in [t for t, _ in blocks if t.startswith(("SUMMARY", "RIC", "TITLE"))]:
    if any("[@" in p for p in B[t]):
        raise SystemExit(f"references not allowed in {t}")


def plain(t):
    t = re.sub(r"«SUP:[^»]*»", "", t)
    return t.replace("*", "")


def words(t):
    return len(re.findall(r"[A-Za-z0-9\u00b7\u2013\-–≥≤%]+", plain(t)))


# ----------------------------------------------------------------------------- docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Cm(2.5))
st = doc.styles["Normal"]
st.font.name = FONT; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(0)
st.paragraph_format.line_spacing = 2.0

# continuous line numbers (useful for reviewers)
ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); ln.set(qn("w:distance"), "283")
sec._sectPr.append(ln)
# page numbers in footer
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    r = fp.add_run()
    if kind:
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), kind); r._r.append(e)
    else:
        e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; r._r.append(e)
    r.font.size = Pt(9)

TOKEN = re.compile(r"(«SUP:[^»]*»|\*[^*]+\*|\[AUTHOR ACTION[^\]]*\])")


def add_runs(p, text, size=None, bold=False, italic=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("«SUP:"):
            r = p.add_run(part[5:-1]); r.font.superscript = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = p.add_run(part[1:-1]); r.italic = True
        elif part.startswith("[AUTHOR ACTION"):
            r = p.add_run(part); r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        else:
            r = p.add_run(part)
        r.bold = r.bold or bold
        if italic:
            r.italic = True
        if size:
            r.font.size = Pt(size)
    return p


def para(text="", size=None, bold=False, italic=False, align=None, after=0, before=0, spacing=None, keep=False):
    p = doc.add_paragraph()
    if align == "c":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format; pf.space_after = Pt(after); pf.space_before = Pt(before)
    if PB[0]:
        pf.page_break_before = True; PB[0] = False
    if spacing:
        pf.line_spacing = spacing
    if keep:
        pf.keep_with_next = True
    add_runs(p, text, size=size, bold=bold, italic=italic)
    return p


def heading(text, level=1):
    return para(text, size=12 if level == 1 else 11, bold=True, italic=(level == 2), before=12 if level == 1 else 6,
                after=0, keep=True)


PB = [False]


def page_break():
    PB[0] = True   # applied as page-break-before on the next paragraph (avoids blank pages)


def set_widths(t, widths):
    tblPr = t._tbl.tblPr
    tw = tblPr.find(qn("w:tblW"))
    if tw is None:
        tw = OxmlElement("w:tblW"); tblPr.append(tw)
    tw.set(qn("w:w"), str(int(sum(widths) * 567))); tw.set(qn("w:type"), "dxa")
    grid = t._tbl.tblGrid
    for gc, wcm in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(wcm * 567)))
    for row in t.rows:
        for j, c in enumerate(row.cells):
            c.width = Cm(widths[min(j, len(widths) - 1)])


def no_lines(p):
    pPr = p._p.get_or_add_pPr(); s = OxmlElement("w:suppressLineNumbers"); pPr.append(s)


def cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders"); tcPr.append(b)
    for edge, val in kw.items():
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), val); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000")
        b.append(e)


def cell_text(cell, text, bold=False, size=8.5, align=None, indent=False):
    cell.text = ""
    p = cell.paragraphs[0]; p.paragraph_format.line_spacing = 1.0
    if align == "c":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if indent:
        p.paragraph_format.left_indent = Cm(0.3)
    add_runs(p, text, size=size, bold=bold)
    no_lines(p)


def lancet_num(s):
    """3,770 -> 3770 ; 23,974 -> 23 974 (thin space)"""
    def rep(m):
        n = int(m.group(0).replace(",", ""))
        return f"{n:,}".replace(",", "\u2009") if n >= 10000 else str(n)
    return re.sub(r"\d{1,3}(?:,\d{3})+", rep, str(s))


def add_table(header_rows, rows, widths, title, note, bold_rows=(), fs=8.5):
    para(title, size=11, bold=True, after=4, spacing=1.0)
    ncol = len(widths)
    t = doc.add_table(rows=len(header_rows) + len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), "57"); e.set(qn("w:type"), "dxa"); mar.append(e)
    tblPr.append(mar)
    for i, hr in enumerate(header_rows):
        j = 0
        for txt, span in hr:
            c = t.cell(i, j)
            if span > 1:
                c = c.merge(t.cell(i, j + span - 1))
            cell_text(c, txt, bold=True, align="c" if j > 0 else None, size=fs)
            if i == 0:
                cell_border(c, top="single")
            if i == len(header_rows) - 1:
                cell_border(c, bottom="single")
            elif span > 1:
                cell_border(c, bottom="single")
            j += span
    for i, row in enumerate(rows):
        ri = len(header_rows) + i
        for j, txt in enumerate(row):
            cell_text(t.cell(ri, j), txt, bold=(i in bold_rows), align="c" if j > 0 else None,
                      indent=(j == 0 and i not in bold_rows), size=fs)
        if i == len(rows) - 1:
            for j in range(ncol):
                cell_border(t.cell(ri, j), bottom="single")
    set_widths(t, widths)
    p = para(note, size=8.5, before=4, spacing=1.0)
    return t


# ----------------------------------------------------------------------------- title page
title = B["TITLE"][0]
para(title, size=15, bold=True, after=12, spacing=1.15)
para("[AUTHOR ACTION: Author names, each with one preferred academic degree, e.g. Firstname Lastname MD]", after=6, spacing=1.15)
para("[AUTHOR ACTION: Affiliations — department, institution, city, country, numbered to match author names]", after=6,
     spacing=1.15)
para("Correspondence to: [AUTHOR ACTION: name, full postal address, email, and telephone]", after=12, spacing=1.15)

sum_tags = [t for t, _ in blocks if t.startswith("SUMMARY_")]
ric_tags = [t for t, _ in blocks if t.startswith("RIC_")]
main_tags = []
started = False
for t, _ in blocks:
    if t == "H1 Introduction":
        started = True
    if t.startswith("H1 Contributors"):
        break
    if started:
        main_tags.append(t)
wc_sum = sum(words(p) for t in sum_tags for p in B[t]) + len(sum_tags)
wc_main = sum(words(p) for t in main_tags for p in B[t] if not t.startswith("H2 Role"))
wc_ric = sum(words(p) for t in ric_tags for p in B[t])
para(f"Word count: Summary {wc_sum} words; main text {wc_main} words (Introduction, Methods, Results, and Discussion).",
     after=0, spacing=1.15)
para(f"Research in context panel {wc_ric} words. References {len(set(order))}. Tables 2. Figures 4. "
     "Supplementary appendix 1 (PDF).", after=0, spacing=1.15)
page_break()

# ----------------------------------------------------------------------------- summary
heading("Summary")
for t in sum_tags:
    lab = t.split("_", 1)[1]
    p = para("", after=6)
    r = p.add_run(lab); r.bold = True
    p.add_run(" ")
    add_runs(p, " ".join(B[t]))
page_break()

# ----------------------------------------------------------------------------- research in context panel
para("", spacing=1.0)
box = doc.add_table(rows=1, cols=1); box.alignment = WD_TABLE_ALIGNMENT.CENTER; set_widths(box, [16.0])
c = box.cell(0, 0); cell_border(c, top="single", bottom="single", left="single", right="single")
c.width = Cm(16)
c.text = ""
p = c.paragraphs[0]; r = p.add_run("Research in context"); r.bold = True; r.font.size = Pt(12)
p.paragraph_format.space_after = Pt(4); p.paragraph_format.line_spacing = 1.15
for t in ric_tags:
    q = c.add_paragraph(); q.paragraph_format.space_before = Pt(6); q.paragraph_format.line_spacing = 1.15
    rr = q.add_run(t.split("_", 1)[1]); rr.bold = True; rr.font.size = Pt(10.5)
    for ptxt in B[t]:
        q = c.add_paragraph(); q.paragraph_format.line_spacing = 1.15; q.paragraph_format.space_after = Pt(2)
        add_runs(q, ptxt, size=10.5)
page_break()

# ----------------------------------------------------------------------------- main text
md_out = [f"# {title}", ""]
md_out += ["## Summary"] + [f"**{t.split('_', 1)[1]}** " + " ".join(B[t]) for t in sum_tags] + [""]
md_out += ["## Research in context"] + [f"**{t.split('_', 1)[1]}** " + " ".join(B[t]) for t in ric_tags] + [""]
for t, _ in blocks:
    if t.startswith(("TITLE", "SUMMARY", "RIC", "TABLE", "FIG")):
        continue
    lvl = 1 if t.startswith("H1 ") else 2
    heading(t[3:], lvl)
    md_out.append(("## " if lvl == 1 else "### ") + t[3:])
    for ptxt in B[t]:
        p = para(ptxt)
        p.paragraph_format.first_line_indent = None   # eCM: no tabs or indents at paragraph start
        md_out.append(re.sub(r"«SUP:([^»]*)»", r"<sup>\1</sup>", ptxt))
    md_out.append("")

# ----------------------------------------------------------------------------- references
heading("References")
md_out.append("## References")
ref_rows = []
for i, k in enumerate(order, 1):
    a, j, tail = ref_text(k)
    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.space_after = Pt(2)
    p.add_run(f"{i}\t{a}"); p.add_run(j); p.add_run(tail)
    md_out.append(f"{i}. {a}{j}{tail}")
    ref_rows.append([i, k, a + j + tail, REFS[k]["doi"], REFS[k]["year"]])
with open(os.path.join(OUT, "reference_list.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["number", "key", "reference", "doi", "year"]); w.writerows(ref_rows)

# ----------------------------------------------------------------------------- tables
page_break()
t1 = list(csv.DictReader(open(os.path.join(RES, "table1.csv"))))
rows = [[lancet_num(r["characteristic"]), lancet_num(r["derivation"]), lancet_num(r["independent"])] for r in t1]
add_table([[("", 1), ("Derivation sample (All-10 model)", 1), ("Independent sample (MRI-only model)", 1)]], rows,
          [7.4, 4.6, 4.6], B["TABLE1_TITLE"][0], " ".join(B["TABLE1_NOTE"]))
md_out += ["", "## " + B["TABLE1_TITLE"][0], "| Characteristic | Derivation | Independent |", "|---|---|---|"]
md_out += [f"| {a} | {b} | {c} |" for a, b, c in rows] + ["", " ".join(B["TABLE1_NOTE"]), ""]

page_break()
t2 = list(csv.DictReader(open(os.path.join(RES, "table2.csv"))))
rows, bold = [], []
for r in t2:
    if r["header"] == "True":
        bold.append(len(rows)); rows.append([r["outcome"], "", "", "", "", "", ""])
    else:
        rows.append([r["outcome"]] + [lancet_num(r[k]).replace("nan", "NA") if r[k] else "NA"
                                      for k in ["d_events", "d_hr", "d_g3", "i_events", "i_hr", "i_g3"]])
hdr = [[("", 1), ("Derivation sample (All-10 model)", 3), ("Independent sample (MRI-only model)", 3)],
       [("Outcome", 1), ("Events/at risk", 1), ("HR per 10 pp (95% CI)", 1), ("Stage ≥3 vs 0, HR (95% CI)", 1),
        ("Events/at risk", 1), ("HR per 10 pp (95% CI)", 1), ("Stage ≥3 vs 0, HR (95% CI)", 1)]]
add_table(hdr, rows, [3.3, 1.65, 2.35, 2.35, 1.65, 2.35, 2.35], B["TABLE2_TITLE"][0], " ".join(B["TABLE2_NOTE"]),
          bold_rows=bold, fs=7.5)
md_out += ["", "## " + B["TABLE2_TITLE"][0],
           "| Outcome | Der events/at risk | Der HR/10 pp | Der ≥3 vs 0 | Ind events/at risk | Ind HR/10 pp | Ind ≥3 vs 0 |",
           "|---|---|---|---|---|---|---|"]
md_out += ["| " + " | ".join(r) + " |" for r in rows] + ["", " ".join(B["TABLE2_NOTE"]), ""]

# ----------------------------------------------------------------------------- figure legends + figures
for n in (1, 2, 3, 4, 5):
    page_break()
    lines = B[f"FIG{n}"]
    para(lines[0], size=10, bold=True, after=4, spacing=1.0)   # eCM: figure heading 10 pt Times New Roman bold
    for l in lines[1:]:
        para(l, size=10, after=4, spacing=1.0)                  # eCM: legends 10 pt, single spaced
    png = os.path.join(FIG, f"Figure{n}", f"Figure{n}_composite.png")
    w, h = Image.open(png).size
    width = 16.0 if h / w < 1.25 else 16.0 * 1.25 * w / h
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.line_spacing = 1.0
    p.add_run().add_picture(png, width=Cm(width)); no_lines(p)
    md_out += ["", f"## {lines[0]}"] + lines[1:] + [f"![Figure {n}]({png})", ""]

doc.save(os.path.join(OUT, "eClinicalMedicine_manuscript.docx"))
open(os.path.join(OUT, "eClinicalMedicine_manuscript.md"), "w").write("\n".join(md_out))

# ----------------------------------------------------------------------------- word counts
sec_wc = {}
cur = None
for t in main_tags:
    if t.startswith("H1 "):
        cur = t[3:]; sec_wc[cur] = 0
    sec_wc[cur] += sum(words(p) for p in B[t]) if not t.startswith("H2 Role") else 0
with open(os.path.join(OUT, "word_counts.txt"), "w") as f:
    f.write(f"Title words: {len(title.split())}\nSummary: {wc_sum}\n")
    for t in sum_tags:
        f.write(f"  {t}: {sum(words(p) for p in B[t])}\n")
    f.write(f"Research in context: {wc_ric}\nMain text (Intro+Methods+Results+Discussion): {wc_main}\n")
    for k, v in sec_wc.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"References: {len(order)}\n")
print(open(os.path.join(OUT, "word_counts.txt")).read())
