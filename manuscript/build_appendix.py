import os as _os
HB_ROOT = _os.environ.get("HB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
UKB_DATA = _os.environ.get("UKB_DATA", _os.path.join(HB_ROOT, "data_ukb"))
"""Build the supplementary appendix (DOCX -> PDF) for the eClinicalMedicine submission.
Two passes: pass 1 renders the PDF and locates the page of every heading; pass 2 writes the page numbers into the contents
list and into appendix_pages.json (used by facts.py for the 'appendix p X' citations in the main text)."""
import json, re, subprocess, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Mm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.path.insert(0, _os.path.join(HB_ROOT, "analysis"))
import common as C

R = C.RES; V = R / "validation"; RS = R / "replication_sustain"
OUT = Path(_os.path.join(HB_ROOT, "manuscript/output")); OUT.mkdir(parents=True, exist_ok=True)
FIG_FLOW = _os.path.join(HB_ROOT, "figs/output_v5/Appendix/FigureS1_participant_flow_and_time_zero.png")
FIG_S1 = _os.path.join(HB_ROOT, "figs/output_v5/Appendix/FigureS2_PVD_sensitivity_and_denovo.png")
FIG_S3 = _os.path.join(HB_ROOT, "figs/output_v5/Appendix/FigureS3_regional_coupling_by_cardiac_phenotype.png")
FIG_S4 = _os.path.join(HB_ROOT, "figs/output_v7/Appendix/FigureS4_regional_by_subtype_stage_independent.png")
TITLE = ("A heart–brain MRI stage and risk of cardiovascular and cerebrovascular events beyond SCORE2: "
         "a prospective analysis of UK Biobank")
PAGES = json.load(open(OUT / "appendix_pages.json")) if (OUT / "appendix_pages.json").exists() and "--final" in sys.argv else {}


# ------------------------------------------------------------------ number formatting (Lancet style)
def d(x, k=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{k}f}".replace("-", "\u2212").replace(".", "\u00b7")


def n(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    x = int(round(x)); return f"{x:,}".replace(",", " ") if x >= 10000 else str(x)


def ev(x):
    """Event counts: small-cell suppression (1-4 shown as <5), consistent with the main figures."""
    x = int(round(x)); return "<5" if 0 < x < 5 else n(x)


def ci(lo, hi, k=2):
    sep = " to " if (lo < 0 or hi < 0) else "\u2013"
    return f"{d(lo, k)}{sep}{d(hi, k)}"


def est(e, lo, hi, k=2):
    if any(pd.isna(v) for v in (e, lo, hi)):
        return "–"
    return f"{d(e, k)} ({ci(lo, hi, k)})"


def pv(p):
    if pd.isna(p):
        return "–"
    if p < 0.0001:
        return "<0\u00b70001"
    t = f"{p:.2g}"
    if "e" in t:
        t = f"{p:.6f}".rstrip("0")
    return t.replace(".", "\u00b7")


OUTLAB = dict(C.OUTCOME_LABEL)
OUTLAB.update(hbve="Heart–brain vascular events (primary)", mace="Stroke, MI, or heart failure", cerebrovascular="Stroke or TIA",
              cardiac="Cardiac events (IHD, HF, or AF)", ckd="Chronic kidney disease", parkinson="Parkinson's disease")
MODLAB = {"all10": "All-10", "mri_plus_m13": "MRI + M13", "mri_only_matched_all10": "MRI only"}

# ------------------------------------------------------------------ document helpers
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Mm(297), Mm(210)
sec.left_margin = sec.right_margin = Mm(20); sec.top_margin = sec.bottom_margin = Mm(20)
st = doc.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(9.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
st.paragraph_format.space_after = Pt(4); st.paragraph_format.line_spacing = 1.2


def footer_page_numbers(section):
    p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for tag, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
        if tag:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag); r._r.append(e)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; r._r.append(e)
    r.font.size = Pt(8.5)


footer_page_numbers(sec)


def para(text="", bold=False, size=None, italic=False, align=None, space_after=4, color=None, keep=False):
    p = doc.add_paragraph()
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    if keep:
        p.paragraph_format.keep_with_next = True
    # simple inline markup: *italic*
    parts = re.split(r"(\*[^*]+\*)", text)
    for part in parts:
        if not part:
            continue
        it = part.startswith("*") and part.endswith("*")
        r = p.add_run(part[1:-1] if it else part)
        r.bold = bold; r.italic = italic or it
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
    return p


ANCHORS = []   # (key, heading text) in order


def heading(key, text, level=1):
    size = {0: 15, 1: 12, 2: 10.5}[level]
    p = para(text, bold=True, size=size, space_after=6, keep=True)
    p.paragraph_format.space_before = Pt(10 if level else 0)
    ANCHORS.append((key, text))
    return p


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def set_cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders"); tcPr.append(b)
    for edge, val in kw.items():
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), val.get("val", "single")); e.set(qn("w:sz"), str(val.get("sz", 6)))
        e.set(qn("w:color"), val.get("color", "000000")); b.append(e)


def table(header, rows, widths_mm, note=None, size=7.6, group_rows=()):
    """Three-line table. group_rows: indices of rows rendered as bold, shaded group headers spanning all columns."""
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths_mm):
        gc.set(qn("w:w"), str(int(w / 25.4 * 1440)))
    for i, row in enumerate([header] + rows):
        cells = t.rows[i].cells
        if i - 1 in group_rows:
            m = cells[0].merge(cells[-1])
            for extra in m.paragraphs[1:]:
                extra._p.getparent().remove(extra._p)
            p = m.paragraphs[0]
            for rr in list(p.runs):
                rr._r.getparent().remove(rr._r)
            p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
            r = p.add_run(str(row[0])); r.bold = True; r.font.size = Pt(size)
            sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "F2F3F5"); m._tc.get_or_add_tcPr().append(sh)
            continue
        for j, v in enumerate(row):
            c = cells[j]; c.width = Mm(widths_mm[j]); c.text = ""
            p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
            if j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(v)); r.font.size = Pt(size); r.bold = (i == 0)
            pr = p._p.get_or_add_pPr(); rpr = OxmlElement("w:rPr"); sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(size * 2)))
            rpr.append(sz); pr.append(rpr)
    for c in t.rows[0].cells:
        set_cell_border(c, top={"sz": 8}, bottom={"sz": 6})
    for c in t.rows[-1].cells:
        set_cell_border(c, bottom={"sz": 8})
    if note:
        p = para(note, size=7.4, space_after=10); p.paragraph_format.space_before = Pt(3)
    else:
        para("", space_after=6)
    return t


# ------------------------------------------------------------------ data
cox = pd.read_csv(R / "out_cox_main.csv")
inc = pd.read_csv(R / "out_incidence_by_stage.csv")
sens = pd.read_csv(R / "out_sensitivity.csv")
s2 = pd.read_csv(R / "out_score2_increment.csv")
s2s = pd.read_csv(R / "out_score2_strata.csv")
cmpd = pd.read_csv(R / "out_compare_predictors.csv")
inv = pd.read_csv(R / "out_involvement.csv")
ph = pd.read_csv(R / "out_ph_test.csv")
sub = pd.read_csv(R / "out_subtype.csv")
pos = pd.read_csv(R / "sustain_event_position_summary.csv")
hf = pd.read_csv(R / "sustain_heart_first.csv")
agr = pd.read_csv(R / "sustain_stage_agreement.csv")
pvo = pd.read_csv(R / "sustain_prevalence_vs_order.csv")
cor = pd.read_csv(R / "sustain_stage_correlates.csv")
dnh = pd.read_csv(RS / "denovo_heart_first_K2.csv"); dns = pd.read_csv(RS / "denovo_similarity_K2.csv")
icor = pd.read_csv(V / "independent_stage_correlates_imaging.csv")
coab = pd.read_csv(V / "heart_brain_coabnormality.csv")
icox = pd.read_csv(V / "independent_landmark_cox_imaging.csv")
iinc = pd.read_csv(V / "independent_landmark_incidence_imaging.csv")
ipre = pd.read_csv(V / "independent_preimaging_disease_imaging.csv")
ilm = pd.read_csv(V / "independent_landmark_cox.csv")
is2 = pd.read_csv(V / "independent_score2_increment_imaging.csv")
is2s = pd.read_csv(V / "independent_score2_strata_imaging.csv")
idv = pd.read_csv(V / "independent_discrimination_vs_counts_imaging.csv")
iinv = pd.read_csv(V / "independent_involvement_landmark_imaging.csv")
iinvp = pd.read_csv(V / "independent_involvement_preimaging_imaging.csv")
cvic = pd.read_csv(_os.path.join(HB_ROOT, "data_aggregate", "panel_b_grouped_bar_values.csv"))
der = pd.read_pickle(R / "cohort_derivation.pkl")
ind = pd.read_pickle(V / "independent_cohort_full_imaging.pkl")
sxs = pd.read_csv(R / "sex_stratified.csv"); sxd = pd.read_csv(R / "sex_stage_distribution.csv")

# ================================================================== title page and contents
para("Supplementary appendix", bold=True, size=16, space_after=10)
para("This appendix formed part of the original submission. It accompanies:", size=9.5)
para(TITLE, bold=True, size=11, space_after=6)
para("[AUTHOR ACTION: author list]", size=9.5, color="9A3412", space_after=14)
para("Contents", bold=True, size=11.5, space_after=4)
TOC = [
    ("m1", "Supplementary methods 1. Data sources, samples, and follow-up"),
    ("m2", "Supplementary methods 2. Marker derivation and normative modelling"),
    ("m3", "Supplementary methods 3. Subtype and Stage Inference: specification, subtype numbering, and staging"),
    ("m4", "Supplementary methods 4. Application to the independent sample and de novo refit"),
    ("m5", "Supplementary methods 5. Outcomes and SCORE2"),
    ("m6", "Supplementary methods 6. Additional statistical details"),
    ("m7", "Supplementary methods 7. Regional analysis of cardiac and brain MRI"),
    ("m8", "Supplementary methods 8. Subtype-specific regional anatomy and out-of-model measures"),
    ("t1", "Table S1. Outcome definitions"),
    ("t2", "Table S2. Missing covariate data"),
    ("t3", "Table S3. Model selection by cross-validation information criterion"),
    ("t4", "Table S4. Event sequences of the All-10 subtypes"),
    ("t5", "Table S5. Posterior probability that cardiac abnormality precedes brain abnormality"),
    ("t6", "Table S6. Similarity of the de novo sequences with the derivation sequences"),
    ("t7", "Table S7. Agreement of stage across models and relation of event order to marker prevalence"),
    ("t8", "Table S8. Clinical correlates of stage"),
    ("t9", "Table S9. Cardiac–brain co-abnormality in the independent sample"),
    ("t10", "Table S10. Stage and incident outcomes in the derivation sample, all models and outcomes"),
    ("t11", "Table S11. Event rates by stage group in the derivation and independent samples"),
    ("t12", "Table S12. Stage and incident outcomes and prevalent disease in the independent sample"),
    ("t13", "Table S13. Sensitivity analyses in the derivation sample"),
    ("t14", "Table S14. Proportional hazards tests"),
    ("t15", "Table S15. Fixed-landmark analysis in the independent sample"),
    ("t16", "Table S16. Subtype comparisons"),
    ("t17", "Table S17. Incremental value of stage beyond SCORE2"),
    ("t18", "Table S18. Event rates by stage within SCORE2 risk categories"),
    ("t19", "Table S19. Stage compared with simpler summaries of marker abnormality"),
    ("t20", "Table S20. Organ involvement and outcomes"),
    ("t21", "Table S21. Stage and the primary outcome in women and men"),
    ("t22", "Table S22. Regional heart–brain coupling by tissue class"),
    ("t23", "Table S23. Stage-matched regional differences between subtypes by tissue class"),
    ("fS1", "Figure S1. Participant flow, study markers, and imaging-based time zero"),
    ("f1", "Figure S2. Positional variance diagrams of the sensitivity models and the de novo refit"),
    ("fS3", "Figure S3. Brain coupling decomposed by cardiac phenotype"),
    ("fS4", "Figure S4. Measured regional abnormality by subtype and stage in the independent sample"),
]
for key, txt in TOC:
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.tab_stops.add_tab_stop(Mm(170), alignment=2, leader=1)
    r = p.add_run(f"{txt}\t{PAGES.get(key, '')}"); r.font.size = Pt(8.8)
page_break()

# ================================================================== supplementary methods
heading("m1", "Supplementary methods 1. Data sources, samples, and follow-up")
para("UK Biobank recruited about 500 000 participants aged 40–69 years in 2006–10 at 22 assessment centres in England, Scotland, "
     "and Wales. From 2014 a subset attended an imaging visit (instance 2), at which brain and cardiac MRI were acquired on the "
     "same day. The date of each visit was taken from field 53; age at imaging was calculated from the imaging date and the year and "
     "month of birth (fields 34 and 52) or, equivalently, from age at recruitment plus the interval between visits. Plasma proteins "
     "were measured at recruitment in about 53 000 participants by the UK Biobank Pharma Proteomics Project (Olink Explore 3072).")
para(f"Of 502 411 participants, 95 031 had at least one study marker and 27 744 had all seven cardiac and brain MRI markers. The "
     f"derivation sample comprised the {n(len(der))} participants who also had the three protein module scores; it was used to fit "
     f"all three SuStaIn models. The remaining {n(len(ind))} participants with complete MRI markers formed the independent sample; "
     "none of them contributed to model fitting, to the normative reference models' selection of markers, or to the choice of "
     "thresholds.")
para("Hospital inpatient diagnoses (Hospital Episode Statistics for England, Scottish Morbidity Records and the Patient Episode "
     "Database for Wales) were obtained from fields 41270 (ICD-10 codes) and 41280 (dates of first diagnosis), and deaths from "
     "fields 40000 (date) and 40001 (underlying cause). The latest diagnosis date in the extract was Sept 30, 2021, which was used "
     "as the common administrative censoring date. Follow-up started at the imaging visit. For each outcome, participants with a "
     "qualifying diagnosis on or before the imaging date were classified as prevalent cases and excluded from the incident analysis "
     "of that outcome; they were retained for the prevalent-disease analyses in the independent sample. Follow-up ended at the first "
     "qualifying diagnosis, death, or Sept 30, 2021. Deaths whose underlying cause matched an outcome definition were counted as "
     "events on the date of death; other deaths were censored.")
para("An earlier version of this analysis measured follow-up from the recruitment visit. Because the imaging visit took place a "
     "median of about 9 years after recruitment, that approach counted events that occurred before the MRI markers were measured; "
     "all results reported here therefore use the imaging visit as time zero.")

heading("m2", "Supplementary methods 2. Marker derivation and normative modelling")
para("Cardiac markers were derived from 82 automatically extracted cardiac and aortic MRI phenotypes (left and right ventricular "
     "volumes, mass, ejection fraction, strain and aortic distensibility). Three domains were defined a priori: LV remodelling "
     "(volume and mass indices), LV myocardial dysfunction (ejection fraction and global strain), and aortic stiffness (ascending and "
     "descending aortic distensibility). Brain markers were derived from 587 imaging-derived phenotypes of the UK Biobank brain "
     "pipeline: hippocampal atrophy (hippocampal grey-matter volumes), cortical thinning (regional cortical thickness), WMH burden "
     "(log-transformed total white matter hyperintensity volume, adjusted for head size), and white-matter free water (isotropic "
     "volume fraction from the NODDI model across white-matter tracts). Multi-phenotype domains were summarised by the first "
     "principal component, with the sign aligned so that higher values indicated greater abnormality (for example, lower ejection "
     "fraction, lower distensibility, lower hippocampal volume or cortical thickness).")
para("The three plasma protein modules (M13, M20, and M5) were identified by weighted gene co-expression network analysis of the "
     "Olink proteome and selected in preceding analyses for their association with cardiac or brain MRI phenotypes; each was "
     "represented by its module eigengene.")
para("Each marker was converted to a z score with normative models fitted in reference participants without cardiovascular, "
     "cerebrovascular, or neurological disease at the time of measurement (proteomics 45 522; cardiac MRI 29 081; brain MRI 39 155). "
     "The models included age and sex terms; z scores were the standardised residuals, signed so that positive values indicate "
     "greater abnormality than expected for age and sex. Because SuStaIn is fitted to these residualised scores, the inferred "
     "sequence reflects abnormality beyond normal ageing rather than chronological age itself.")

heading("m3", "Supplementary methods 3. Subtype and Stage Inference: specification, subtype numbering, and staging")
para("We used the z-score implementation of SuStaIn (pySuStaIn 0.1). Each marker contributed events at z=1, 2, and 3 (aortic "
     "stiffness only at z=1 in the MRI-only model, because higher thresholds were reached by very few participants), giving 30 "
     "events in All-10, 24 in MRI + M13, and 18 in MRI only. Models with one to five subtypes were fitted with 25 start points and "
     "100 000 MCMC iterations (random seed 20260712), and compared with the ten-fold cross-validation information criterion (CVIC; "
     "table S3).")
para("pySuStaIn reports individual subtype assignments in order of decreasing mean MCMC subtype fraction, whereas the stored "
     "sequences retain the order in which subtypes were created during fitting. To avoid mislabelling, we re-derived every "
     "participant's subtype and stage probabilities from the stored MCMC samples, confirmed that they reproduced the exported "
     "probabilities, and matched each displayed subtype to its sequence on this basis. Subtypes are numbered by decreasing fraction "
     "(All-10: S1, LV dysfunction-led; S2, LV remodelling-led).")
para("Each participant was assigned the maximum-likelihood subtype and stage. Stage 0 indicates no event reached. Because the "
     "number of events differs between models, stage was expressed as a percentage of the maximum stage (30, 24 or 18) and "
     "analysed per 10 percentage points; it was also grouped as 0, 1, 2, and 3 or higher. The posterior expected stage (the "
     "probability-weighted mean over stages and subtypes) was used in a sensitivity analysis. In the MRI + M13 model, the subtype "
     "fractions remained at their initial values (0·5) during both expectation maximisation and MCMC, so they were not estimated; "
     "for this model we report the share of stage-positive participants assigned to each subtype (S1 54·4%, S2 45·6%). The event "
     "sequences and individual stages of this model, which are derived from the posterior, were used as a sensitivity analysis only.")
para("For each subtype we summarised, across MCMC samples, the position of the first cardiac event and of the first brain event "
     "and the probability that the former preceded the latter (table S5). We also classified each model stage by the organ systems "
     "with at least one event reached by that stage (none, heart only, brain only, heart and brain) and correlated each subtype's "
     "event order with marker prevalence at z≥1 (table S7).")

heading("m4", "Supplementary methods 4. Application to the independent sample and de novo refit")
para("The MRI-only model was applied to the independent sample without refitting. For each participant, subtype and stage "
     "probabilities were computed from 1000 posterior sequence samples of the derivation model, using the same z score thresholds, "
     "and the maximum-likelihood stage was used in the analyses. Observed cardiac–brain co-abnormality was summarised with ORs "
     "for each pair of cardiac and brain markers at z≥1 and z≥2, and by the number of abnormal cardiac markers (table S9).")
para("To test whether the sequence could be recovered without the derivation data, we refitted the MRI-only model de novo in a "
     "random subsample of 6000 independent participants (random seed 20260712), with one to three subtypes, ten start points, and "
     "20 000 MCMC iterations; a full-sample refit was not computationally feasible. The two-subtype solution was compared with the "
     "derivation models by Spearman correlation of posterior mean event positions over shared events (table S6; figure S2).")

heading("m5", "Supplementary methods 5. Outcomes and SCORE2")
para("Outcome definitions are listed in table S1. ICD-10 codes were matched by prefix. Cardiovascular death was defined by an "
     "underlying cause of death in chapter I (I00–I99). Depression, anxiety, Parkinson's disease, and chronic kidney disease were "
     "examined as non-vascular comparators.")
para("SCORE2 (age <70 years) and SCORE2-OP (age ≥70 years) 10-year risks of fatal and non-fatal cardiovascular disease were "
     "calculated with the published sex-specific equations and the recalibration for the low-risk region, which includes the UK "
     "(SCORE2 scales: men −0·5699 and 0·7476, women −0·7380 and 0·7019; SCORE2-OP: men −0·34 and 1·19, women −0·52 and 1·01). "
     "Inputs were age at imaging and systolic blood pressure, total and HDL cholesterol, current smoking, and diabetes measured at "
     "recruitment, because blood was not drawn at the imaging visit. The implementation was checked against the CRAN package "
     "RiskScorescvd. Risk categories followed the 2021 ESC age-specific thresholds: <50 years, low-to-moderate <2·5%, high 2·5% to "
     "<7·5%, very high ≥7·5%; 50–69 years, <5%, 5% to <10%, ≥10%; ≥70 years, <7·5%, 7·5% to <15%, ≥15%. SCORE2 was entered into "
     "Cox models as its complementary log–log transform.")

heading("m6", "Supplementary methods 6. Additional statistical details")
para("In the derivation sample, missing covariates were imputed once with chained equations (scikit-learn IterativeImputer, 20 "
     "iterations, random seed 20260712) using all covariates and the marker z scores; outcomes were not used in imputation. "
     "Complete-case analysis was a sensitivity analysis (table S13). In the independent sample, model 2 and the SCORE2 analyses "
     "were fitted in participants with complete covariates. Cox models were fitted with lifelines 0.30 with a negligible ridge "
     "penalty (10⁻⁴) for numerical stability. Proportional hazards were assessed with the rank-transformed Schoenfeld residual "
     "test (table S14).")
para("In the derivation sample, the C-index difference (ΔC) between Cox models with SCORE2 alone and with SCORE2 plus stage was "
     "bootstrapped 1000 times; in each resample both models were refitted and the optimism (the difference between the bootstrap "
     "and the original-sample ΔC of the refitted models) was averaged and subtracted. In the independent sample (about 20 000 "
     "participants and two parameters) optimism is negligible; 95% CIs were obtained from 1000 bootstrap resamples with coefficients "
     "fixed at their full-sample estimates. Continuous NRI (category-free, >0) and IDI were estimated at 3 years from predicted "
     "risks of the two Cox models. Net benefit in decision curve analysis was computed at 3 years from Kaplan–Meier estimates of "
     "risk among participants above each threshold probability (0·5–10%). Event rates are per 1000 person-years with exact Poisson "
     "95% CIs. Event counts of 1–4 are shown as <5.")
heading("m7", "Supplementary methods 7. Regional analysis of cardiac and brain MRI")
para("To locate the stage and the heart–brain coupling anatomically, we rebuilt region-level normative z scores for the "
     "imaging measures that make up the seven MRI domain scores: cortical thickness in the 68 Desikan–Killiany regions "
     "(FreeSurfer), isotropic volume fraction (free water) in 48 TBSS white-matter tracts and 12 probabilistic (ProbtrackX) "
     "tracts, left and right hippocampal volume (FIRST), and whole-brain white-matter hyperintensity volume (BIANCA; the "
     "WMH domain z score); and, for the heart, wall thickness and circumferential and radial strain in the 16 AHA segments and "
     "12 global left-ventricular and aortic measures. Normative models mirrored those of the domain scores: ordinary least "
     "squares on age and sex (plus intracranial volume for hippocampal volume) in the disease-free reference group, residuals "
     "divided by the reference residual SD, oriented so that higher values are more abnormal, and truncated at ±5.")
para("Analyses were done in the independent sample (n=23 974), which was not used for model fitting; the derivation sample "
     "is shown for comparison. We described regional mean z by stage group (0, 1–2, and ≥3). For coupling, each brain "
     "region was regressed on a cardiac composite (the mean of the three cardiac domain z scores, standardised) and, "
     "separately, on each cardiac domain score; each cardiac measure was regressed on a white-matter composite (mean of WMH "
     "and free-water z) and a grey-matter composite (mean of cortical-thinning and hippocampal-atrophy z). Model A adjusted "
     "for age and sex; model B also adjusted for systolic blood pressure, diabetes, hypertension, current smoking, and body-"
     "mass index (complete case). False discovery rate (Benjamini–Hochberg) was controlled within each sample, model, and "
     "exposure. Tissue-class summaries are medians of the regional β values, with 95% CIs from 1000 participant-level "
     "bootstrap resamples (seed 20261025); the white-matter minus cortex difference was tested within the same resamples. "
     "Cortical maps use the fsaverage5 surface and white-matter maps the JHU-ICBM atlas in MNI space (display only). These "
     "analyses are descriptive: the domain scores that define the stage are built from these regions (table S22; figure S3).")
heading("m8", "Supplementary methods 8. Subtype-specific regional anatomy and out-of-model measures", level=1)
para("To test whether the two subtypes differ in where abnormality is expressed, and whether any difference is visible "
     "beyond the measures used to fit the model, we extended the regional set to 313 measures (263 brain and 50 cardiac). "
     "In addition to the measures in supplementary methods 7, we added fractional anisotropy (FA) and mean diffusivity (MD) "
     "in the 48 TBSS and 12 probabilistic tracts and the volumes of six subcortical nuclei in each hemisphere (thalamus, "
     "caudate, putamen, pallidum, amygdala, and accumbens; FIRST), with normative models specified exactly as in "
     "supplementary methods 7 (intracranial volume added for volumes) and oriented so that higher z is more abnormal (lower "
     "FA and volume, higher MD). FA, MD, subcortical volumes, segmental wall thickness, and three free-water tracts, 151 "
     "measures in total, are not inputs to any SuStaIn domain score (out-of-model measures); the remaining 162 are.")
para("Participants were grouped into stage 0 (a shared reference, because the subtype label carries no information at "
     "stage 0) and, for each subtype, stages 1–2 and stage 3 or higher; we report the mean regional z in each group, "
     "with the median posterior probability of the assigned subtype. Subtype differences were estimated among "
     "participants at stage 1 or higher by linear regression of each regional z score on subtype, stage (continuous), "
     "age, and sex, with Benjamini–Hochberg control across the 313 measures within each sample. In the derivation sample, "
     "subtype and stage were those of the All-10 model; in the independent sample, those of the MRI-only model applied "
     "without refitting, whose first two subtypes start with LV myocardial dysfunction and LV remodelling, as do All-10 "
     "subtypes 1 and 2. The small third MRI-only subtype (n=281) has no All-10 counterpart and was excluded. Replication "
     "was assessed by the Spearman correlation between the derivation and independent difference maps and by the "
     "proportion of measures with the same sign. In sensitivity analyses, the comparison was restricted to participants "
     "with a posterior subtype probability of at least 0·70 or 0·60. Finally, heart–brain coupling (supplementary methods "
     "7, model A and model B) was re-estimated for the out-of-model brain measures, and the median regional β for FA, MD, "
     "and subcortical volumes was compared with that for cortical thickness within the same 1000 participant-level "
     "bootstrap resamples. Groups or contrasts based on fewer than 20 participants were not reported (table S23; "
     "figure S4).")
page_break()

# ================================================================== tables
heading("t1", "Table S1. Outcome definitions")
rows = []
for k in ["stroke", "ischaemic_stroke", "tia", "mi", "ihd", "af", "hf", "pad", "dementia", "depression", "anxiety", "parkinson", "ckd"]:
    rows.append([OUTLAB.get(k, k), ", ".join(c if len(c) == 3 else f"{c[:3]}.{c[3:]}" for c in C.ICD[k]), "Inpatient diagnosis or underlying cause of death"])
rows.append(["Cardiovascular death", "I00–I99", "Underlying cause of death (field 40001)"])
rows.append(["All-cause death", "Any", "Date of death (field 40000)"])
comp = {"hbve": "Stroke, TIA, ischaemic heart disease, heart failure, or atrial fibrillation", "mace": "Stroke, myocardial infarction, or heart failure",
        "cerebrovascular": "Stroke or TIA", "cardiac": "Ischaemic heart disease, heart failure, or atrial fibrillation",
        "mace_cvd": "Stroke, myocardial infarction, heart failure, or cardiovascular death"}
rows.append(["Composite outcomes", "", ""])
for k, v in comp.items():
    rows.append([OUTLAB[k], "Components as above", v])
table(["Outcome", "ICD-10 codes", "Source / definition"], rows, [52, 50, 68],
      note="Codes were matched by prefix in fields 41270/41280 (hospital inpatient records) and 40001 (underlying cause of death). "
           "The earliest qualifying date defined the event. Prevalent cases (qualifying record on or before the imaging date) were "
           "excluded from the incident analysis of that outcome. TIA=transient ischaemic attack. PAD=peripheral arterial disease.",
      group_rows=(len(rows) - 6,))

heading("t2", "Table S2. Missing covariate data")
mvars = [("Systolic blood pressure", "SBP"), ("Total cholesterol", "CHOL"), ("HDL cholesterol", "HDLc"), ("Diabetes", "Diabetes"),
         ("Hypertension", "Hypertension"), ("Current smoking", "current_smoker"), ("Body-mass index", "BMI"), ("HbA1c", "HbA1c"),
         ("Cystatin C", "CysC")]
rows = []
for lab, c in mvars:
    a = f"{n(der[c].isna().sum())} ({d(100 * der[c].isna().mean(), 1)}%)" if c in der else "–"
    b = f"{n(ind[c].isna().sum())} ({d(100 * ind[c].isna().mean(), 1)}%)" if c in ind else "Not used"
    rows.append([lab, a, b])
table(["Covariate", f"Derivation (n={n(len(der))})", f"Independent (n={n(len(ind))})"], rows, [62, 54, 54],
      note="Values are number (%) missing. In the derivation sample missing values were imputed by chained equations; in the "
           "independent sample model 2 and SCORE2 analyses were restricted to participants with complete data. HbA1c and cystatin C "
           "were used only in the further-adjusted sensitivity analysis of the derivation sample.")

heading("t3", "Table S3. Model selection by cross-validation information criterion")
rows = []
for m in ["All-10", "MRI+M13", "MRI matched"]:
    g = cvic[cvic.model == m].sort_values("n_subtypes")
    for r in g.itertuples():
        rows.append([{"All-10": "All-10", "MRI+M13": "MRI + M13", "MRI matched": "MRI only"}[m] if r.n_subtypes == 1 else "",
                     str(r.n_subtypes), d(r.CVIC, 1), d(r.delta_CVIC_from_best, 1), "Selected" if r.minimum_CVIC else ""])
table(["Model", "Subtypes", "CVIC", "ΔCVIC from minimum", ""], rows, [34, 22, 36, 44, 34],
      note="CVIC from ten-fold cross-validation in the derivation sample (n=3770); lower is better. For All-10, the three-subtype "
           "solution was nearly equivalent to the selected two-subtype solution (ΔCVIC 1·7).")

page_break()
heading("t4", "Table S4. Event sequences of the All-10 subtypes")
rows = []
for sb in [1, 2]:
    g = pos[(pos.model == "all10") & (pos.subtype == sb)].sort_values("ml_position")
    fr = g.fraction.iloc[0]
    rows.append([f"S{sb}, {'LV dysfunction-led' if sb == 1 else 'LV remodelling-led'} (fraction {d(100 * fr, 1)}%)", "", "", ""])
    for r in g.itertuples():
        rows.append([str(r.ml_position), f"{r.label}, z={r.z}", r.organ, f"{d(r.post_median, 0)} ({d(r.post_lo, 0)}–{d(r.post_hi, 0)})"])
gr = (0, 31)
table(["ML position", "Event", "System", "Posterior position, median (95% CrI)"], rows, [24, 64, 28, 54],
      note="ML=maximum likelihood. CrI=credible interval across MCMC samples. Fractions are posterior medians of the MCMC "
           "subtype fraction.", size=7.2, group_rows=gr)

heading("t5", "Table S5. Posterior probability that cardiac abnormality precedes brain abnormality")
rows = []
for r in hf.sort_values(["model", "subtype"], key=lambda s: s.map({"all10": 0, "mri_plus_m13": 1, "mri_only_matched_all10": 2}) if s.name == "model" else s).itertuples():
    rows.append([f"{MODLAB[r.model]}, S{r.subtype}", d(r.p_heart_before_brain, 3), d(r.first_heart_pos_median, 0), d(r.first_brain_pos_median, 0),
                 d(r.p_heart_before_protein, 3) if r.model != "mri_only_matched_all10" else "–",
                 d(r.p_aortic_before_wmh, 3)])
for r in dnh.itertuples():
    rows.append([f"De novo refit (n=6000), S{r.subtype}", d(r.p_heart_before_brain, 3), d(r.first_heart_pos_median, 0), d(r.first_brain_pos_median, 0), "–", "–"])
table(["Model and subtype", "P(first cardiac before first brain event)", "First cardiac event, median position",
       "First brain event, median position", "P(cardiac before protein)", "P(aortic stiffness z1 before WMH z1)"], rows,
      [44, 27, 25, 25, 23, 26],
      note="Probabilities are proportions of MCMC samples. Positions are medians across MCMC samples. In the MRI + M13 model the "
           "protein event is M13 only.", size=7.2)

heading("t6", "Table S6. Similarity of the de novo sequences with the derivation sequences")
rows = []
for r in dns.itertuples():
    rows.append([f"S{r.denovo_subtype}", f"{MODLAB[r.derivation_model]}, S{r.derivation_subtype}", str(r.n_shared_events), d(r.spearman, 2), pv(r.p)])
table(["De novo subtype", "Derivation model and subtype", "Shared events", "Spearman ρ", "p"], rows, [30, 50, 28, 30, 32],
      note="Spearman correlation of posterior mean event positions over events shared by both models (MRI events only). De novo "
           f"fractions: S1 {d(100 * dnh.fraction.iloc[0], 1)}%, S2 {d(100 * dnh.fraction.iloc[1], 1)}%.")

heading("t7", "Table S7. Agreement of stage across models and relation of event order to marker prevalence")
rows = [["A. Agreement of individual stage between models", "", ""]]
for r in agr.itertuples():
    rows.append([f"{MODLAB[r.m1]} vs {MODLAB[r.m2]}", d(r.spearman, 2), f"{d(100 * r.both_positive, 1)}%"])
rows.append(["B. Correlation of event order with marker prevalence at z≥1", "", ""])
for r in pvo.itertuples():
    rows.append([f"{MODLAB[r.model]}, S{r.subtype}", d(r.spearman_order_vs_prevalence, 2), pv(r.p)])
table(["Comparison", "Spearman ρ", "Concordance of stage >0 / p"], rows, [90, 36, 44],
      note="A: concordance is the proportion of participants classified as stage-positive (stage >0) by one model who were also "
           "stage-positive by the other. B: correlation between each event's ML position and the prevalence of the corresponding "
           "marker at z≥1 (first event per marker); a weak correlation indicates that the order is not simply a ranking of "
           "prevalence.", group_rows=(0, len(agr) + 1))

page_break()
heading("t8", "Table S8. Clinical correlates of stage")
lab_ind = {"Age": "Age at imaging (per SD)", "male": "Male sex", "SBP": "Systolic BP (per SD)", "current_smoker": "Current smoking",
           "Diabetes": "Diabetes", "Hypertension": "Hypertension", "BMI": "BMI (per SD)", "CHOL": "Total cholesterol (per SD)",
           "HDLc": "HDL-C (per SD)", "score2_baseline": "SCORE2 risk (per SD)"}
rows = [["A. Derivation sample (All-10)", "", ""]]
for r in cor[cor.model == "all10"].itertuples():
    rows.append([r.label, est(r.beta, r.lo, r.hi, 3), pv(r.p) + f" (n={n(r.n)})"])
rows.append(["B. Independent sample (MRI-only model applied)", "", ""])
for r in icor.itertuples():
    rows.append([lab_ind.get(r.variable, r.variable), est(r.beta, r.lo, r.hi, 3), pv(r.p) + f" (n={n(r.n)})"])
table(["Characteristic", "β (95% CI)", "p (n)"], rows, [70, 50, 50],
      note="β is the difference in standardised stage (SD units) per unit of the characteristic, from linear regression adjusted "
           "for age and sex (each characteristic modelled separately). Continuous characteristics are per SD.",
      group_rows=(0, int((cor.model == "all10").sum()) + 1))

heading("t9", "Table S9. Cardiac–brain co-abnormality in the independent sample")
c = coab[(coab["sample"] == "independent")]
heart_lab = {"LV_remodeling_Z": "LV remodelling", "LV_myocardial_dysfunction_Z": "LV dysfunction", "Aortic_stiffness_Z": "Aortic stiffness"}
brain_lab = {"Hippocampal_atrophy_Z": "Hippocampal atrophy", "Cortical_thinning_Z": "Cortical thinning", "WMH_burden_Z": "WMH burden",
             "FreeWater_ISOVF_Z": "Free water", "any_brain": "Any brain marker"}
rows = []
for z in [1, 2]:
    rows.append([f"Threshold z≥{z}", "", "", "", ""])
    for h in heart_lab:
        g = c[(c.heart == h) & (c.z == z)]
        cells = []
        for b in ["Hippocampal_atrophy_Z", "Cortical_thinning_Z", "WMH_burden_Z", "FreeWater_ISOVF_Z"]:
            q = g[g.brain == b]
            cells.append(est(q.OR.iloc[0], q.lo.iloc[0], q.hi.iloc[0]) if len(q) else "–")
        rows.append([heart_lab[h]] + cells)
gidx = tuple(i for i, r in enumerate(rows) if r[0].startswith("Threshold"))
anyt = []
for z in [1, 2]:
    q = c[(c.heart == "any_heart") & (c.brain == "any_brain") & (c.z == z)]
    if len(q):
        anyt.append(f"z≥{z}, {est(q.OR.iloc[0], q.lo.iloc[0], q.hi.iloc[0])}")
any_txt = ("Any cardiac marker vs any brain marker: " + "; ".join(anyt) + ". ") if anyt else ""
dose = c[c.heart.str.startswith("n_heart_abnormal")].sort_values("heart")
dn_txt = "; ".join(f"{h.split('=')[1]} abnormal cardiac markers, {d(100 * q, 1)}%" for h, q in zip(dose.heart, dose.p_brain_given_heart)) if "p_brain_given_heart" in dose else ""
table(["Cardiac marker", "Hippocampal atrophy", "Cortical thinning", "WMH burden", "Free water"], rows, [38, 33, 33, 33, 33],
      note=f"ORs (95% CI) for brain abnormality given cardiac abnormality at the same threshold, n={n(len(ind))}. Because z scores are "
           f"residualised for age and sex, co-abnormality is not explained by these factors. {any_txt}Proportion with any brain marker "
           f"abnormal (z≥1) by number of abnormal cardiac markers: {dn_txt}.", size=7.2, group_rows=gidx)

page_break()
heading("t10", "Table S10. Stage and incident outcomes in the derivation sample, all models and outcomes")
order = ["hbve", "mace", "mace_cvd", "cerebrovascular", "cardiac", "stroke", "tia", "mi", "ihd", "af", "hf", "cv_death", "death",
         "depression", "anxiety", "parkinson", "ckd"]
rows = []
for o in order:
    g = cox[(cox.outcome == o) & (cox.exposure == "per10pp")]
    def cell(m, a):
        q = g[(g.model == m) & (g.adj == a)]
        return est(q.HR.iloc[0], q.lo.iloc[0], q.hi.iloc[0]) if len(q) else "–"
    q0 = g[(g.model == "all10") & (g.adj == "M2")]
    rows.append([OUTLAB[o], f"{n(q0.n.iloc[0])} / {ev(q0.events.iloc[0])}", cell("all10", "M1"), cell("all10", "M2"),
                 cell("mri_plus_m13", "M2"), cell("mri_only_matched_all10", "M2")])
table(["Outcome", "n / events", "All-10, model 1", "All-10, model 2", "MRI + M13, model 2", "MRI only, model 2"], rows,
      [44, 22, 26, 26, 26, 26],
      note="HR (95% CI) per 10-percentage-point increase in stage. Model 1: age and sex. Model 2: additionally systolic blood "
           "pressure, current smoking, diabetes, total and HDL cholesterol, BMI, and hypertension. Follow-up from the imaging visit "
           "to Sept 30, 2021; prevalent cases excluded per outcome.", size=7.2)

heading("t11", "Table S11. Event rates by stage group in the derivation and independent samples")
rows = []
for o in ["hbve", "mace", "cerebrovascular", "cardiac", "stroke", "ihd", "af", "hf", "cv_death"]:
    r1 = []
    for g in ["0", "1", "2", ">=3"]:
        q = inc[(inc.outcome == o) & (inc.stage_grp == g)]
        r1.append(f"{d(q.rate.iloc[0], 1)} ({ev(q.events.iloc[0])}/{n(q.n.iloc[0])})" if len(q) else "–")
    r2 = []
    for g in ["0", "1", "2", ">=3"]:
        q = iinc[(iinc.outcome == o) & (iinc.grp.astype(str) == g)]
        r2.append(f"{d(q.rate.iloc[0], 1)} ({ev(q.events.iloc[0])}/{n(q.n.iloc[0])})" if len(q) else "–")
    rows.append([OUTLAB[o], *r1]); rows.append(["   independent", *r2])
table(["Outcome (derivation; independent)", "Stage 0", "Stage 1", "Stage 2", "Stage ≥3"], rows, [50, 30, 30, 30, 30],
      note="Rate per 1000 person-years (events/participants at risk). Derivation: All-10 model. Independent: MRI-only model applied "
           "without refitting. Follow-up from the imaging visit.", size=7.0)

heading("t12", "Table S12. Stage and incident outcomes and prevalent disease in the independent sample")
rows = []
for o in ["hbve", "mace", "mace_cvd", "cerebrovascular", "cardiac", "stroke", "ihd", "af", "hf", "cv_death", "depression", "parkinson", "ckd"]:
    g = icox[icox.outcome == o]
    def c2(a):
        q = g[g.adj == a]; return est(q.HR.iloc[0], q.lo.iloc[0], q.hi.iloc[0]) if len(q) else "–"
    q1 = g[g.adj == "M1"]; pr = ipre[ipre.outcome == o]
    rows.append([OUTLAB[o], f"{n(q1.n.iloc[0])} / {ev(q1.events.iloc[0])}" if len(q1) else "–", c2("M1"), c2("M2"), c2("M2_grp>=3vs0"),
                 (f"{est(pr.OR.iloc[0], pr.lo.iloc[0], pr.hi.iloc[0])}; {ev(pr.cases.iloc[0])} cases") if len(pr) else "–"])
table(["Outcome", "n / events (model 1)", "Incident HR, model 1", "Incident HR, model 2", "Stage ≥3 vs 0, model 2", "Prevalent disease, OR"],
      rows, [40, 24, 25, 25, 25, 31],
      note="Incident: HR (95% CI) per 10 percentage points of stage, follow-up from each participant's imaging visit. Model 2 in "
           "participants with complete covariates. Prevalent: OR per 10 percentage points for a diagnosis on or before the imaging "
           "date, adjusted for age and sex.", size=7.0)

page_break()
heading("t13", "Table S13. Sensitivity analyses in the derivation sample")
rows = []
for a in sens.analysis.unique():
    g = sens[sens.analysis == a]
    cells = []
    for o in ["hbve", "mace", "cerebrovascular", "cardiac"]:
        q = g[g.outcome == o]
        cells.append(f"{est(q.HR.iloc[0], q.lo.iloc[0], q.hi.iloc[0])}" if len(q) else "–")
    q = g[g.outcome == "hbve"]
    lab = a.replace("P>=0.60", "P≥0·60").replace("(imputed covariates, M2)", "(model 2, imputed covariates)")
    rows.append([lab, f"{n(q.n.iloc[0])} / {ev(q.events.iloc[0])}" if len(q) else "–", *cells])
table(["Analysis", "n / events (primary)", "Heart–brain vascular events", "Stroke, MI, or HF", "Stroke or TIA", "Cardiac events"], rows,
      [46, 22, 26, 26, 25, 25],
      note="HR (95% CI) per 10 percentage points of All-10 stage, model 2 unless stated. Posterior expected stage is the "
           "probability-weighted stage.", size=7.0)

heading("t14", "Table S14. Proportional hazards tests")
rows = [[OUTLAB[r.outcome], pv(r.p_stage), pv(r.p_global_min)] for r in ph.itertuples()]
table(["Outcome", "p, stage term", "Smallest p, any covariate"], rows, [80, 45, 45],
      note="Rank-transformed Schoenfeld residual test for the All-10 stage in model 2.")

heading("t15", "Table S15. Fixed-landmark analysis in the independent sample")
rows = []
for o in ["hbve", "mace", "cerebrovascular", "cardiac", "stroke", "ihd", "af", "hf", "cv_death"]:
    g = ilm[ilm.outcome == o]
    def c3(a):
        q = g[g.adj == a]; return est(q.HR.iloc[0], q.lo.iloc[0], q.hi.iloc[0]) if len(q) else "–"
    q1 = g[g.adj == "M1"]
    rows.append([OUTLAB[o], f"{n(q1.n.iloc[0])} / {ev(q1.events.iloc[0])}" if len(q1) else "–", c3("M1"), c3("M2"), c3("M2_grp>=3vs0")])
table(["Outcome", "n / events", "HR, model 1", "HR, model 2", "Stage ≥3 vs 0, model 2"], rows, [52, 28, 30, 30, 30],
      note="Time zero April 1, 2020 for all participants (imaging before this date), excluding participants with the outcome before "
           "the landmark; follow-up to Sept 30, 2021. HR per 10 percentage points of stage.", size=7.2)

heading("t16", "Table S16. Subtype comparisons")
rows = []
for r in sub[sub.model == "all10"].itertuples():
    rows.append([OUTLAB[r.outcome], f"{ev(r.ev_S1)} / {ev(r.ev_S2)}", est(r.HR_S2vsS1, r.lo_S2vsS1, r.hi_S2vsS1), pv(r.p_S2vsS1), pv(r.p_interaction)])
table(["Outcome", "Events, S1 / S2", "HR, S2 vs S1", "p", "p, stage × subtype"], rows, [52, 28, 36, 24, 30],
      note="Stage-positive participants of the All-10 model; model 2 adjusted additionally for stage. The interaction test compares "
           "the stage–outcome slope between subtypes.", size=7.2)

page_break()
heading("t17", "Table S17. Incremental value of stage beyond SCORE2")
rows = [["A. Derivation sample (All-10)", "", "", "", "", ""]]
for o in ["hbve", "mace", "cerebrovascular", "cardiac", "stroke", "ihd", "af", "hf", "cv_death"]:
    q = s2[(s2.outcome == o) & (s2.model == "all10")]
    if not len(q):
        continue
    q = q.iloc[0]
    rows.append([OUTLAB[o], f"{n(q.n)} / {ev(q.events)}", f"{d(q.C_score2, 3)} → {d(q.C_score2_stage, 3)}",
                 f"{d(q.dC, 3)} ({ci(q.dC_lo, q.dC_hi, 3)}); corrected {d(q.dC_corrected, 3)}", pv(q.LR_p), f"{d(q.NRI, 2)} / {d(q.IDI, 4)}"])
rows.append(["B. Independent sample (MRI-only model applied)", "", "", "", "", ""])
k0 = len(rows) - 1
for o in ["hbve", "mace", "cerebrovascular", "cardiac", "stroke", "ihd", "af", "hf", "cv_death"]:
    q = is2[is2.outcome == o]
    if not len(q):
        continue
    q = q.iloc[0]
    rows.append([OUTLAB[o], f"{n(q.n)} / {ev(q.events)}", f"{d(q.C_score2, 3)} → {d(q.C_score2_stage, 3)}",
                 f"{d(q.dC, 3)} ({ci(q.dC_lo, q.dC_hi, 3)})", pv(q.LR_p), "–"])
table(["Outcome", "n / events", "C-index, SCORE2 → + stage", "ΔC (95% CI)", "LR p", "NRI / IDI at 3 years"], rows,
      [40, 21, 26, 41, 17, 25],
      note="Cox models with complementary log–log SCORE2 risk, with and without stage. Derivation: bootstrap 95% CI and "
           "optimism-corrected ΔC (1000 resamples, models refitted). Independent: bootstrap 95% CI with coefficients fixed. "
           "LR=likelihood ratio test. NRI=continuous net reclassification improvement. IDI=integrated discrimination improvement.",
      size=6.9, group_rows=(0, k0))

heading("t18", "Table S18. Event rates by stage within SCORE2 risk categories")
rows = []
for lab_s, df in [("Derivation", s2s), ("Independent", is2s)]:
    rows.append([f"{lab_s} sample", "", "", ""])
    for o in ["hbve", "mace", "stroke"]:
        for sg in ["Low-to-moderate", "High or very high"]:
            cells = []
            for gg in ["Stage 0", "Stage 1-2", "Stage >=3"]:
                q = df[(df.outcome == o) & (df.score2_group == sg) & (df.stage_group == gg)]
                cells.append(f"{d(q.rate.iloc[0], 1)} ({ev(q.events.iloc[0])}/{n(q.n.iloc[0])})" if len(q) else "–")
            rows.append([f"{OUTLAB[o].replace(' (primary)', '')}, {sg.lower()}", *cells])
gi = tuple(i for i, r in enumerate(rows) if r[0].endswith("sample"))
table(["Outcome and SCORE2 category", "Stage 0", "Stages 1–2", "Stage ≥3"], rows, [70, 33, 33, 34],
      note="Rate per 1000 person-years (events/participants). SCORE2 categories by 2021 ESC age-specific thresholds; low-to-moderate "
           "vs high or very high.", size=7.0, group_rows=gi)

heading("t19", "Table S19. Stage compared with simpler summaries of marker abnormality")
rows = [["A. Derivation sample: SCORE2 plus each summary (primary outcome)", "", "", ""]]
lab_p = {"all10_sd": "Stage, All-10", "mri_plus_m13_sd": "Stage, MRI + M13", "mri_only_matched_all10_sd": "Stage, MRI only",
         "n_abnormal_z1_sd": "Number of abnormal markers (z≥1)", "mean_abn_z_sd": "Mean marker z score",
         "all10_sd|count": "Stage, beyond marker count", "all10_sd|meanz": "Stage, beyond mean z score", "meanz|all10_sd": "Mean z score, beyond stage"}
for r in cmpd[cmpd.outcome == "hbve"].itertuples():
    rows.append([lab_p.get(r.predictor, r.predictor), est(r.HR, r.lo, r.hi), d(r.C_new, 3) if not pd.isna(r.C_new) else "–", pv(r.LR_p)])
rows.append(["B. Independent sample: clinical risk factors plus each summary (C-index)", "", "", ""])
k1 = len(rows) - 1
for r in idv.itertuples():
    rows.append([OUTLAB[r.outcome].replace(" (primary)", ""), f"risk factors {d(r.C_rf, 3)}",
                 f"stage {d(r.C_rf_p10, 3)}; count {d(r.C_rf_n_abn, 3)}", f"mean z {d(r.C_rf_mean_z, 3)}"])
table(["Predictor or outcome", "HR per SD (95% CI) / base", "C-index", "LR p / mean z"], rows, [62, 36, 40, 32],
      note="A: HR per SD of each summary added to SCORE2; nested rows test whether one summary adds to another. B: C-index of "
           "Cox models with model 2 risk factors alone or plus stage, abnormal marker count, or mean z score.", size=7.0,
      group_rows=(0, k1))

page_break()
heading("t20", "Table S20. Organ involvement and outcomes")
lvl = ["Heart only", "Brain only", "Heart and brain"]
rows = [["A. Derivation: model-implied involvement at each participant's stage (All-10)", "", "", ""]]
for o in ["hbve", "mace", "cerebrovascular", "cardiac"]:
    g = inv[(inv.scheme == "model_all10") & (inv.outcome == o)]
    cells = []
    for l in lvl:
        q = g[g.level == l]; cells.append(f"{est(q.HR.iloc[0], q.lo.iloc[0], q.hi.iloc[0])}; {ev(q.events.iloc[0])}/{n(q.n.iloc[0])}" if len(q) else "–")
    rows.append([OUTLAB[o].replace(" (primary)", ""), *cells])
rows.append(["B. Independent: observed abnormality (any marker z≥1), incident events", "", "", ""])
k1 = len(rows) - 1
for o in ["hbve", "cerebrovascular", "cardiac", "stroke", "ihd", "af", "hf"]:
    g = iinv[(iinv.scheme == "obs_cat") & (iinv.outcome == o)]
    cells = []
    for l in lvl:
        q = g[g.level == l]; cells.append(f"{est(q.HR.iloc[0], q.lo.iloc[0], q.hi.iloc[0])}; {ev(q.events.iloc[0])}/{n(q.n.iloc[0])}" if len(q) else "–")
    rows.append([OUTLAB[o].replace(" (primary)", ""), *cells])
rows.append(["C. Independent: observed abnormality, disease on or before imaging", "", "", ""])
k2 = len(rows) - 1
for o in ["hbve", "cerebrovascular", "cardiac", "stroke", "ihd", "af", "hf"]:
    g = iinvp[(iinvp.scheme == "obs_cat") & (iinvp.outcome == o)]
    cells = []
    for l in lvl:
        q = g[g.level == l]; cells.append(f"{est(q.OR.iloc[0], q.lo.iloc[0], q.hi.iloc[0])}; {ev(q.cases.iloc[0])}/{n(q.n.iloc[0])}" if len(q) else "–")
    rows.append([OUTLAB[o].replace(" (primary)", ""), *cells])
table(["Outcome", "Heart only", "Brain only", "Heart and brain"], rows, [41, 43, 43, 43],
      note="Reference: no organ involved (stage 0 or no marker z≥1). A and B: HR (95% CI), model 2; events/participants. C: OR (95% CI) "
           "adjusted for age and sex; cases/participants.", size=6.9, group_rows=(0, k1, k2))
heading("t21", "Table S21. Stage and the primary outcome in women and men")
rows, grp_rows = [], []
for smp, lab in [("derivation", "A. Derivation sample (All-10 model)"), ("independent", "B. Independent sample (MRI-only model)")]:
    g = sxs[sxs["sample"] == smp].set_index("sex"); q = sxd[sxd["sample"] == smp].set_index("sex")
    grp_rows.append(len(rows))
    rows.append([f"{lab}; p for interaction {pv(g.loc['Interaction (male vs female)', 'p'])}", "", "", "", "", "", ""])
    for sx, sl in [("Female", "Women"), ("Male", "Men")]:
        rows.append([sl, n(q.loc[sx, "n"]), f"{d(q.loc[sx, 'stage0_pct'], 1)}%", f"{d(q.loc[sx, 'stage3plus_pct'], 1)}%",
                     f"{n(g.loc[sx, 'n'])} / {ev(g.loc[sx, 'events'])}", d(g.loc[sx, "rate"], 1),
                     est(g.loc[sx, "HR"], g.loc[sx, "lo"], g.loc[sx, "hi"])])
table(["Sex", "Participants", "Stage 0", "Stage ≥3", "At risk / events", "Rate", "HR (95% CI), model 2"], rows,
      [27, 22, 19, 19, 30, 18, 35],
      note="Sex from UK Biobank records (field 31, recorded at recruitment and updated by self-report); gender was not assessed. "
           "Markers were normalised for age and sex before staging, and sequences were estimated in women and men together. "
           "Primary outcome: stroke, transient ischaemic attack, ischaemic heart disease, heart failure, or atrial fibrillation after "
           "the imaging visit. Rates per 1000 person-years. HR per 10 percentage points of stage from model 2 without the sex term, "
           "fitted separately in women and men; p for interaction from model 2 with a stage-by-sex product term. In the independent "
           "sample, model 2 is complete-case.", size=7.2, group_rows=tuple(grp_rows))

heading("t22", "Table S22. Regional heart–brain coupling by tissue class")
_RG = R / "regional"
_cp = pd.read_csv(_RG / "regional_coupling.csv")
_ts = pd.read_csv(_RG / "regional_tissue_summary.csv")
def _cell(smp, model, ex, sel):
    g = _cp[(_cp["sample"] == smp) & (_cp.model == model) & (_cp.exposure == ex)]
    g = g[sel(g)]
    return f"{d(g.beta.median(), 3)}; {(g.q < 0.05).sum()}/{len(g)}"
_brain = [("Cortical thickness", lambda g: g.tissue == "Cortical thickness"),
          ("Hippocampal volume", lambda g: g.tissue == "Hippocampal volume"),
          ("White-matter free water", lambda g: g.tissue == "White-matter free water"),
          ("White-matter hyperintensity", lambda g: g["var"] == "WMH_burden_Z_raw")]
rows = [["A. Brain regions; exposure, cardiac burden", "", "", "", "", "", ""]]
for lab, sel in _brain:
    rows.append([lab, _cell("independent", "A", "cardiac_comp", sel), _cell("independent", "B", "cardiac_comp", sel),
                 _cell("independent", "A", "LV_remodeling_Z_raw", sel),
                 _cell("independent", "A", "LV_myocardial_dysfunction_Z_raw", sel),
                 _cell("independent", "A", "Aortic_stiffness_Z_raw", sel), _cell("derivation", "A", "cardiac_comp", sel)])
table(["Tissue class", "Cardiac composite, A", "Cardiac composite, B", "LV remodelling, A", "LV dysfunction, A",
       "Aortic stiffness, A", "Derivation: composite, A"], rows, [33, 23, 23, 23, 23, 23, 24], size=6.7, group_rows=(0,))
_heart = [(t, (lambda t: (lambda g: g.tissue == t))(t)) for t in
          ("Wall thickness", "Circumferential strain", "Radial strain", "Global CMR")]
rows = [["B. Cardiac measures; exposure, brain white-matter or grey-matter burden", "", "", "", "", "", ""]]
for lab, sel in _heart:
    rows.append([lab if lab != "Global CMR" else "Global LV and aortic", _cell("independent", "A", "brain_wm_comp", sel),
                 _cell("independent", "A", "brain_gm_comp", sel), _cell("independent", "B", "brain_wm_comp", sel),
                 _cell("independent", "B", "brain_gm_comp", sel), _cell("derivation", "A", "brain_wm_comp", sel),
                 _cell("derivation", "A", "brain_gm_comp", sel)])
table(["Measure class", "White matter, A", "Grey matter, A", "White matter, B", "Grey matter, B",
       "Derivation: white matter, A", "Derivation: grey matter, A"], rows, [33, 23, 23, 23, 23, 23, 24], size=6.7,
      group_rows=(0,))
rows = [["C. Tissue-class medians with bootstrap 95% CI; exposure, cardiac burden", "", "", "", ""]]
for ct in ["Cortical thickness", "Hippocampal volume", "White-matter free water", "WM free water minus cortical thickness"]:
    cells = []
    for smp, model in [("independent", "A"), ("independent", "B"), ("derivation", "A"), ("derivation", "B")]:
        r = _ts[(_ts["sample"] == smp) & (_ts.model == model) & (_ts.contrast == ct)].iloc[0]
        cells.append(f"{d(r.median_beta, 3)} ({ci(r.lo, r.hi, 3)})")
    rows.append([ct.replace("WM free water minus cortical thickness", "White matter minus cortex"), *cells])
table(["Contrast", "Independent, A", "Independent, B", "Derivation, A", "Derivation, B"], rows, [44, 32, 32, 32, 32],
      note="Cells in A and B: median regional β (z per 1 SD exposure); regions with FDR q<0·05 / regions. Brain "
           "regions: 68 cortical, 2 hippocampal, 60 white-matter tracts (48 TBSS, 12 ProbtrackX), and whole-brain WMH. "
           "Cardiac measures: 16 AHA segments each for wall thickness and circumferential and radial strain, and 12 global "
           "measures. Higher z denotes more abnormal for every measure. Model A: age and sex; model B: plus systolic blood "
           "pressure, diabetes, hypertension, current smoking, and body-mass index. Bootstrap: 1000 participant-level resamples.",
      size=6.9, group_rows=(0,))
heading("t23", "Table S23. Stage-matched regional differences between subtypes by tissue class")
_RS = R / "regional_subtype"
_rd = pd.read_csv(_RS / "dictionary_ext.csv").set_index("var")
_rc = pd.read_csv(_RS / "subtype_contrast.csv"); _rc["tissue"] = _rc["var"].map(_rd.tissue)
_rr = pd.read_csv(_RS / "contrast_replication.csv").set_index("set")
_rt = pd.read_csv(_RS / "tissue_summary_ext.csv")
_ORDER = [("Wall thickness", True), ("Circumferential strain", False), ("Radial strain", False),
          ("Aortic distensibility", False), ("Cortical thickness", False), ("Hippocampal volume", False),
          ("Subcortical volume", True), ("White-matter FA", True), ("White-matter MD", True),
          ("White-matter free water", False), ("White-matter hyperintensity", False)]
def _c23(smp, an, t):
    g = _rc[(_rc["sample"] == smp) & (_rc.analysis == an) & (_rc.tissue == t)]
    return f"{d(g.beta.median())}; {(g.q < 0.05).sum()}/{len(g)}"
def _n23(smp, an):
    g = _rc[(_rc["sample"] == smp) & (_rc.analysis == an)]
    return f"{int(g.n_s1.max())}/{int(g.n_s2.max())}"
rows = []
for t, oom in _ORDER:
    g = _rc[(_rc["sample"] == "independent") & (_rc.analysis == "primary") & (_rc.tissue == t)]
    rho = d(_rr.loc[t, "spearman_rho"]) if t in _rr.index else "–"
    lab = t.replace("White-matter", "WM") + ("†" if oom else "")
    if t == "White-matter free water":
        lab += " (3 of 60†)"
    rows.append([lab, str(len(g)), _c23("derivation", "primary", t), _c23("independent", "primary", t), rho,
                 _c23("derivation", "p_subtype>=0.70", t), _c23("independent", "p_subtype>=0.70", t)])
rows.append(["Participants, subtype 1/subtype 2", "", _n23("derivation", "primary"), _n23("independent", "primary"), "",
             _n23("derivation", "p_subtype>=0.70"), _n23("independent", "p_subtype>=0.70")])
table(["Tissue class", "Measures", "Derivation", "Independent", "ρ between samples", "Derivation, p≥0·70",
       "Independent, p≥0·70"], rows, [36, 15, 24, 24, 20, 24, 24], size=6.7)
_rep = (f"Replication of the difference map: Spearman ρ {d(_rr.loc['all regions', 'spearman_rho'])} across "
        f"{int(_rr.loc['all regions', 'n_regions'])} measures, {d(_rr.loc['out-of-model', 'spearman_rho'])} across the "
        f"{int(_rr.loc['out-of-model', 'n_regions'])} out-of-model and {d(_rr.loc['in-model', 'spearman_rho'])} across the "
        f"{int(_rr.loc['in-model', 'n_regions'])} in-model measures; same sign in "
        f"{round(100 * _rr.loc['all regions', 'sign_concordance'])}%.")
rows = [["Coupling with the cardiac composite", "", "", ""]]
for ct, lab in (("White-matter FA minus cortical thickness", "WM FA† minus cortex"),
                ("White-matter MD minus cortical thickness", "WM MD† minus cortex"),
                ("Subcortical volume minus cortical thickness", "Subcortical volume† minus cortex")):
    cells = []
    for smp, model in (("independent", "A"), ("independent", "B"), ("derivation", "A")):
        r = _rt[(_rt["sample"] == smp) & (_rt.model == model) & (_rt.contrast == ct)].iloc[0]
        cells.append(f"{d(r.median_beta, 3)} ({ci(r.lo, r.hi, 3)}); p={'<0·001' if r.p_boot < 0.001 else d(r.p_boot, 3)}")
    rows.append([lab, *cells])
table(["Contrast of median regional β", "Independent, A", "Independent, B", "Derivation, A"], rows, [52, 38, 38, 38],
      note="Upper table: median over the measures of each tissue class of β for subtype 1 minus subtype 2 (z units; "
           "positive, more abnormal in subtype 1), from linear models of each regional z score on subtype, stage, age, and "
           "sex among participants at stage 1 or higher; measures with Benjamini–Hochberg q<0·05 / measures. "
           "p≥0·70: participants with posterior subtype probability of at least 0·70. ρ: Spearman correlation of regional β "
           "between samples (tissue classes with at least 12 measures). " + _rep + " Lower table: difference in the median "
           "regional β (z per 1 SD cardiac composite) between each out-of-model tissue class and cortical thickness, with "
           "bootstrap 95% CI and p (1000 resamples); model A, age and sex; model B, plus vascular risk factors. "
           "†Out-of-model measure (not an input to any SuStaIn domain score). WM=white matter.",
      size=6.9, group_rows=(0,))
page_break()
heading("fS1", "Figure S1. Participant flow, study markers, and imaging-based time zero")
doc.add_picture(FIG_FLOW, width=Mm(165))
para("(A) Participant flow from the UK Biobank cohort to the derivation sample (cardiac MRI, brain MRI, and plasma "
     "proteomics; used for model fitting) and the independent sample (cardiac and brain MRI; not used for model "
     "fitting). (B) The ten markers: three cardiac MRI, four brain MRI, and three plasma-protein modules. (C) Time zero "
     "was the imaging visit (field 53, instance 2); people with the outcome on or before imaging were excluded, follow-up "
     "ended on 30 September 2021, deaths from the outcome were events, and other deaths were censored.", size=8)
page_break()
heading("f1", "Figure S2. Positional variance diagrams of the sensitivity models and the de novo refit")
doc.add_picture(FIG_S1, width=Mm(165))
para("(A, B) MRI + M13 model and (C–E) MRI-only model in the derivation sample (n=3770); (F, G) MRI-only model refitted de novo, "
     "with two subtypes, in a random subsample of 6000 participants of the independent sample. Shading shows the posterior "
     "probability of each event (rows) at each position (columns) across MCMC samples; rows are ordered by maximum-likelihood "
     "position. In every subtype the first cardiac event preceded the first brain event (table S5). Percentages are model "
     "fractions, except for MRI + M13, for which fractions were not estimated and the share of stage-positive participants "
     "assigned to each subtype is shown.", size=8)
page_break()
heading("fS3", "Figure S3. Brain coupling decomposed by cardiac phenotype")
doc.add_picture(FIG_S3, width=Mm(165))
para("(A) Brain regions in the independent sample (n=23 974) coloured by β (z per 1 SD) for the cardiac composite "
     "(as in figure 3B) and for each cardiac domain score, adjusted for age and sex; faded regions, FDR q≥0·05. Right, "
     "median regional β per tissue class and the number of regions with q<0·05. (B) Tissue-class medians with "
     "bootstrap 95% CIs by adjustment model and sample. White-matter coupling was carried mainly by aortic stiffness and LV "
     "remodelling, with little contribution from LV myocardial dysfunction.", size=8)
page_break()
heading("fS4", "Figure S4. Measured regional abnormality by subtype and stage in the independent sample")
doc.add_picture(FIG_S4, width=Mm(165))
para("The matrix of figure 2B repeated in the independent sample (n=23 974), with subtype and stage from the MRI-only model "
     "applied without refitting. Rows: stage 0 (shared reference) and stages 1–2 and 3 or higher of the subtype led by LV "
     "myocardial dysfunction (S1) and of the subtype led by LV remodelling (S2); the small third MRI-only subtype (n=281) "
     "is not shown. Cells show the mean regional z score (higher, more abnormal) for wall thickness (template heart and "
     "bull's-eye plot of the 16 American Heart Association segments), circumferential (Ecc) and radial (Err) strain, "
     "ascending (Asc) and descending (Desc) aortic distensibility, cortical thickness (left hemisphere), the volumes of "
     "the hippocampus and six subcortical nuclei, white-matter fractional anisotropy (FA) and free water (axial section "
     "at z=+6 mm, MNI), and WMH volume. p is the median posterior probability of the assigned subtype, which was lower "
     "than in the derivation sample. †Not an input to any SuStaIn domain score. Templates are for display only.", size=8)
# ------------------------------------------------------------------ save, render, locate pages
docx_p = OUT / "Supplementary_appendix.docx"
doc.save(docx_p)
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(docx_p)], capture_output=True)
pdf_p = OUT / "Supplementary_appendix.pdf"
npages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf_p)], capture_output=True, text=True).stdout).group(1))
pages = {}
for pg in range(2, npages + 1):
    txt = subprocess.run(["pdftotext", "-f", str(pg), "-l", str(pg), "-layout", str(pdf_p), "-"], capture_output=True, text=True).stdout
    flat = re.sub(r"\s+", " ", txt)
    for key, h in ANCHORS:
        if key not in pages and re.sub(r"\s+", " ", h)[:60] in flat:
            pages[key] = pg
missing = [k for k, _ in ANCHORS if k not in pages]
json.dump(pages, open(OUT / "appendix_pages.json", "w"), indent=1)
print(f"pages={npages}; located {len(pages)}/{len(ANCHORS)}; missing={missing}")
print(pages)
