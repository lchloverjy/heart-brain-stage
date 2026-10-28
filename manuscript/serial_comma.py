"""Apply the eClinicalMedicine/Lancet serial (Oxford) comma to genuine lists of three or more items.

Each replacement is an explicit phrase (no blind regex) so clause commas are never touched.
Run once; re-running is safe because every target string no longer exists after replacement.
"""
import sys

PAIRS = [
    # outcomes and disease lists
    ("ischaemic heart disease, heart failure and atrial fibrillation", "ischaemic heart disease, heart failure, and atrial fibrillation"),
    ("stroke, heart failure, atrial fibrillation and cardiovascular death", "stroke, heart failure, atrial fibrillation, and cardiovascular death"),
    ("stroke, heart failure and cardiovascular death", "stroke, heart failure, and cardiovascular death"),
    ("troke, myocardial infarction or heart failure", "troke, myocardial infarction, or heart failure"),
    ("Stroke, MI or HF", "Stroke, MI, or HF"),
    ("Stroke, MI or heart failure", "Stroke, MI, or heart failure"),
    ("heart disease, heart failure or atrial fibrillation", "heart disease, heart failure, or atrial fibrillation"),
    ("cerebrovascular events, cardiac events, the individual components and cardiovascular death",
     "cerebrovascular events, cardiac events, the individual components, and cardiovascular death"),
    ("myocardial infarction, transient ischaemic attack, all-cause death and anxiety",
     "myocardial infarction, transient ischaemic attack, all-cause death, and anxiety"),
    ("cardiovascular, cerebrovascular and neurological disease", "cardiovascular, cerebrovascular, and neurological disease"),
    ("I60, I61, I63 and I64", "I60, I61, I63, and I64"),
    ("cholesterol, smoking and diabetes", "cholesterol, smoking, and diabetes"),
    ("I11.0, I13.0 and I13.2", "I11.0, I13.0, and I13.2"),
    ("anxiety, Parkinson's disease and chronic kidney disease", "anxiety, Parkinson's disease, and chronic kidney disease"),
    # markers and models
    ("three cardiac MRI, four brain MRI and three plasma proteomic markers", "three cardiac MRI, four brain MRI, and three plasma proteomic markers"),
    ("three cardiac MRI domain scores, four brain MRI domain scores and three plasma protein",
     "three cardiac MRI domain scores, four brain MRI domain scores, and three plasma protein"),
    ("LV remodelling, LV myocardial dysfunction and aortic stiffness", "LV remodelling, LV myocardial dysfunction, and aortic stiffness"),
    ("M13, M20 and M5", "M13, M20, and M5"),
    ("followed by the M13 module, hippocampal atrophy and cortical thinning", "followed by the M13 module, hippocampal atrophy, and cortical thinning"),
    ("first cardiac, protein and brain event", "first cardiac, protein, and brain event"),
    ("proteomics 45 522, cardiac MRI 29 081 and brain MRI 39 155", "proteomics 45 522, cardiac MRI 29 081, and brain MRI 39 155"),
    ("heart–brain stage, subtype and assignment probabilities", "heart–brain stage, subtype, and assignment probabilities"),
    # thresholds and groups
    ("z=1, 2 and 3", "z=1, 2, and 3"),
    ("z=1, 2 or 3", "z=1, 2, or 3"),
    ("0, 1, 2 and 3 or higher", "0, 1, 2, and 3 or higher"),
    ("204, 10, 42 and 23 records", "204, 10, 42, and 23 records"),
    # covariates
    ("total and HDL cholesterol, BMI and hypertension", "total and HDL cholesterol, BMI, and hypertension"),
    ("total and HDL cholesterol, body-mass index and hypertension", "total and HDL cholesterol, body-mass index, and hypertension"),
    ("HbA1c, cystatin C and Townsend deprivation index", "HbA1c, cystatin C, and Townsend deprivation index"),
    # follow-up end
    ("the event, death or Sept 30, 2021", "the event, death, or Sept 30, 2021"),
    ("first event, death or Sept 30, 2021", "first event, death, or Sept 30, 2021"),
    ("diagnosis, death or Sept 30, 2021", "diagnosis, death, or Sept 30, 2021"),
    # background and discussion
    ("white matter hyperintensities, brain volumes and cognition", "white matter hyperintensities, brain volumes, and cognition"),
    ("shared haemodynamics, autonomic and neurohumoral signalling and inflammation",
     "shared haemodynamics, autonomic and neurohumoral signalling, and inflammation"),
    ('"event-based model" or "disease progression model"', '"event-based model", or "disease progression model"'),
    # declarations
    ("data collection, data analysis, data interpretation or writing of the report",
     "data collection, data analysis, data interpretation, or writing of the report"),
    ("tool, version and scope of use", "tool, version, and scope of use"),
    # second pass (systematic scan of the built manuscript, parentheticals removed)
    ("stroke, cardiovascular disease or heart failure", "stroke, cardiovascular disease, or heart failure"),
    ("white matter hyperintensities, brain volume and cognition", "white matter hyperintensities, brain volume, and cognition"),
    ("share risk factors, frequently co-occur and together account", "share risk factors, frequently co-occur, and together account"),
    ("cardiac MRI, brain MRI and plasma proteomic markers", "cardiac MRI, brain MRI, and plasma proteomic markers"),
    ("improved risk assessment beyond SCORE2 and whether these findings", "improved risk assessment beyond SCORE2, and whether these findings"),
    ("LV myocardial dysfunction and aortic stiffness, derived", "LV myocardial dysfunction, and aortic stiffness, derived"),
    ("(WMH) burden and white-matter free water", "(WMH) burden, and white-matter free water"),
    ("myocardial infarction [I21–I23] or heart failure", "myocardial infarction [I21–I23], or heart failure"),
    ("abnormal marker count or mean z score", "abnormal marker count, or mean z score"),
    (" at stages 1–2 and {", " at stages 1–2, and {"),
    ("{s2lo_g0}, {s2lo_g12} and ", "{s2lo_g0}, {s2lo_g12}, and "),
    ("{is2lo_g0}, {is2lo_g12} and ", "{is2lo_g0}, {is2lo_g12}, and "),
    ("for the abnormal marker count and {cmp_c_meanz}", "for the abnormal marker count, and {cmp_c_meanz}"),
    ("modelled cardiac, brain and proteomic markers", "modelled cardiac, brain, and proteomic markers"),
    ("excluded prevalent disease and accounted for deaths", "excluded prevalent disease, and accounted for deaths"),
    ("approved the final version and had final responsibility", "approved the final version, and had final responsibility"),
    ("Analysis code, model thresholds and aggregate results", "Analysis code, model thresholds, and aggregate results"),
    ('"Plasma proteomics, cardiac MRI and brain MRI"', '"Plasma proteomics, cardiac MRI, and brain MRI"'),
    ("Study design, heart–brain markers and time axis", "Study design, heart–brain markers, and time axis"),
    ("had no proteomic data and was not used for model fitting", "had no proteomic data, and was not used for model fitting"),
    ("order of cardiac, protein and brain abnormality", "order of cardiac, protein, and brain abnormality"),
    ("the brain in coronal section and the three protein modules", "the brain in coronal section, and the three protein modules"),
    ("I13.0, and I13.2) and atrial fibrillation (I48)", "I13.0, and I13.2), and atrial fibrillation (I48)"),
    ("current smoking ({beta_smk} SD) and SCORE2", "current smoking ({beta_smk} SD), and SCORE2"),
    ("heart failure ({hf_m2}) and ischaemic heart disease", "heart failure ({hf_m2}), and ischaemic heart disease"),
    ("stroke ({pre_stroke}) and heart failure", "stroke ({pre_stroke}), and heart failure"),
    ("atrial fibrillation ({ind_af}) and cardiovascular death", "atrial fibrillation ({ind_af}), and cardiovascular death"),
    ("Parkinson's disease ({ind_pd}) or chronic kidney disease", "Parkinson's disease ({ind_pd}), or chronic kidney disease"),
    ("for cerebrovascular events and {dc_stroke}", "for cerebrovascular events, and {dc_stroke}"),
    ("stroke ({ind_dc_stroke}) and heart failure", "stroke ({ind_dc_stroke}), and heart failure"),
    ("Data are mean (SD), median (IQR) or n (%)", "Data are mean (SD), median (IQR), or n (%)"),
    ("(logistic regression adjusted for age and sex) and with incident", "(logistic regression adjusted for age and sex), and with incident"),
]

if __name__ == "__main__":
    for fn in sys.argv[1:]:
        s = open(fn, encoding="utf-8").read()
        n_tot = 0
        for a, b in PAIRS:
            n = s.count(a)
            if n:
                s = s.replace(a, b); n_tot += n
        open(fn, "w", encoding="utf-8").write(s)
        print(f"{fn}: {n_tot} replacements")
