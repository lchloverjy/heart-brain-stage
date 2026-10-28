"""Paragraph-level Chinese parallel translation of the submission manuscript (for author review only)."""
import re, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "output", "eClinicalMedicine_manuscript.md")
OUT = os.path.join(HERE, "output", "translation_zh.json")
MODEL = "gpt-6.1-sol"
client = OpenAI()

SYSTEM = """你是资深医学期刊编辑与神经/心血管流行病学专家，负责把一篇投给 eClinicalMedicine（Lancet 子刊）的英文论文逐段翻译为规范、严谨、流畅的中文学术文本，供中文作者审阅。要求：
1. 忠实原文，不增删信息，不夸大或弱化结论；保持原有的谨慎措辞（如 associated with 译为“与……相关”，不得译为因果）。
2. 所有数字、单位、置信区间、P/q 值、HR、β、ρ、n、样本量与百分比必须与原文完全一致。
3. 原样保留 <sup>…</sup> 标签及其中内容、*斜体* 与 **粗体** 标记、Markdown 表格的竖线结构、行首的 #/##/### 标题标记与编号。
4. 以下名称保持英文不译：SuStaIn、All-10、MRI + M13、MRI-only、M13、M20、M5、UK Biobank、UKB-PPP、Olink、CVIC、SCORE2、SCORE2-OP、pySuStaIn、STROBE、C-index、NRI、ICD-10、de novo。保留 Lancet 居中点小数（如 1·47）与千位空格写法（如 23 974），不得改为普通小数点或逗号。
5. 专业术语使用国内规范译法：heart–brain vascular events 心脑血管事件；derivation sample 推导样本；independent sample 独立样本；heart-first 心脏先行；co-abnormality 共异常；organ-concordant 器官对应；decision curve 决策曲线；net benefit 净获益；positional variance diagram 位置方差图；left-ventricular remodelling 左心室重构；myocardial dysfunction 心肌功能障碍；aortic stiffness 主动脉僵硬度；white-matter hyperintensity 白质高信号；cortical thinning 皮层变薄；hippocampal atrophy 海马萎缩；free water 自由水；stage 分期；subtype 亚型；hazard ratio 风险比；false discovery rate 错误发现率；landmark analysis 界标分析；cross-validation 交叉验证；just-identified 恰好识别；chance-normalized certainty 机会校正确定性；clinical risk score 临床风险评分；heart–brain vascular composite 心脑血管复合终点；non-vascular brain composite 非血管性脑疾病复合终点；re-stratify 再分层；concentration curve 事件集中曲线；multi-system involvement 多系统受累；weighted κ 加权κ；person-years 人年。
6. 方括号中的 AUTHOR ACTION 译为“【作者待补：……】”，[AA] 译为“【待补】”。
7. 只输出译文，不要任何解释。"""


def blocks_of(md):
    return [l for l in md.split("\n") if l.strip()]
IN_REFS = {"on": False}
def is_reference(b):
    if b.startswith("## References"):
        IN_REFS["on"] = True; return False
    if b.startswith("## ") and IN_REFS["on"]:
        IN_REFS["on"] = False
    return IN_REFS["on"] or b.startswith("![") or bool(re.fullmatch(r"\|[-|]+\|", b.strip()))
def translate(b):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": b}],
                max_completion_tokens=6000,
                extra_body={"reasoning": {"effort": "low"}},
            )
            out = (r.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as e:
            err = str(e)
            if "Insufficient credits" in err or "Unsupported model" in err:
                raise
    return None


def main():
    md = open(SRC).read()
    blocks = blocks_of(md)
    # references block: keep English; everything else translated
    in_refs = False
    jobs = []
    for i, b in enumerate(blocks):
        if b.startswith("## References"):
            in_refs = True; jobs.append((i, b, "keep_heading")); continue
        if in_refs and b.startswith("## ") or (in_refs and b.startswith("**Table")):
            in_refs = False
        if in_refs or b.startswith("![") or re.fullmatch(r"\|[-|]+\|", b.strip()):
            jobs.append((i, b, "keep")); continue
        jobs.append((i, b, "tr"))
    cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [(i, b) for i, b, k in jobs if k == "tr" and b not in cache]
    print(f"blocks {len(blocks)}, to translate {len(todo)}", flush=True)
    with ThreadPoolExecutor(6) as ex:
        for (i, b), out in zip(todo, ex.map(lambda x: translate(x[1]), todo)):
            if out:
                cache[b] = out
            print(i, "ok" if out else "FAIL", flush=True)
    json.dump(cache, open(OUT, "w"), ensure_ascii=False, indent=0)
    res = []
    for i, b, k in jobs:
        if k == "keep_heading":
            res.append("## 参考文献（保留英文原格式）")
        elif k == "keep":
            res.append(b)
        else:
            res.append(cache.get(b, "【翻译失败，保留原文】\n" + b))
    open(os.path.join(HERE, "output", "eClinicalMedicine_manuscript_中文对照稿.md"), "w").write("\n\n".join(res))
    # integrity check: sup tags and numbers
    bad = []
    for b, z in cache.items():
        s1 = re.findall(r"<sup>[^<]+</sup>", b); s2 = re.findall(r"<sup>[^<]+</sup>", z)
        n1 = sorted(re.findall(r"\d+(?:\.\d+)?", re.sub(r"<sup>[^<]+</sup>", "", b)))
        n2 = sorted(re.findall(r"\d+(?:\.\d+)?", re.sub(r"<sup>[^<]+</sup>", "", z)))
        if s1 != s2 or n1 != n2:
            bad.append((b[:80], len(s1), len(s2), sorted(set(n1) ^ set(n2))[:10]))
    print("integrity issues:", len(bad))
    for x in bad:
        print(x)


if __name__ == "__main__":
    main()
