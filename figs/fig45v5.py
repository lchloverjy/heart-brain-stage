"""Figures 4 and 5 (v5) - renumbered from v4 Figures 3 and 4 without any change to estimates or layout.

v4 Figure 3 (events after imaging)      -> v5 Figure 4   (module fig3v4)
v4 Figure 4 (clinical value beyond SCORE2) -> v5 Figure 5 (module fig4v4)

Usage: python3 fig45v5.py [4|5|all]
"""
import shutil
import sys
import tempfile
from pathlib import Path

import style4 as S

HERE = Path(__file__).resolve().parent
MAP = {"4": ("fig3v4", "Fig3", "Figure3", "Fig4", "Figure4"),
       "5": ("fig4v4", "Fig4", "Figure4", "Fig5", "Figure5")}


def build(num):
    mod, old_p, old_c, new_p, new_c = MAP[num]
    M = __import__(mod)
    out = HERE / "output_v5" / new_c
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        M.OUT = tmp
        S.set_style()
        M.render("all")
        for f in sorted(Path(tmp).iterdir()):
            name = f.name.replace(old_c, new_c).replace(old_p, new_p)
            shutil.copy2(f, out / name)
            print("saved", out / name)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for n in (["4", "5"] if which == "all" else [which]):
        build(n)
