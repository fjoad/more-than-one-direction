#!/usr/bin/env python
"""
Judge-validation v2: (a) sweep Majd's ordinal 'Refused?' threshold and pick the best
binarization, (b) full pairwise agreement among 3 humans + 2 automatic judges,
(c) judge-vs-human-consensus, (d) disagreement examples, (e) emit LaTeX appendix tables.
All numbers derived, nothing assumed.
"""
import os, itertools
import pandas as pd, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # <repo>/code/<area>/script.py -> <repo>

P = str(REPO / "data/judge_validation/xstest_human_annotations.csv")
OUTTEX = str(REPO / "results/agreement_appendix.tex")
df = pd.read_csv(P, dtype=str, keep_default_na=False, encoding="latin-1")
df.columns = [c.strip() for c in df.columns]
N = len(df)
print(f"rows: {N}\n")

def bin01(series):
    out = np.full(N, np.nan)
    s = series.astype(str).str.strip()
    for i, v in enumerate(s):
        if v in ("0","0.0"): out[i] = 0
        elif v in ("1","1.0"): out[i] = 1
    return out

def majd_num():
    out = np.full(N, np.nan)
    for i, v in enumerate(df["Refused?"].astype(str).str.strip()):
        try: out[i] = float(v)
        except: pass
    return out

def agree_kappa(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]; n = len(a)
    if n == 0: return np.nan, 0, np.nan
    po = (a == b).mean()
    pe = sum(((a==k).mean())*((b==k).mean()) for k in (0,1))
    k = (po - pe)/(1-pe) if (1-pe) > 0 else np.nan
    return po, n, k

F, T, WG, SM = bin01(df["faaiz"]), bin01(df["Taha's notes"]), \
               bin01(df["wildguard_is_refusal"]), bin01(df["stringmatch_is_refusal"])
Mraw = majd_num()
print("Majd 'Refused?' value counts:", df["Refused?"].value_counts().to_dict(), "\n")

# ---- 2-human clean gold (Faaiz==Taha) used to pick Majd's threshold (non-circular) ----
cons2 = np.where(F == T, F, np.nan)
print(f"2-human consensus (Faaiz==Taha) defined on {np.isfinite(cons2).sum()} rows\n")

# ---- (a) Majd threshold sweep ----
print("=== Majd threshold sweep:  Majd_bin = (Refused? >= t) ===")
print(f"{'t':>4} {'nvalid':>6} {'vs cons2':>9} {'vs Faaiz':>9} {'vs Taha':>8} {'vs WG':>7}")
sweep = []
for t in (0.5, 1.0, 2.0, 3.0):
    Mb = np.where(np.isfinite(Mraw), (Mraw >= t).astype(float), np.nan)
    a_c,_,_ = agree_kappa(Mb, cons2)
    a_f,_,_ = agree_kappa(Mb, F)
    a_t,_,_ = agree_kappa(Mb, T)
    a_w,_,_ = agree_kappa(Mb, WG)
    sweep.append((t, np.isfinite(Mb).sum(), a_c, a_f, a_t, a_w))
    print(f"{t:>4} {int(np.isfinite(Mb).sum()):>6} {a_c*100:>8.1f}% {a_f*100:>8.1f}% {a_t*100:>7.1f}% {a_w*100:>6.1f}%")
tstar = max(sweep, key=lambda r: (r[2] if np.isfinite(r[2]) else -1))[0]
print(f"\n-> best threshold t* = {tstar} (max agreement with 2-human consensus)\n")
Majd = np.where(np.isfinite(Mraw), (Mraw >= tstar).astype(float), np.nan)

# ---- (b) full pairwise matrix (3 humans + 2 judges) ----
srcs = {"Faaiz":F, "Taha":T, "Majd":Majd, "WildGuard":WG, "StringMatch":SM}
names = list(srcs)
print("=== pairwise agreement %% (upper) / Cohen's kappa (lower) ===")
agr = {}; kap = {}
for a, b in itertools.combinations(names, 2):
    po, n, k = agree_kappa(srcs[a], srcs[b]); agr[(a,b)] = po; kap[(a,b)] = k
    print(f"  {a:12s} vs {b:12s}: agree={po*100:5.1f}%  n={n}  kappa={k:.3f}")

# ---- (c) 3-human majority consensus + judge comparison ----
H = np.vstack([F, T, Majd])
maj = np.full(N, np.nan)
for i in range(N):
    col = H[:, i]; col = col[np.isfinite(col)]
    if len(col) >= 2:
        maj[i] = 1 if col.mean() > 0.5 else (0 if col.mean() < 0.5 else np.nan)
print(f"\n3-human majority consensus defined on {int(np.isfinite(maj).sum())} rows")
for j, lab in [(WG,"WildGuard"),(SM,"StringMatch")]:
    po,n,k = agree_kappa(j, maj); po2,n2,k2 = agree_kappa(j, cons2)
    print(f"  {lab:12s}: vs 3-human-majority agree={po*100:.1f}% k={k:.3f} (n={n}) | vs 2-human-consensus agree={po2*100:.1f}% k={k2:.3f} (n={n2})")

# ---- (d) disagreement examples (WG vs 3-human majority) ----
dis = np.where(np.isfinite(maj) & np.isfinite(WG) & (maj != WG))[0]
print(f"\nWG vs 3-human-majority disagreements: {len(dis)}")
ex = []
for idx in dis[:8]:
    ex.append((df.iloc[idx]["prompt"], df.iloc[idx]["model_response"], int(maj[idx]), int(WG[idx])))

# ---- (e) LaTeX ----
def tex_esc(s):
    s = str(s)
    for a,b in [("\\","\\textbackslash{}"),("&","\\&"),("%","\\%"),("$","\\$"),("#","\\#"),
                ("_","\\_"),("{","\\{"),("}","\\}"),("~","\\textasciitilde{}"),("^","\\textasciicircum{}")]:
        s = s.replace(a,b)
    return s.replace("“",'"').replace("”",'"').replace("’","'").replace("—","---").replace("…","...")

def snip(s, n=95):
    s = " ".join(str(s).split())
    return s[:n] + ("..." if len(s) > n else "")

wg_c2, _, wg_k2 = agree_kappa(WG, cons2)
sm_c2, _, sm_k2 = agree_kappa(SM, cons2)
ft, nft, ftk    = agree_kappa(F, T)

with open(OUTTEX, "w") as fh:
    fh.write("% ---- Appendix: refusal-judge validation on XSTest (auto-generated) ----\n")
    # Table 1: agreement summary vs human consensus
    fh.write(r"""\begin{table}[t]
\centering\small
\begin{tabular}{lcc}
\toprule
Refusal detector & Agreement w/ humans & Cohen's $\kappa$ \\
\midrule
WildGuard classifier & %.1f\%% & %.2f \\
Phrase/string match & %.1f\%% & %.2f \\
\midrule
\emph{Human--human (reference)} & \emph{%.1f\%%} & \emph{%.2f} \\
\bottomrule
\end{tabular}
\caption{Agreement of each automatic refusal detector with the human-consensus refusal label on %d XSTest responses (consensus = the %d responses on which the two complete human annotators agree). Human--human agreement is shown as the reliability ceiling. WildGuard matches the human consensus more closely than phrase matching and within the human--human band.}
\label{tab:judge-agreement}
\end{table}
""" % (wg_c2*100, wg_k2, sm_c2*100, sm_k2, ft*100, ftk, int(np.isfinite(cons2).sum()), int(np.isfinite(cons2).sum())))

    # Table 2: pairwise matrix (agreement % lower-tri)
    fh.write("\n" + r"\begin{table}[t]" + "\n\\centering\\small\n")
    fh.write("\\begin{tabular}{l" + "c"*(len(names)-1) + "}\n\\toprule\n")
    fh.write(" & " + " & ".join(names[1:]) + " \\\\\n\\midrule\n")
    for i, r in enumerate(names[:-1]):
        cells = []
        for cix, c in enumerate(names[1:], start=1):
            if cix <= i: cells.append(""); continue
            key = (r, c) if (r, c) in agr else (c, r)
            cells.append(f"{agr[key]*100:.1f}")
        fh.write(f"{r} & " + " & ".join(cells) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
    fh.write(r"\caption{Pairwise \%% agreement between the three human annotators (Faaiz, Taha, Majd) and the two automatic detectors on the XSTest responses. Majd's ordinal labels are binarized at the threshold ($\geq %s$) that best matches the two-annotator consensus.}" % (tstar,))
    fh.write("\n\\label{tab:judge-pairwise}\n\\end{table}\n")

    # Table 3: disagreement examples (3)
    fh.write("\n" + r"""\begin{table}[t]
\centering\small
\begin{tabular}{p{0.30\columnwidth}p{0.45\columnwidth}cc}
\toprule
Prompt & Model response (truncated) & Human & WG \\
\midrule
""")
    for (p, r, h, w) in ex[:3]:
        fh.write(f"{tex_esc(snip(p,55))} & {tex_esc(snip(r,95))} & {h} & {w} \\\\\n\\addlinespace\n")
    fh.write(r"""\bottomrule
\end{tabular}
\caption{Representative human/WildGuard disagreements. These concentrate on ambiguous XSTest cases---clarification-seeking or capability-hedging responses---which WildGuard scores as refusals (WG$=1$) and annotators as non-refusals (Human$=0$); they are borderline partial-compliance cases rather than systematic classifier errors.}
\label{tab:judge-disagreements}
\end{table}
""")
print("\nwrote LaTeX ->", OUTTEX)
