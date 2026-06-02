import json, numpy as np
import pandas as pd
import pingouin as pg
from pathlib import Path
from scipy import stats
import sys
sys.path.insert(0, 'src')
from explore_persona_space.experiments.i406_conditions import CONDITIONS

D = json.load(open('eval_results/issue_406/divergence/D_matrix.json'))
G1 = json.load(open('eval_results/issue_462/cross_eval/G_logprob_matrix_ep1.json'))
KL = D['KL']; PT = D['prompt_tokens']; G_mat = G1['G']
cls_by_cid = {c.cid: c.cls for c in CONDITIONS}
cids = [c.cid for c in CONDITIONS]

per_cell_dir = Path('eval_results/issue_462/cross_eval/per_cell_ep1')
def length_for_cell(ci, cj):
    p = per_cell_dir / f'G_{ci}__{cj}.json'
    if p.exists():
        d = json.loads(p.read_text())
        prompt_lens = d.get('prompt_lens_per_q', [])
        R_lens = d.get('R_lens_per_q', [])
        if prompt_lens and R_lens:
            return float(np.mean([p+r for p,r in zip(prompt_lens, R_lens, strict=True)]))
    return float(PT[ci][cj])

rows = []
for ci in cids:
    for cj in cids:
        if ci == cj: continue
        g = G_mat[ci][cj]
        L = length_for_cell(ci, cj)
        rows.append({
            'T_i': ci, 'T_j': cj,
            'class_i': cls_by_cid[ci],
            'class_j': cls_by_cid[cj],
            'D': KL[ci][cj],
            'G_logprob': float(g['g_logprob']),
            'delta_g': float(g['delta_g']),
            'b_logprob': float(g['b_logprob']),
            'prompt_plus_R_tokens': L,
            'log_prompt_tokens': float(np.log(max(L, 1.0))),
        })
df = pd.DataFrame(rows).dropna(subset=['D'])

def report(label, sub):
    n = len(sub)
    sat = (sub['G_logprob'].abs() < 0.1).sum() / n
    print(f'\n== {label}: n={n}, frac_saturated={sat:.3f} ==')
    for y in ['G_logprob','delta_g']:
        # length-partial spearman
        r = pg.partial_corr(data=sub, x='D', y=y, covar=['log_prompt_tokens'], method='spearman')
        rho = float(r['r'].values[0]); p = float(r['p_val'].values[0])
        # also simple spearman for codex's claim
        rho_simple, p_simple = stats.spearmanr(sub['D'], sub[y])
        print(f'  partial-rho(D, {y:>9}) = {rho:>+.4f}, p = {p:.2e}   |  simple-rho = {rho_simple:>+.4f}, p = {p_simple:.2e}')

# Full
report('FULL ep1 (n=240)', df)

# Drop A3 (any cell where source or target is A3)
df_noA3 = df[~((df['T_i'] == 'A3') | (df['T_j'] == 'A3'))].copy()
report('Drop A3 (pirate-captain, source or target)', df_noA3)

# Drop A4
df_noA4 = df[~((df['T_i'] == 'A4') | (df['T_j'] == 'A4'))].copy()
report('Drop A4 (stand-up comedian, source or target)', df_noA4)

# Drop A3 + A4
df_no34 = df[~((df['T_i'].isin(['A3','A4'])) | (df['T_j'].isin(['A3','A4'])))].copy()
report('Drop A3 + A4 (pirate + comedian)', df_no34)

# Drop A3 + A4 + A5 (the three "stylized")
df_no345 = df[~((df['T_i'].isin(['A3','A4','A5'])) | (df['T_j'].isin(['A3','A4','A5'])))].copy()
report('Drop A3 + A4 + A5 (all 3 stylized personas)', df_no345)

# Sanity: ep1 saturation overall
sat = (df['G_logprob'].abs() < 0.1).sum() / len(df)
print(f'\nFULL ep1 saturation (g_logprob within 0.1 of 0): {sat:.4f}')
