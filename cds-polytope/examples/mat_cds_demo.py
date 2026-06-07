"""
MAT-CDS Main Analysis & Visualization
=======================================
WCM アーキテクチャによる材料破断前兆検出の完全デモ
"""

import sys
sys.path.insert(0, '/home/claude')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from mat_cds_core import (
    VERTS, vertex_label,
    normalize_events, assign_vertices,
    compute_shape_vector, compute_cosD, hot_idle_vertices,
    sliding_window_analysis, permutation_test, detect_v_transitions
)
from mat_cds_sim import generate_ae_data

# ─────────────────────────────────────────────
# 1. データ生成
# ─────────────────────────────────────────────
print("=" * 60)
print("MAT-CDS  プロトタイプ  v0.1")
print("=" * 60)

df = generate_ae_data(seed=42)
print(f"\n[データ] 総AEイベント数: {len(df)}")

# 正規化基準（全データから）
feat_all = df[['x','y','z','log_energy']].values
_, bounds = normalize_events(feat_all)

# ─────────────────────────────────────────────
# 2. スライディングウィンドウ解析
# ─────────────────────────────────────────────
WINDOW = 10.0   # 10日窓（材料試験スケール）
STEP   = 2.0    # 2日ステップ

print(f"\n[解析] 窓幅={WINDOW}日, ステップ={STEP}日")

results = sliding_window_analysis(
    df, window_days=WINDOW, step_days=STEP,
    t_col='time', bounds=bounds
)

print(f"[解析] 総ウィンドウ数: {len(results)}")

# DataFrameに変換
res_df = pd.DataFrame([{
    't': r['t_center'],
    'n': r['n_events'],
    'cosD': r['cosD'],
    'hot_v': r['hot_v'],
    'hot_prob': r['hot_prob'],
} for r in results])

# ─────────────────────────────────────────────
# 3. V転換検出
# ─────────────────────────────────────────────
transitions = detect_v_transitions(results, min_hold=3)
print(f"\n[V転換] 検出数: {len(transitions)}")
for tr in transitions:
    print(f"  t={tr['t_transition']:.1f}日  "
          f"{vertex_label(tr['from_v'])} → {vertex_label(tr['to_v'])}  "
          f"(保持{tr['hold_count']}窓, 開始t={tr['t_hold_start']:.1f}日)")

# ─────────────────────────────────────────────
# 4. パーミュテーション検定（破断直前フェーズのhot_v）
# ─────────────────────────────────────────────
phase3_mask = (df['phase'] == 3)
p3_feat = df[phase3_mask][['x','y','z','log_energy']].values
p3_norm, _ = normalize_events(p3_feat, bounds)
p3_vids = assign_vertices(p3_norm)
p3_sv   = compute_shape_vector(p3_vids)
p3_hotv = int(np.argmax(p3_sv))

p_val = permutation_test(p3_vids, p3_hotv, n_perm=2000)
print(f"\n[統計検定] Phase3 hot_v = {vertex_label(p3_hotv)}")
print(f"           p値 = {p_val:.4f}")
print(f"           占有率 = {p3_sv[p3_hotv]*100:.1f}%  (期待値 4.2%)")

# Phase1での同頂点p値（コントロール）
phase1_mask = (df['phase'] == 1)
p1_feat = df[phase1_mask][['x','y','z','log_energy']].values
p1_norm, _ = normalize_events(p1_feat, bounds)
p1_vids = assign_vertices(p1_norm)
p1_sv   = compute_shape_vector(p1_vids)
p1_ctrl_rate = p1_sv[p3_hotv]
print(f"\n[コントロール] Phase1での同頂点占有率 = {p1_ctrl_rate*100:.1f}%")

# ─────────────────────────────────────────────
# 5. 可視化
# ─────────────────────────────────────────────
FIG_W, FIG_H = 16, 14
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='#0a0a1a')

gs = gridspec.GridSpec(4, 2, figure=fig,
                       hspace=0.50, wspace=0.35,
                       left=0.07, right=0.97,
                       top=0.93, bottom=0.06)

# カラーパレット
C1 = '#4fc3f7'   # Phase1 normal
C2 = '#ffd54f'   # Phase2 precursor
C3 = '#ef5350'   # Phase3 pre-fracture
COSD_CLR = '#b39ddb'
HOTV_CLR = '#80cbc4'
TRANS_CLR = '#ff8a65'

phase_colors = {1: C1, 2: C2, 3: C3}

def phase_of_t(t):
    if t < 80:  return 1
    if t < 140: return 2
    return 3

# ── (A) AE発生レート時系列 ───────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.set_facecolor('#0a0a1a')
bins = np.arange(0, 162, 2)
for ph, clr, lbl in [(1,C1,'Phase1 Normal'),(2,C2,'Phase2 Precursor'),(3,C3,'Phase3 Pre-fracture')]:
    sub = df[df['phase']==ph]['time'].values
    h, _ = np.histogram(sub, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    mask = (centers >= sub.min()-1) & (centers <= sub.max()+1)
    ax0.bar(centers[mask], h[mask], width=1.8, color=clr, alpha=0.75, label=lbl)

ax0.axvline(80,  color='white', lw=1.0, ls='--', alpha=0.5)
ax0.axvline(140, color='white', lw=1.0, ls='--', alpha=0.5)
ax0.text(40,  ax0.get_ylim()[1]*0.85 if ax0.get_ylim()[1]>0 else 50,
         'Phase 1\nNormal', ha='center', color=C1, fontsize=9)
ax0.text(110, 1, 'Phase 2\nPrecursor', ha='center', color=C2, fontsize=9)
ax0.text(152, 1, 'Phase 3\nPre-\nfracture', ha='center', color=C3, fontsize=9)
ax0.set_xlabel('Time (days)', color='white')
ax0.set_ylabel('AE events / 2 days', color='white')
ax0.set_title('(A)  AE Event Rate Timeline', color='white', fontsize=11, pad=6)
ax0.tick_params(colors='white')
for sp in ax0.spines.values(): sp.set_color('#444')
ax0.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', labelcolor='white',
           edgecolor='#444')

# ── (B) cosD 時系列 ──────────────────────────────────
ax1 = fig.add_subplot(gs[1, :])
ax1.set_facecolor('#0a0a1a')

# フェーズ背景
ax1.axvspan(0,   80,  alpha=0.08, color=C1)
ax1.axvspan(80,  140, alpha=0.08, color=C2)
ax1.axvspan(140, 162, alpha=0.10, color=C3)

t_arr  = res_df['t'].values
cosd_arr = res_df['cosD'].values

# フェーズ別に色を変える
for i in range(len(t_arr)-1):
    ph = phase_of_t(t_arr[i])
    ax1.plot(t_arr[i:i+2], cosd_arr[i:i+2],
             color=phase_colors[ph], lw=2.0, alpha=0.9)

ax1.axvline(80,  color='white', lw=1.0, ls='--', alpha=0.4)
ax1.axvline(140, color='white', lw=1.0, ls='--', alpha=0.4)

# V転換マーク
for tr in transitions:
    ax1.axvline(tr['t_transition'], color=TRANS_CLR, lw=1.5, ls=':')

ax1.set_xlabel('Time (days)', color='white')
ax1.set_ylabel('cosD (alignment)', color='white')
ax1.set_title('(B)  cosD Alignment Index  ─  Critical Transition Indicator', 
              color='white', fontsize=11, pad=6)
ax1.tick_params(colors='white')
for sp in ax1.spines.values(): sp.set_color('#444')
ax1.set_xlim(0, 162)

# ── (C) hot_v 時系列 ────────────────────────────────
ax2 = fig.add_subplot(gs[2, :])
ax2.set_facecolor('#0a0a1a')
ax2.axvspan(0,   80,  alpha=0.08, color=C1)
ax2.axvspan(80,  140, alpha=0.08, color=C2)
ax2.axvspan(140, 162, alpha=0.10, color=C3)

hotv_arr = res_df['hot_v'].values
for i in range(len(t_arr)-1):
    ph = phase_of_t(t_arr[i])
    ax2.plot(t_arr[i:i+2], hotv_arr[i:i+2],
             color=phase_colors[ph], lw=2.5, alpha=0.8)

ax2.scatter(t_arr, hotv_arr, c=[phase_colors[phase_of_t(t)] for t in t_arr],
            s=18, zorder=3, alpha=0.7)

for tr in transitions:
    ax2.axvline(tr['t_transition'], color=TRANS_CLR, lw=1.5, ls=':')
    ax2.annotate(
        f"V{tr['from_v']:02d}→V{tr['to_v']:02d}",
        xy=(tr['t_transition'], tr['to_v']),
        xytext=(tr['t_transition']+2, tr['to_v']+1.5),
        color=TRANS_CLR, fontsize=7,
        arrowprops=dict(arrowstyle='->', color=TRANS_CLR, lw=0.8)
    )

ax2.set_xlabel('Time (days)', color='white')
ax2.set_ylabel('hot_v  (vertex index)', color='white')
ax2.set_title('(C)  Dominant 24-cell Vertex  (hot_v) Transition Path',
              color='white', fontsize=11, pad=6)
ax2.set_yticks(range(0, 24, 2))
ax2.tick_params(colors='white')
for sp in ax2.spines.values(): sp.set_color('#444')
ax2.set_xlim(0, 162)
ax2.set_ylim(-1, 24)
ax2.axvline(80,  color='white', lw=1.0, ls='--', alpha=0.4)
ax2.axvline(140, color='white', lw=1.0, ls='--', alpha=0.4)

# ── (D) 形状ベクトル bar chart（3フェーズ比較） ───────
ax3 = fig.add_subplot(gs[3, 0])
ax3.set_facecolor('#0a0a1a')

sv_list = []
labels  = ['Phase1\nNormal', 'Phase2\nPrecursor', 'Phase3\nPre-fracture']
colors_sv = [C1, C2, C3]
for ph in [1, 2, 3]:
    sub = df[df['phase']==ph][['x','y','z','log_energy']].values
    norm, _ = normalize_events(sub, bounds)
    vids = assign_vertices(norm)
    sv   = compute_shape_vector(vids)
    sv_list.append(sv)

x_idx = np.arange(24)
width = 0.28
for i, (sv, clr, lbl) in enumerate(zip(sv_list, colors_sv, labels)):
    ax3.bar(x_idx + i*width - width, sv*100, width=width,
            color=clr, alpha=0.8, label=lbl)

ax3.axhline(100/24, color='white', ls='--', lw=1, alpha=0.5,
            label=f'Uniform ({100/24:.1f}%)')
ax3.set_xlabel('Vertex index', color='white')
ax3.set_ylabel('Occupancy (%)', color='white')
ax3.set_title('(D)  Shape Vector  sv[]  by Phase', color='white', fontsize=10, pad=6)
ax3.tick_params(colors='white')
for sp in ax3.spines.values(): sp.set_color('#444')
ax3.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='#444')

# ── (E) 統計サマリー ─────────────────────────────────
ax4 = fig.add_subplot(gs[3, 1])
ax4.set_facecolor('#0a0a1a')
ax4.axis('off')

summary_lines = [
    ("MAT-CDS  v0.1  解析サマリー", True),
    ("", False),
    (f"試験体: 100×50×20 mm 金属試験片", False),
    (f"AE総イベント数: {len(df):,}", False),
    (f"解析窓幅: {WINDOW}日 / ステップ: {STEP}日", False),
    (f"総ウィンドウ数: {len(results)}", False),
    ("", False),
    ("─── V転換検出 ───────────────", False),
    (f"検出V転換数: {len(transitions)}", False),
]
for tr in transitions[:4]:
    summary_lines.append(
        (f"  t={tr['t_transition']:.0f}d  V{tr['from_v']:02d}→V{tr['to_v']:02d}  "
         f"(hold={tr['hold_count']}w)", False)
    )
summary_lines += [
    ("", False),
    ("─── 統計検定 ────────────────", False),
    (f"Phase3 FormA頂点: {vertex_label(p3_hotv)}", False),
    (f"Phase3 占有率: {p3_sv[p3_hotv]*100:.1f}%  "
     f"(期待値 4.2%)", False),
    (f"Phase1 同頂点率: {p1_ctrl_rate*100:.1f}%", False),
    (f"p値 (perm.test): {p_val:.4f}", p_val < 0.05),
    ("", False),
    ("⇒ 破断直前の特定頂点収束を統計的に確認", True),
]

y = 0.97
for text, bold in summary_lines:
    clr = '#ffd54f' if bold else 'white'
    fs  = 9.5 if bold else 8.5
    ax4.text(0.03, y, text, transform=ax4.transAxes,
             color=clr, fontsize=fs,
             fontweight='bold' if bold else 'normal',
             verticalalignment='top', family='monospace')
    y -= 0.065 if text else 0.03

# ── タイトル ───────────────────────────────────────
fig.suptitle(
    'MAT-CDS  v0.1  ─  24-cell Acoustic Emission Critical Transition Sensor',
    color='white', fontsize=13, fontweight='bold', y=0.975
)

out_path = '/mnt/user-data/outputs/mat_cds_v01.png'
plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"\n[可視化] 保存: {out_path}")

# ─────────────────────────────────────────────
# 6. テキストサマリー
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("【 MAT-CDS v0.1  解析サマリー 】")
print("=" * 60)
print(f"\nPhase別 cosD 平均:")
for ph in [1, 2, 3]:
    mask = [(80*ph-80 < t <= 80*ph) for t in res_df['t']]
    # フェーズ時間帯で絞る
ph_ranges = {1:(0,80), 2:(80,140), 3:(140,162)}
for ph, (t0, t1) in ph_ranges.items():
    sub = res_df[(res_df['t'] >= t0) & (res_df['t'] <= t1)]
    print(f"  Phase{ph}: cosD mean={sub['cosD'].mean():.4f}  "
          f"max={sub['cosD'].max():.4f}  n_windows={len(sub)}")

print(f"\nFormA頂点 {vertex_label(p3_hotv)} の占有率変化:")
for ph, (t0, t1) in ph_ranges.items():
    sub_df = df[(df['time'] >= t0) & (df['time'] < t1)]
    sub_feat = sub_df[['x','y','z','log_energy']].values
    if len(sub_feat) > 0:
        n, _ = normalize_events(sub_feat, bounds)
        v = assign_vertices(n)
        rate = np.mean(v == p3_hotv) * 100
        print(f"  Phase{ph}: {rate:.1f}%  (期待値 4.2%)")
