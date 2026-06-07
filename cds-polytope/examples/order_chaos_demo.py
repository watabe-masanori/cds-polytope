"""
CDS Extended — Order-Chaos Spectrum
=====================================
「乱れの方向への収束」の数学的実装

【 ポロックの洞察 】
    Jackson Pollock のドリップペインティングは
    一見カオスだが、フラクタル次元 D ≈ 1.7 という
    「特定の乱れ方」に収束している。
    （Richard Taylor 1999, Nature）

    これは「整列収束（WCM FormA型）」とは逆の現象：
    ・整列収束：全イベントが1方向へ → エントロピー H 低下
    ・乱れ収束：イベントが「特定の乱れ方」へ → H はほぼ一定だが
                「特定のパターン」への収束が起きている

【 新指標：Alignment Polarity (AP) 】
    AP = −ΔH_norm × f(cosD)

    AP > 0  整列収束  (WCM FormA、地震前兆、材料破断)
    AP ≈ 0  臨界点    (秩序⇔混沌の境界 ← 最も危険)
    AP < 0  乱れ収束  (ECG不整脈、金融クラッシュ、ポロック的状態)

【 フラクタル次元との接続 】
    整列収束 → 空間分布が1点に集中 → D → 0
    乱れ収束 → 空間分布がフラクタル → D ≈ 1.5〜2.0
    完全ランダム → D → 3（空間次元）
"""

import sys
sys.path.insert(0, '/home/claude')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import uniform_filter1d

from cds_highdim_core import CDSEngine, E8_UNIT, C24_4D_UNIT
from auto_axis_selector_v3 import AutoAxisSelectorV3
from mat_cds_sim import generate_ae_data


# ═══════════════════════════════════════════════════════
# 1. 拡張 CDSEngine：AP指標を追加
# ═══════════════════════════════════════════════════════
class CDSEngineExtended(CDSEngine):
    """
    CDSEngine に Alignment Polarity (AP) を追加した拡張版。
    """
    def alignment_polarity(self, sv: np.ndarray,
                            sv_baseline: np.ndarray) -> float:
        """
        AP = −ΔH_norm × tanh(cosD × 5)

        Parameters
        ----------
        sv          : 現在の形状ベクトル
        sv_baseline : Phase1（正常期）の平均形状ベクトル
        """
        H_max = np.log(self.K)
        H_now = -np.sum(sv * np.log(sv + 1e-12))
        H_bas = -np.sum(sv_baseline * np.log(sv_baseline + 1e-12))
        dH    = (H_now - H_bas) / H_max          # ΔH 正規化
        cosD  = self.cosD(sv)
        AP    = -dH * np.tanh(cosD * 5)
        return float(AP)

    def fractal_proxy(self, sv: np.ndarray) -> float:
        """
        sv[] から「フラクタル次元プロキシ」を計算。
        情報次元 D1 = H / log(K) の逆数的な量。
        D1 → 0: 完全整列（1点集中）
        D1 → 1: 完全ランダム（一様分布）
        """
        H = -np.sum(sv * np.log(sv + 1e-12))
        return float(H / np.log(self.K))

    def scan_extended(self, events_all, times, window, step,
                      bounds=None, phase_baseline=(0, 80)):
        """
        通常のスキャン + AP + フラクタルプロキシ
        """
        if bounds is None:
            _, bounds = self.normalize(events_all)

        # Phase1 のベースライン sv を先に計算
        t0b, t1b = phase_baseline
        mask_b   = (times >= t0b) & (times < t1b)
        if mask_b.sum() >= 5:
            normed_b, _ = self.normalize(events_all[mask_b], bounds)
            rids_b       = self.assign(normed_b)
            sv_baseline  = self.shape_vector(rids_b)
        else:
            sv_baseline  = self.uniform.copy()

        # スライディングウィンドウ
        t_start, t_end = times.min(), times.max()
        results = []
        t = t_start
        while t + window <= t_end + step:
            mask = (times >= t) & (times < t + window)
            win  = events_all[mask]
            r    = self.analyze_window(win, bounds)

            sv = r['sv']
            r['AP']      = self.alignment_polarity(sv, sv_baseline)
            r['frac_D']  = self.fractal_proxy(sv)
            r['t_center']= t + window / 2
            r['n']       = len(win)
            results.append(r)
            t += step

        return results, sv_baseline


# ═══════════════════════════════════════════════════════
# 2. 3種のシミュレーションデータを生成
# ═══════════════════════════════════════════════════════
rng = np.random.default_rng(42)

def make_aligned_data(rng, N=5000):
    """WCM型：Phase3でFormAに整列収束"""
    df    = generate_ae_data(seed=42)
    return df[['x','y','z','log_energy']].values, df['time'].values, df['phase'].values

def make_chaos_data(rng, N=5000):
    """ポロック型：Phase3で「乱れ」に収束（カオス的臨界）"""
    t   = np.sort(rng.uniform(0, 160, N))
    ph  = np.where(t < 80, 1, np.where(t < 140, 2, 3))

    # Phase1: 緩やかなランダム
    # Phase2: 揺らぎが増大
    # Phase3: 複数の方向にバラバラに飛ぶ（多峰性分布）
    x = np.zeros(N); y = np.zeros(N)
    z = np.zeros(N); e = np.zeros(N)

    for i in range(N):
        if ph[i] == 1:
            x[i] = rng.normal(0.5, 0.1)
            y[i] = rng.normal(0.5, 0.1)
            z[i] = rng.normal(0.5, 0.1)
            e[i] = rng.normal(0.5, 0.1)
        elif ph[i] == 2:
            x[i] = rng.normal(0.5, 0.25)
            y[i] = rng.normal(0.5, 0.25)
            z[i] = rng.normal(0.5, 0.25)
            e[i] = rng.normal(0.5, 0.25)
        else:
            # 8つの極端な方向に散乱（フラクタル的分散）
            cluster = rng.integers(0, 8)
            cx = [0,1,0,1,0,1,0,1][cluster]
            cy = [0,0,1,1,0,0,1,1][cluster]
            cz = [0,0,0,0,1,1,1,1][cluster]
            x[i] = rng.normal(cx, 0.08)
            y[i] = rng.normal(cy, 0.08)
            z[i] = rng.normal(cz, 0.08)
            e[i] = rng.normal(0.5, 0.3)

    feat = np.column_stack([x, y, z, e])
    return feat, t, ph

def make_pollock_data(rng, N=8000):
    """
    ポロック的データ：
    フラクタル次元 D≈1.7 のパターンを持つ「乱れ」が
    Phase3で出現する。
    Lévy飛行（べき乗則ステップ）で生成。
    """
    t   = np.sort(rng.uniform(0, 160, N))
    ph  = np.where(t < 80, 1, np.where(t < 140, 2, 3))

    x = np.zeros(N); y = np.zeros(N)
    z = np.zeros(N); e = np.zeros(N)

    levy_alpha = 1.7   # ポロックのフラクタル次元に対応

    for i in range(N):
        if ph[i] == 1:
            # 正常：短距離のランダムウォーク
            x[i] = rng.normal(0.5, 0.1)
            y[i] = rng.normal(0.5, 0.1)
            z[i] = rng.normal(0.5, 0.1)
            e[i] = rng.exponential(0.1)

        elif ph[i] == 2:
            # 前兆：ステップサイズが増大
            step = rng.pareto(2.5) * 0.1  # 重い尾
            angle = rng.uniform(0, 2*np.pi)
            x[i] = np.clip(0.5 + step*np.cos(angle), 0, 1)
            y[i] = np.clip(0.5 + step*np.sin(angle), 0, 1)
            z[i] = rng.normal(0.5, 0.2)
            e[i] = rng.exponential(0.2)

        else:
            # ポロック型：Lévy飛行（フラクタル次元≈1.7）
            # べき乗則 P(step) ∝ step^(-levy_alpha)
            step  = rng.pareto(levy_alpha - 1) * 0.15
            angle = rng.uniform(0, 2*np.pi)
            elev  = rng.uniform(0, np.pi)
            x[i] = np.clip(0.5 + step*np.sin(elev)*np.cos(angle), 0,1)
            y[i] = np.clip(0.5 + step*np.sin(elev)*np.sin(angle), 0,1)
            z[i] = np.clip(0.5 + step*np.cos(elev),               0,1)
            e[i] = rng.pareto(1.5) * 0.1   # エネルギーも重い尾

    feat = np.column_stack([x, y, z, e])
    return feat, t, ph


# ═══════════════════════════════════════════════════════
# 3. 解析実行
# ═══════════════════════════════════════════════════════
eng_ext = CDSEngineExtended(C24_4D_UNIT, name='CDS-4D Extended')
WINDOW, STEP = 10.0, 2.0

datasets = {
    'Aligned\n(WCM/Material)': make_aligned_data(rng),
    'Chaos\n(ECG/Crash)':      make_chaos_data(rng),
    'Pollock\n(Lévy/Fractal)': make_pollock_data(rng),
}

print("=" * 60)
print("CDS Order-Chaos Spectrum  —  Alignment Polarity (AP)")
print("=" * 60)

results_all = {}
for name, (feat, times, phases) in datasets.items():
    _, bounds = eng_ext.normalize(feat)
    res, sv_base = eng_ext.scan_extended(
        feat, times, WINDOW, STEP, bounds)

    t_arr  = np.array([r['t_center'] for r in res])
    cosD   = np.array([r['cosD']     for r in res])
    AP     = np.array([r['AP']       for r in res])
    fracD  = np.array([r['frac_D']   for r in res])
    hotv   = np.array([r['hot_v']    for r in res])

    m1 = (t_arr >= 0)   & (t_arr < 80)
    m3 = (t_arr >= 140) & (t_arr < 162)
    AP_ph1  = AP[m1].mean() if m1.sum()>0 else 0
    AP_ph3  = AP[m3].mean() if m3.sum()>0 else 0
    fD_ph1  = fracD[m1].mean() if m1.sum()>0 else 0
    fD_ph3  = fracD[m3].mean() if m3.sum()>0 else 0

    results_all[name] = {
        't': t_arr, 'cosD': cosD, 'AP': AP,
        'fracD': fracD, 'hotv': hotv,
        'AP_ph1': AP_ph1, 'AP_ph3': AP_ph3,
        'fD_ph1': fD_ph1, 'fD_ph3': fD_ph3,
    }

    tag = name.split('\n')[0]
    print(f"\n[{tag}]")
    print(f"  AP  Ph1→Ph3: {AP_ph1:+.4f} → {AP_ph3:+.4f}  "
          f"{'整列収束↑' if AP_ph3>0.01 else '乱れ収束↓' if AP_ph3<-0.01 else '臨界点⚡'}")
    print(f"  FD  Ph1→Ph3: {fD_ph1:.4f} → {fD_ph3:.4f}  "
          f"({'秩序化' if fD_ph3<fD_ph1 else 'カオス化'})")


# ═══════════════════════════════════════════════════════
# 4. 可視化
# ═══════════════════════════════════════════════════════
# カスタムカラーマップ：青（乱れ）→白（臨界）→赤（整列）
cmap_ap = LinearSegmentedColormap.from_list(
    'pollock_order',
    ['#4169e1','#7ec8e3','#f0f0f0','#ff8c69','#dc143c'],
    N=256
)

fig = plt.figure(figsize=(20, 22), facecolor='#030308')
gs  = gridspec.GridSpec(5, 3, figure=fig,
    hspace=0.52, wspace=0.32,
    left=0.06, right=0.97, top=0.94, bottom=0.04)

fig.suptitle(
    'CDS Order-Chaos Spectrum  —  Alignment Polarity (AP)\n'
    '"乱れの方向への収束"  |  Pollock\'s Lévy fractal meets 24-cell geometry',
    color='white', fontsize=12, fontweight='bold', y=0.975
)

COLORS = {
    'Aligned\n(WCM/Material)': ['#4fc3f7','#ffd54f','#ef5350'],
    'Chaos\n(ECG/Crash)':      ['#81c784','#ffb74d','#f48fb1'],
    'Pollock\n(Lévy/Fractal)': ['#ce93d8','#ff8a65','#ffd54f'],
}

def ph_bg(ax, cphs):
    for t0,t1,c in [(0,80,cphs[0]),(80,140,cphs[1]),(140,162,cphs[2])]:
        ax.axvspan(t0,t1,alpha=0.07,color=c)
    ax.axvline(80, color='white',lw=0.8,ls='--',alpha=0.3)
    ax.axvline(140,color='white',lw=0.8,ls='--',alpha=0.3)

# ── Row 0: AP 時系列（メイン） ────────────────────────
for col,(name,(r,cphs)) in enumerate(
        zip(results_all.keys(),
            [(r, COLORS[k]) for k,r in results_all.items()])):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor('#0a0a1e')
    ph_bg(ax, cphs)
    r   = results_all[name]
    t   = r['t']
    AP  = r['AP']
    sm  = uniform_filter1d(AP, 7)

    # AP の符号で色を変える
    for i in range(len(t)-1):
        c_val = (sm[i]+0.3)/0.6   # -0.3〜+0.3 を 0〜1 に正規化
        clr   = cmap_ap(np.clip(c_val, 0, 1))
        ax.plot(t[i:i+2], sm[i:i+2], color=clr, lw=2.5, alpha=0.9)

    ax.axhline(0, color='white', lw=1.0, ls='-', alpha=0.4)
    ax.fill_between(t, sm, 0,
        where=(sm>0), alpha=0.20, color='#dc143c', label='Order↑')
    ax.fill_between(t, sm, 0,
        where=(sm<0), alpha=0.20, color='#4169e1', label='Chaos↓')

    tag = name.split('\n')[0]
    ap3 = r['AP_ph3']
    verdict = ('整列収束 (Order)' if ap3>0.01
               else '乱れ収束 (Chaos)' if ap3<-0.01
               else '臨界点 (Critical)')
    ax.set_title(f'{tag}\nAP(Ph3)={ap3:+.4f}  →  {verdict}',
                 color='white', fontsize=9, pad=4)
    ax.set_ylabel('Alignment Polarity (AP)', color='#aaa', fontsize=8)
    ax.set_xlabel('Time', color='#aaa', fontsize=8)
    ax.legend(fontsize=7, facecolor='#0a0a1e',
              labelcolor='white', edgecolor='#444')
    ax.tick_params(colors='#888', labelsize=7)
    [sp.set_color('#333') for sp in ax.spines.values()]
    ax.set_xlim(0, 162)

# ── Row 1: フラクタル次元プロキシ ────────────────────
for col, name in enumerate(results_all.keys()):
    ax  = fig.add_subplot(gs[1, col])
    ax.set_facecolor('#0a0a1e')
    cphs = COLORS[name]
    ph_bg(ax, cphs)
    r   = results_all[name]
    t   = r['t']
    fD  = r['fracD']
    smf = uniform_filter1d(fD, 7)
    ax.plot(t, smf, color=cphs[1], lw=2.2, alpha=0.9)
    ax.fill_between(t, smf, alpha=0.15, color=cphs[1])
    ax.axhline(r['fD_ph1'], color='white', lw=0.8, ls='--', alpha=0.4)
    ax.set_title(f'Fractal Proxy D₁\nPh1={r["fD_ph1"]:.3f}→Ph3={r["fD_ph3"]:.3f}',
                 color='white', fontsize=9, pad=4)
    ax.set_ylabel('D₁ = H/log(K)', color='#aaa', fontsize=8)
    ax.set_xlabel('Time', color='#aaa', fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.tick_params(colors='#888', labelsize=7)
    [sp.set_color('#333') for sp in ax.spines.values()]
    ax.set_xlim(0, 162)

# ── Row 2: AP × fracD 位相図（ポロック図） ───────────
for col, name in enumerate(results_all.keys()):
    ax  = fig.add_subplot(gs[2, col])
    ax.set_facecolor('#0a0a1e')
    r   = results_all[name]
    t   = r['t']
    AP  = r['AP']
    fD  = r['fracD']

    ph_colors = np.array([
        COLORS[name][0 if tt<80 else 1 if tt<140 else 2]
        for tt in t])
    for i, (ap, fd, clr) in enumerate(zip(AP, fD, ph_colors)):
        ax.scatter(fd, ap, color=clr, s=8, alpha=0.5)

    # 4象限の名前
    ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.4)
    ax.axvline(0.85, color='white', lw=0.8, ls='--', alpha=0.4)
    ax.text(0.72, 0.15, 'Order\n(WCM)',
            ha='center', color='#dc143c', fontsize=8, alpha=0.7)
    ax.text(0.72,-0.15, 'Chaos\n(Pollock)',
            ha='center', color='#4169e1', fontsize=8, alpha=0.7)
    ax.text(0.93, 0.02, 'Random',
            ha='center', color='white', fontsize=7, alpha=0.5)

    ax.set_title(f'AP × Fractal Phase Space\n(blue=Ph1, yellow=Ph2, red/pink=Ph3)',
                 color='white', fontsize=8.5, pad=4)
    ax.set_xlabel('Fractal Proxy D₁', color='#aaa', fontsize=8)
    ax.set_ylabel('Alignment Polarity AP', color='#aaa', fontsize=8)
    ax.tick_params(colors='#888', labelsize=7)
    [sp.set_color('#333') for sp in ax.spines.values()]

# ── Row 3: ポロックの「乱れ収束」の視覚化 ────────────
ax = fig.add_subplot(gs[3, :])
ax.set_facecolor('#080814')

# ポロックデータのイベントを3フェーズで散布
feat_pol, times_pol, ph_pol = make_pollock_data(rng, N=1500)
cphs_pol = ['#4fc3f7','#ffd54f','#ffd54f']
POLLOCK_COLORS = {1:'#4fc3f7', 2:'#ff8a65', 3:'#ffd54f'}

for phase in [1, 2, 3]:
    m   = (ph_pol == phase)
    sz  = 4 if phase < 3 else 8
    al  = 0.3 if phase < 3 else 0.7
    ax.scatter(feat_pol[m, 0] + (phase-1)*1.2,
               feat_pol[m, 1],
               c=POLLOCK_COLORS[phase], s=sz,
               alpha=al, edgecolors='none')

ax.text(0.5,  0.85, 'Phase 1: Normal',
        ha='center', transform=ax.transAxes,
        color=POLLOCK_COLORS[1], fontsize=9)
ax.text(1.7,  0.85, 'Phase 2: Precursor',
        ha='center', transform=ax.transAxes,
        color=POLLOCK_COLORS[2], fontsize=9)

# フラクタル的散乱
ax.text(1.7 + (feat_pol[ph_pol==3,0].mean()),
        0.9, 'Phase 3: Lévy Fractal\n(Pollock-type chaos)',
        ha='center', va='center', color=POLLOCK_COLORS[3],
        fontsize=10, fontweight='bold')

ax.set_xlim(-0.2, 3.3)
ax.set_title(
    '(D)  Spatial Distribution: Normal → Precursor → Pollock-type Lévy Fractal\n'
    '"The chaos has structure — just like a Pollock painting"',
    color='white', fontsize=10, pad=6)
ax.tick_params(colors='#888', labelsize=7)
[sp.set_color('#222') for sp in ax.spines.values()]

# ── Row 4: AP の概念図（「秩序←→臨界←→混沌」スペクトル）
ax = fig.add_subplot(gs[4, :])
ax.set_facecolor('#080814')
ax.axis('off')

# グラデーションバー
gradient = np.linspace(0, 1, 300).reshape(1, -1)
ax.imshow(gradient, extent=[0.05, 0.95, 0.55, 0.75],
          cmap=cmap_ap, aspect='auto',
          transform=ax.transAxes, alpha=0.9)

# ラベル
labels = [
    (0.05, 'AP ≪ 0\nPure Chaos\n(Brownian)'),
    (0.22, 'AP < 0\nLévy Fractal\n(Pollock D≈1.7)'),
    (0.50, 'AP ≈ 0\nCritical Point\n⚡ Phase transition'),
    (0.72, 'AP > 0\nOrder emerging\n(WCM precursor)'),
    (0.92, 'AP ≫ 0\nFull alignment\n(FormA)'),
]
for x, txt in labels:
    ax.text(x, 0.35, txt, ha='center', va='top',
            transform=ax.transAxes, color='white', fontsize=8.5,
            multialignment='center')

ax.text(0.5, 0.95, 'Alignment Polarity Spectrum  AP  ∈ (−∞, +∞)',
        ha='center', va='top', transform=ax.transAxes,
        color='white', fontsize=11, fontweight='bold')
ax.text(0.5, 0.05,
        '"Pollock painted in the region AP < 0  —  '
        'not random, but structured chaos.  '
        'The same geometry underlies ECG arrhythmia, '
        'market crashes, and material fatigue."',
        ha='center', va='bottom', transform=ax.transAxes,
        color='#aaa', fontsize=9, style='italic')

out = '/mnt/user-data/outputs/cds_order_chaos_v01.png'
plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(),
            bbox_inches='tight')
plt.close()
print(f"\n[可視化] {out}")

# ── テキストサマリー
print("\n" + "=" * 60)
print("【 秩序-混沌スペクトル：3データの比較 】")
print("=" * 60)
for name, r in results_all.items():
    tag = name.split('\n')[0]
    ap3 = r['AP_ph3']
    fD3 = r['fD_ph3']
    verdict = ('整列収束 →「1点への収束」', '#WCM型',
               f'AP={ap3:+.4f}') if ap3>0.01 else \
              ('乱れ収束 →「構造的カオス」','#ポロック型',
               f'AP={ap3:+.4f}') if ap3<-0.01 else \
              ('臨界状態 →「相転移の瞬間」','#最危険',
               f'AP={ap3:+.4f}')
    print(f"\n  {tag:20}: {verdict[0]}")
    print(f"    AP(Ph3) = {ap3:+.6f}   FD(Ph3) = {fD3:.4f}")
    print(f"    解釈: {verdict[1]}")
