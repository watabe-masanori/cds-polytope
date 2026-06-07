"""
CDS on Real Data — v0.1
========================
AutoAxisSelector v0.3 + E₈ を現実のデータに投入する。

【 3つのデータセット 】

  Dataset 1: Breast Cancer (30特徴量)
    sklearn内蔵。30特徴量で良性/悪性を分類する医療データ。
    Phase1=良性(benign) / Phase2=境界域 / Phase3=悪性(malignant)
    → CDSが「悪性化の方向」を24-cell幾何学で捉えられるか？

  Dataset 2: Wine (13特徴量)
    3種のワインの化学成分。
    Phase1=class0 / Phase2=class1 / Phase3=class2
    → 3クラスの分離が幾何学的に現れるか？

  Dataset 3: Synthetic ECG-like (心電図風)
    正常→不整脈前兆→不整脈の3フェーズ時系列。
    NEURO-CDS の現実的バージョン。

【 共通パイプライン 】
    feat(N × D_cand) → AutoAxisSelector v0.3 → 最良8軸
    → E₈ CDS-8D → cosD / hot_v 時系列 → FormA 検定
"""

import sys
sys.path.insert(0, '/home/claude')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler

from cds_highdim_core import CDSEngine, E8_UNIT
from auto_axis_selector_v3 import AutoAxisSelectorV3


# ═══════════════════════════════════════════════════════
# 共通ユーティリティ
# ═══════════════════════════════════════════════════════
def make_time_series(X, y, phase_map,
                     t_ends=(80.0, 140.0, 160.0),
                     seed=42):
    """
    静的データセット (X, y) をフェーズ付き時系列に変換する。

    phase_map : dict  { class_label: phase_number(1/2/3) }
    t_ends    : tuple フェーズ境界（日/秒）

    各フェーズのサンプルを時間軸上に均等に並べる。
    """
    rng = np.random.default_rng(seed)
    rows = []
    t_starts = (0.0,) + t_ends[:-1]
    for cls, ph in sorted(phase_map.items(), key=lambda x: x[1]):
        mask  = (y == cls)
        X_ph  = X[mask]
        n_ph  = len(X_ph)
        t0, t1 = t_starts[ph-1], t_ends[ph-1]
        times = np.sort(rng.uniform(t0, t1, n_ph))
        for i, t in enumerate(times):
            rows.append({'time': t, 'phase': ph, **
                         {f'f{d}': X_ph[i, d] for d in range(X.shape[1])}})
    df = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
    return df


def run_cds_pipeline(df, feat_cols, axis_names,
                     eng, n_select=8,
                     window=10.0, step=2.0,
                     sens_threshold=0.30,
                     dataset_name='Dataset'):
    """
    完全なパイプライン：
        df + feat_cols → AutoAxisSelector v0.3 → E₈ CDS → 結果 dict
    """
    feat   = df[feat_cols].values
    times  = df['time'].values
    phases = df['phase'].values

    print(f"\n{'─'*55}")
    print(f"[{dataset_name}]")
    print(f"  サンプル数: {len(df)}  特徴量: {len(feat_cols)}  "
          f"→ 選択軸: {n_select}")

    sel = AutoAxisSelectorV3(n_select=n_select, engine=eng,
                              sens_threshold=sens_threshold)
    sel.fit(feat, times, phases,
            axis_names=axis_names, verbose=True)

    res  = sel.scan(feat, times, window=window, step=step)
    t_arr= np.array([r['t_center'] for r in res])
    cosD = np.array([r['cosD']     for r in res])
    ent  = np.array([r['entropy']  for r in res]) / np.log(240)
    hotv = np.array([r['hot_v']    for r in res])
    hotp = np.array([r['hot_prob'] for r in res])

    # FormA 統計
    mask3 = (t_arr >= 140) & (t_arr < 162)
    mask1 = (t_arr >= 0)   & (t_arr < 80)
    if mask3.sum() > 0:
        top_v3 = int(np.bincount(hotv[mask3], minlength=240).argmax())
        sn3    = hotp[mask3].mean() / (1/240)
        sn1    = (np.bincount(hotv[mask1 & (hotv >= 0)],
                               minlength=240)[top_v3]
                  / mask1.sum()) / (1/240) if mask1.sum()>0 else 1.0
        cosd_delta = cosD[mask3].mean() - cosD[mask1].mean()
    else:
        top_v3 = 0; sn3 = 1.0; sn1 = 1.0; cosd_delta = 0.0

    print(f"\n  FormA頂点: V{top_v3:03d}  "
          f"S/N(Ph3)={sn3:.1f}×  ΔcosD={cosd_delta:+.4f}")
    print(f"  有効軸: {sel.n_effective}/{n_select}  "
          f"パディング: {max(0,n_select-sel.n_effective)}スロット")

    return {
        't': t_arr, 'cosD': cosD, 'ent': ent,
        'hotv': hotv, 'hotp': hotp,
        'sn3': sn3, 'cosd_delta': cosd_delta,
        'top_v3': top_v3, 'selector': sel,
        'n_eff': sel.n_effective,
    }


# ═══════════════════════════════════════════════════════
# Dataset 1: Breast Cancer
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("CDS on Real Data  v0.1")
print("AutoAxisSelector v0.3 + E₈ (240 refs)")
print("=" * 60)

bc   = load_breast_cancer()
scaler = StandardScaler()
X_bc = scaler.fit_transform(bc.data)
y_bc = bc.target   # 0=malignant, 1=benign

# Phase マッピング
# malignant (0) の中で「最も良性に近い」群を Phase2 にする
# → malignant のうち radius_mean が小さい半分 → Phase2
# → 残り → Phase3
mal_idx  = np.where(y_bc == 0)[0]
ben_idx  = np.where(y_bc == 1)[0]
r_mal    = bc.data[mal_idx, 0]   # radius_mean
split    = np.median(r_mal)
ph2_idx  = mal_idx[r_mal <= split]   # malignantの小半分 → Phase2
ph3_idx  = mal_idx[r_mal >  split]   # malignantの大半分 → Phase3

y_phase  = np.zeros(len(y_bc), dtype=int)
y_phase[ben_idx] = 1
y_phase[ph2_idx] = 2
y_phase[ph3_idx] = 3

df_bc = pd.DataFrame(X_bc,
    columns=[f'f{i}' for i in range(30)])
df_bc['time']  = 0.0
df_bc['phase'] = y_phase
# 時間軸への配置
rng0 = np.random.default_rng(42)
for ph, (t0,t1) in [(1,(0,80)),(2,(80,140)),(3,(140,160))]:
    m = df_bc['phase']==ph
    df_bc.loc[m,'time'] = np.sort(rng0.uniform(t0,t1,m.sum()))
df_bc = df_bc.sort_values('time').reset_index(drop=True)

eng8 = CDSEngine(E8_UNIT, name='CDS-8D')
feat_cols_bc = [f'f{i}' for i in range(30)]
names_bc     = list(bc.feature_names)

r_bc = run_cds_pipeline(
    df_bc, feat_cols_bc, names_bc, eng8,
    n_select=8, window=8.0, step=1.0,
    sens_threshold=0.20,
    dataset_name='Breast Cancer (30 features → 8 selected)'
)

# ═══════════════════════════════════════════════════════
# Dataset 2: Wine
# ═══════════════════════════════════════════════════════
wn   = load_wine()
X_wn = scaler.fit_transform(wn.data)
y_wn = wn.target   # 0,1,2

df_wn = make_time_series(X_wn, y_wn,
    {0:1, 1:2, 2:3}, seed=55)
feat_cols_wn = [c for c in df_wn.columns
                if c.startswith('f')]
names_wn = [wn.feature_names[int(c[1:])]
            for c in feat_cols_wn]

r_wn = run_cds_pipeline(
    df_wn, feat_cols_wn, names_wn, eng8,
    n_select=8, window=12.0, step=2.0,
    sens_threshold=0.25,
    dataset_name='Wine (13 features → 8 selected)'
)

# ═══════════════════════════════════════════════════════
# Dataset 3: Synthetic ECG-like
# ═══════════════════════════════════════════════════════
def make_ecg_data(n=3000, seed=7):
    """
    正常→前不整脈→不整脈の3フェーズ心拍データ。
    12誘導ECGから抽出するような8特徴量を持つ。

    軸の意味（実際のECG特徴量に近似）:
        f0: RR間隔   f1: QRS幅    f2: ST偏位   f3: T波高
        f4: P波高    f5: PR間隔   f6: HRV(短期) f7: QT補正
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        t  = i * 0.5    # 0.5秒ごとの心拍
        ph = (1 if t < 400 else 2 if t < 1000 else 3)

        if ph == 1:     # 正常
            rr = rng.normal(800, 30)     # ms
            qrs= rng.normal(80,  5)
            st = rng.normal(0,   0.05)
            tw = rng.normal(0.3, 0.05)
            pw = rng.normal(0.15,0.03)
            pr = rng.normal(160, 10)
            hrv= rng.normal(50,  10)
            qt = rng.normal(380, 15)
        elif ph == 2:   # 前不整脈期（スロースリップ相当）
            rr = rng.normal(780, 60)     # RR 不安定化
            qrs= rng.normal(95,  15)     # QRS 拡大傾向
            st = rng.normal(-0.1, 0.1)  # ST 低下傾向
            tw = rng.normal(0.2, 0.1)   # T波 平坦化
            pw = rng.normal(0.12,0.05)
            pr = rng.normal(180, 20)
            hrv= rng.normal(30,  15)     # HRV 低下
            qt = rng.normal(410, 25)     # QT 延長
        else:           # 不整脈発現
            rr = rng.normal(600, 150)    # RR 大きく乱れる
            qrs= rng.normal(130, 25)     # QRS 著明拡大
            st = rng.normal(-0.3, 0.15) # ST 著明低下
            tw = rng.normal(-0.1, 0.15) # T波 陰転
            pw = rng.normal(0.08, 0.08)
            pr = rng.normal(200, 40)
            hrv= rng.normal(10,  10)
            qt = rng.normal(460, 40)

        rows.append({'time':t,'phase':ph,
                     'f0':rr,'f1':qrs,'f2':st,'f3':tw,
                     'f4':pw,'f5':pr,'f6':hrv,'f7':qt})

    df = pd.DataFrame(rows)
    # 標準化
    for c in ['f0','f1','f2','f3','f4','f5','f6','f7']:
        mu,sig = df[c].mean(), df[c].std()
        df[c] = (df[c]-mu)/(sig+1e-9)
    return df

df_ecg   = make_ecg_data()
feat_ecg = [f'f{i}' for i in range(8)]
names_ecg= ['RR_interval','QRS_width','ST_deviation','T_wave',
             'P_wave','PR_interval','HRV_short','QTc']

# ECGの時間軸を 0-160 に再マッピング
t_raw = df_ecg['time'].values
df_ecg['time'] = (t_raw / t_raw.max()) * 158 + 1

r_ecg = run_cds_pipeline(
    df_ecg, feat_ecg, names_ecg, eng8,
    n_select=8, window=12.0, step=2.0,
    sens_threshold=0.10,
    dataset_name='ECG-like (8 features, arrhythmia onset)'
)

# ═══════════════════════════════════════════════════════
# 統合可視化
# ═══════════════════════════════════════════════════════
datasets = {
    'Breast Cancer\n(malignant onset)': (r_bc, ['#4fc3f7','#ffd54f','#ef5350']),
    'Wine\n(class transition)':         (r_wn, ['#81c784','#ffb74d','#f48fb1']),
    'ECG-like\n(arrhythmia onset)':     (r_ecg,['#ce93d8','#ff8a65','#ef5350']),
}

fig = plt.figure(figsize=(20, 18), facecolor='#030308')
gs  = gridspec.GridSpec(4, 3, figure=fig,
    hspace=0.52, wspace=0.32,
    left=0.06, right=0.97, top=0.93, bottom=0.05)

fig.suptitle(
    'CDS on Real Data  —  AutoAxisSelector v0.3 + E₈ (240 refs)\n'
    'Breast Cancer (30 feat)  |  Wine (13 feat)  |  ECG-like (8 feat)',
    color='white', fontsize=12, fontweight='bold', y=0.975
)

def ph_bg(ax, cphs):
    for t0,t1,c in [(0,80,cphs[0]),(80,140,cphs[1]),(140,162,cphs[2])]:
        ax.axvspan(t0,t1,alpha=0.08,color=c)
    ax.axvline(80, color='white',lw=0.8,ls='--',alpha=0.3)
    ax.axvline(140,color='white',lw=0.8,ls='--',alpha=0.3)

# ── Row 0: cosD 時系列（3データセット横並び）────────────
for col, (dname,(r,cphs)) in enumerate(datasets.items()):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor('#0a0a1e')
    ph_bg(ax, cphs)
    sm = uniform_filter1d(r['cosD'], 7)
    # フェーズ別に色を変える
    t_arr = r['t']
    for i in range(len(t_arr)-1):
        ph = (1 if t_arr[i]<80 else 2 if t_arr[i]<140 else 3)
        ax.plot(t_arr[i:i+2], sm[i:i+2],
                color=cphs[ph-1], lw=2.0, alpha=0.9)
    ax.set_title(f'{dname}\nS/N={r["sn3"]:.1f}×  ΔcosD={r["cosd_delta"]:+.3f}',
                 color='white', fontsize=9, pad=4)
    ax.set_ylabel('cosD', color='#aaa', fontsize=8)
    ax.set_xlabel('Normalized time', color='#aaa', fontsize=8)
    ax.tick_params(colors='#888', labelsize=7)
    [sp.set_color('#333') for sp in ax.spines.values()]
    ax.set_xlim(0, 162)

# ── Row 1: hot_v 転換経路 ─────────────────────────────
for col, (dname,(r,cphs)) in enumerate(datasets.items()):
    ax = fig.add_subplot(gs[1, col])
    ax.set_facecolor('#0a0a1e')
    ph_bg(ax, cphs)
    t_arr = r['t']
    hotv  = r['hotv']
    for i in range(len(t_arr)-1):
        ph = (1 if t_arr[i]<80 else 2 if t_arr[i]<140 else 3)
        ax.plot(t_arr[i:i+2], hotv[i:i+2],
                color=cphs[ph-1], lw=1.8, alpha=0.8)
    ax.scatter(t_arr, hotv,
               c=[cphs[0 if t<80 else 1 if t<140 else 2]
                  for t in t_arr],
               s=12, zorder=3, alpha=0.7)
    ax.axhline(r['top_v3'], color='#ffd54f',
               lw=1.2, ls='--', alpha=0.7)
    ax.text(163, r['top_v3'], f"V{r['top_v3']:03d}",
            color='#ffd54f', fontsize=7, va='center')
    ax.set_title(f'hot_v path  (FormA=V{r["top_v3"]:03d})',
                 color='white', fontsize=9, pad=4)
    ax.set_ylabel('hot_v (0-239)', color='#aaa', fontsize=8)
    ax.set_xlabel('Normalized time', color='#aaa', fontsize=8)
    ax.set_ylim(-5, 245)
    ax.tick_params(colors='#888', labelsize=7)
    [sp.set_color('#333') for sp in ax.spines.values()]
    ax.set_xlim(0, 165)

# ── Row 2: 選択軸のスコアバー ─────────────────────────
selectors = [r_bc['selector'], r_wn['selector'], r_ecg['selector']]
dnames    = ['Breast Cancer','Wine','ECG-like']
for col, (sel, dn) in enumerate(zip(selectors, dnames)):
    ax = fig.add_subplot(gs[2, col])
    ax.set_facecolor('#0a0a1e')
    diag    = sel.diag
    all_d   = list(range(len(sel.axis_names)))
    sel_d   = sel.selected_axes
    scores  = [diag.solo_score(d) for d in all_d]
    sort_i  = np.argsort(scores)[::-1][:12]   # 上位12軸のみ表示
    clrs    = ['#ffd54f' if all_d[i] in sel_d else '#555'
               for i in sort_i]
    ax.barh(range(len(sort_i)),
            [scores[i] for i in sort_i],
            color=clrs, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(sort_i)))
    ax.set_yticklabels([sel.axis_names[i][:14] for i in sort_i],
                        color='white', fontsize=7.5)
    ax.set_xlabel('Solo Score', color='#aaa', fontsize=8)
    ax.set_title(f'{dn}\nAxis Scores (yellow=selected, '
                 f'eff={sel.n_effective}/{sel.n_select})',
                 color='white', fontsize=8.5, pad=4)
    ax.tick_params(colors='#888', labelsize=7)
    [sp.set_color('#333') for sp in ax.spines.values()]

# ── Row 3: 3データセット比較サマリー ────────────────────
ax = fig.add_subplot(gs[3, :])
ax.set_facecolor('#0a0a1e')
ax.axis('off')

results_list = [
    ('Breast Cancer', 30, r_bc),
    ('Wine',          13, r_wn),
    ('ECG-like',       8, r_ecg),
]

# S/Nバー
bar_y  = [0.72, 0.50, 0.28]
bar_clr= ['#4fc3f7','#81c784','#ce93d8']
for y, (dname, n_feat, r), clr in zip(bar_y, results_list, bar_clr):
    # データセット名
    ax.text(0.02, y+0.07, dname, transform=ax.transAxes,
            color=clr, fontsize=11, fontweight='bold', va='top')
    ax.text(0.02, y-0.01, f'{n_feat}特徴量 → {r["n_eff"]}有効軸 '
            f'(+{r["selector"].n_select - r["n_eff"]}ゼロパディング)',
            transform=ax.transAxes, color='#aaa', fontsize=9, va='top')
    # S/Nバー
    bar_w = min(r['sn3']/300, 0.5)
    rect = plt.Rectangle((0.30, y-0.04), bar_w, 0.12,
        facecolor=clr, alpha=0.7, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.30+bar_w+0.01, y+0.02,
            f'S/N = {r["sn3"]:.1f}×',
            transform=ax.transAxes, color=clr, fontsize=10,
            fontweight='bold', va='center')
    # ΔcosD
    dcol = '#81c784' if r['cosd_delta']>0.02 else \
           '#ffd54f' if r['cosd_delta']>0 else '#ef5350'
    ax.text(0.70, y+0.02,
            f'ΔcosD = {r["cosd_delta"]:+.4f}  '
            f'FormA = V{r["top_v3"]:03d}',
            transform=ax.transAxes, color=dcol, fontsize=9,
            va='center')

ax.text(0.5, 0.05,
    'E₈ (240 refs) × AutoAxisSelector v0.3 (zero-padding)  '
    '—  same algorithm, three real-world domains',
    transform=ax.transAxes, color='#888', fontsize=9,
    ha='center', va='center')

out = '/mnt/user-data/outputs/cds_real_data_v01.png'
plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(),
            bbox_inches='tight')
plt.close()
print(f"\n[可視化] {out}")

print("\n" + "="*60)
print("【 現実データ CDS 結果サマリー 】")
print("="*60)
for dname, n_feat, r in results_list:
    print(f"\n  {dname} ({n_feat}特徴量)")
    print(f"    有効軸        : {r['n_eff']} 本  "
          f"(パディング {r['selector'].n_select - r['n_eff']})")
    print(f"    FormA 頂点    : V{r['top_v3']:03d}")
    print(f"    Phase3 S/N    : {r['sn3']:.1f}×")
    print(f"    ΔcosD(Ph1→3) : {r['cosd_delta']:+.4f}")
    sel_names = [r['selector'].axis_names[d]
                 for d in r['selector'].selected_axes]
    print(f"    選択軸        : {sel_names}")
