"""図の文字(軸ラベル・題・凡例・注記)の日本語化。英語で描いた図の文字を、保存時に辞書で置き換えて `x_ja.png` も出す(figio.save_figure)。

- 文字列は英語のままコードに書く。置換は ①完全一致(短い語) ②部分文字列(長い句から順) ③正規表現(数値つき) の順。
- 辞書にない英語は、そのまま残る(未訳は `untranslated()` で確認できる)。
- 日本語フォントは IPAexGothic / Noto Sans CJK JP / Yu Gothic / Meiryo / MS Gothic のいずれか(WSL では /mnt/c/Windows/Fonts も探す)。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# ① 完全一致(凡例などの短い語)
EXACT = {
    "thigh": "大腿", "shank": "下腿", "foot": "足部", "toe": "つま先", "ankle": "足首", "knee": "膝", "hip": "股関節", "ball": "ボール",
    "big_toe": "つま先", "observed": "実測", "m": "m", "EI": "EI", "Ball speed": "ボール速度",
}

# ② 部分文字列(長い句を先に置換する。順序は長さ降順に自動で並べる)
PHRASES = {
    "CE: constant error (signed)": "CE: 定誤差 (符号つき)", "VE: variable error (SD)": "VE: 変動誤差 (SD)", "TE: total error (RMSE)": "TE: 総合誤差 (RMSE)",
    "Horizontal y (left-right)": "左右 y", "Vertical z (up-down)": "上下 z",
    ": scores (bars) with 95% interval": " : スコア (棒) と95%区間",
    ": trials (red = mean, gray = mean ± VE)": " : 試行ごと (赤 = 平均、灰 = 平均 ± VE)",
    "Error of each trial (m)": "試行ごとのずれ (m)",
    "[left-footed subjects mirrored]": "[左足の被験者は左右反転]",
    # accuracy_score: タスク精度スコア
    "Accuracy score per subject (11 m)": "被験者ごとの精度スコア (距離11 m)",
    "Angle seen from the ball (deg)": "ボールから見た角度 (deg)",
    "CE: constant error (bias)": "CE: 定誤差 (狙いの偏り)", "VE: variable error (spread)": "VE: 変動誤差 (ばらつき)", "TE: total error": "TE: 総合誤差",
    "Error at the goal plane (m)": "ゴール面での誤差 (m)",
    "Errors from the target (circle: radius = VE)": "目標からのずれ (円の半径 = VE)",
    "Horizontal error, kicking-leg side + (m)": "左右のずれ、蹴り足側が + (m)",
    "Vertical error (m)": "上下のずれ (m)",
    "Radial error per trial (m)": "試行ごとの半径誤差 (m)",
    "Trial-level score (red = median)": "試行ごとのスコア (赤 = 中央値)",
    "95% interval by source of uncertainty": "不確かさの源ごとの95%区間",
    "trials only (resampling)": "試行の標本誤差だけ (引き直し)",
    "reading error only (SD 10 cm)": "読み取り誤差だけ (SD 10 cm)",
    "trials + reading error (SD 10 cm)": "試行の標本誤差 + 読み取り誤差 (SD 10 cm)",
    "How precisely must each input be measured?": "各入力をどの精度で測る必要があるか",
    "Required measurement SD (deg or mm)": "必要な測定のSD (deg または mm)",
    "foot velocity azimuth": "足先速度の方位", "foot velocity elevation": "足先速度の仰角",
    "contact offset, lateral": "接触点のずれ(左右)", "contact offset, vertical": "接触点のずれ(上下)",
    "How many kicks to compare two subjects' VE?": "2人のVEを比べるのに何本要るか",
    "Power to detect a VE difference between two subjects": "2人のVEの差を検出できる確率",
    "Trials per subject": "被験者あたりの試行数",
    "VE ratio": "VEの比", "isotropic": "等方", "elongated": "細長い",
    # trialdep_figures
    "Slope 0 = unrelated to the previous kick / -1 = reverses the previous error completely / + = stays near the previous one":
        "傾き 0 = 前の1本と無関係 / −1 = 前のずれを完全に反対へ戻す / 正 = 前と近い所へ行く",
    "Landing points joined in trial order (number = trial, dark = early, bright = late, x = target)":
        "着弾を試行順に矢印でつないだ図 (数字 = 試行番号、暗い = 早い、明るい = 遅い、× = 目標)",
    "Error of this kick (m, mean removed)": "今回のずれ (m, 平均を引いた)",
    "Error of the next kick (m, mean removed)": "次回のずれ (m, 平均を引いた)",
    "slope 0: unrelated": "傾き0: 無関係", "slope -1: full reversal": "傾き−1: 完全に反対へ戻す",
    "all subjects: slope": "全員: 傾き",
    "(shuffled order: 95% in": "(試行順をシャッフルした95%範囲:",
    "slope": "傾き",
    "Lag-1 autocorrelation (dot = observed, gray = 95% range of shuffled order)": "ラグ1自己相関 (点 = 実測、灰色 = 試行順シャッフルの95%範囲)",
    "linear trend removed": "直線の傾向を除いた", "original": "元の着弾",
    "Which knob carries the correction? (on the diagonal = chance)": "どのつまみが補正を担うか? (対角線上 = 偶然の範囲)",
    "Expected p-value if only chance": "偶然だけの場合に期待される p 値",
    "paths are correlated: the band is a guide": "経路は互いに相関するので、帯は目安",
    "95% range of shuffled order": "試行順シャッフルの95%範囲",
    "Observed p-value, indirect effect via knob": "観測された p 値 (つまみ経由の間接効果)",
    "95% range if chance": "偶然なら95%がこの範囲", "paths (knob x error pair)": "個の経路 (つまみ × 誤差の組)",
    "smallest corrected p": "補正後の最小の p",
    "; linear trend removed": ";  直線の傾向を除いた残差",
    # mediation: 前の誤差 → つまみ → 次の誤差
    "indirect effect via knob (* p<0.1)": "つまみ経由の間接効果 (* p<0.1)",
    "knob(n+1) response to previous error (* p<0.1)": "前の誤差に対する次のつまみの応答 (* p<0.1)",
    "error y(n)": "誤差 y(n)", "error z(n)": "誤差 z(n)",
    # sequential: 試行間の依存
    "Correction matrix K: next error = K x current error (* = outside 95% range of shuffled order)": "補正係数行列 K: 次の誤差 = K × 今の誤差 (* = 試行順シャッフルの95%範囲の外)",
    "Dot = observed, gray bar = 95% range after shuffling trial order (red dot = outside)": "点 = 実測、灰色の棒 = 試行順をシャッフルしたときの95%範囲 (赤い点 = 範囲の外)",
    "Lag-1 autocorrelation": "ラグ1自己相関",
    "Horizontal y (kicking-leg side +)": "左右 y (蹴り足側 +)",
    "Ellipse major axis": "楕円の長軸方向", "Ellipse minor axis": "楕円の短軸方向",
    "Vertical z": "上下 z", "Trial number": "試行番号", "(unstable)": "(不安定)",
    # analyze: 軸・題
    "Time from impact (s)": "インパクトからの時間 (s)",
    "Time of peak angular velocity from impact (ms)": "角速度ピークの時刻(インパクト基準, ms)",
    "Momentum (kg m/s)": "運動量 (kg m/s)",
    "Angular velocity (rad/s)": "角速度 (rad/s)",
    "Segment energy (J)": "セグメントエネルギー (J)",
    "Effective mass of kicking foot (kg)": "蹴り足の有効質量 (kg)",
    "Effective mass (kg)": "有効質量 (kg)",
    "Speed (m/s)": "速さ (m/s)",
    "Angle (deg)": "角度 (deg)",
    "Ball speed (km/h)": "ボール速度 (km/h)",
    "Ball speed (m/s)": "ボール速度 (m/s)",
    "Ball speed / toe speed": "ボール速度 / つま先速度",
    "Toe speed at impact (m/s)": "インパクト時のつま先速度 (m/s)",
    "(A) Whole-body momentum (forward)": "(A) 全身の運動量(前方)",
    "(B) Kicking-leg shank angular velocity": "(B) 蹴り脚の下腿の角速度",
    "(A) Kicking-foot toe speed": "(A) 蹴り足のつま先の速さ",
    "(B) Kicking-leg knee flexion": "(B) 蹴り脚の膝屈曲",
    "(C) Segment angular velocities (mean)": "(C) セグメントの角速度(平均)",
    "(D) Ball speed per trial": "(D) 試行ごとのボール速度",
    "(A) Kicking-foot effective mass": "(A) 蹴り足の有効質量",
    "(B) At impact": "(B) インパクト時",
    "(C) Mean of last 10 ms": "(C) 直前10msの平均",
    "(C) Proximal-to-distal sequence": "(C) 近位から遠位への順序",
    "(D) Ball vs toe speed (dotted: ratio)": "(D) ボール速度とつま先速度(点線: 比)",
    "(E) Collision transfer ratio": "(E) 衝突の伝達比",
    "(F) EI vs ball speed": "(F) EI とボール速度",
    "thin = trials, thick = mean": "細線 = 各試行、太線 = 平均",
    "segment kinetic energy (translation + rotation)": "セグメントの運動エネルギー(並進 + 回転)",
    "effective mass (planar 3-link leg, Khatib)": "有効質量(平面3リンクの脚、Khatib)",
    "Subject comparison: mean ± SD over trials": "被験者間の比較: 試行の平均 ± SD",
    "Segment kinetic energy: subject means": "セグメントの運動エネルギー: 被験者の平均",
    "Effective energy $E_b$": "有効エネルギー $E_b$",
    "Mechanical work $W_p$": "力学的仕事 $W_p$",
    "Effectiveness index EI": "有効性指標 EI",
    "v: ball speed 0.01-0.02 s after impact, m = 0.43 kg": "v: インパクト後0.01〜0.02秒のボール速度、m = 0.43 kg",
    "kicking-leg sum of |dE| over the 0.1 s before impact": "インパクト前0.1秒の蹴り脚の|dE|の合計",
    "Kicking leg torso": "蹴り脚側の体幹", "Kicking leg thigh": "蹴り脚の大腿", "Kicking leg shank": "蹴り脚の下腿",
    "Support leg torso": "支持脚側の体幹", "Support leg thigh": "支持脚の大腿", "Support leg shank": "支持脚の下腿",
    "mean (": "平均 (",
    # render.plot_summary
    "segment angular velocity [deg/s]": "セグメント角速度 [deg/s]",
    "speed [m/s]": "速さ [m/s]",
    "knee flexion [deg]": "膝屈曲 [deg]",
    "height y [m]": "高さ y [m]",
    "time from impact [s]": "インパクトからの時間 [s]",
    "time [s]": "時間 [s]",
    "ball=n/a": "ボール=なし",
    "ball=": "ボール=",
    "fc=": "遮断周波数=",
    # imu3d check.png
    "IMU(Xsens) + 2D video 3D lower-body": "IMU(Xsens) + 2D映像による下半身3D運動学",
    "hip angle [deg]": "股関節角度 [deg]", "knee angle [deg]": "膝関節角度 [deg]", "ankle angle [deg]": "足関節角度 [deg]",
    "IMU-FK x": "IMU順運動学 x", "IMU-FK z": "IMU順運動学 z",
    "KNEE (": "膝 (", "ANKLE (": "足首 (", "MP (": "つま先 (",
    # landing
    "Horizontal y, kicking-leg side + (m)": "水平 y、蹴り足側が + (m)",
    "Horizontal y (m)": "水平 y (m)",
    "Vertical z (m)": "鉛直 z (m)",
    "95% ellipse (tilt in this plot ": "95%楕円 (この図での傾き ",
    "left-footed": "左足", "right-footed": "右足",
    # kickfactors
    "COM - ball x (m)": "重心 − ボール x (m)",
    "support ankle - ball x (m)": "支持足首 − ボール x (m)",
    "trunk lean fwd (deg)": "体幹前傾 (deg)",
    "foot pitch (deg)": "足部ピッチ (deg)",
    "toe velocity elevation (deg)": "足先速度の仰角 (deg)",
    "swing-arc phase (ms after lowest)": "スイング円弧の位相 (最下点からの ms)",
    "foot mid - ball center y (m)": "足の中点 − ボール中心 y (m)",
    "ball launch elevation 2D (deg)": "ボール出射仰角 2D (deg)",
    "landing y, kick side + (m)": "着弾 y、蹴り足側が + (m)",
    "landing z (rank-normal, speed peaks regressed out)": "着弾 z (順位正規化、速度ピークの影響を除去)",
    "landing z (m)": "着弾 z (m)",
    ": impact-time factors vs landing": ": インパクト時の要因と着弾",
    "swing-plane model (tilt ": "スイング面モデル (傾き ",
    "rho=": "ρ=",
    # contact_sensitivity
    "effect of +1 typical SD on ": "典型的な1 SD 増加の影響: ",
    "vertical z at the goal (m)": "ゴールでの上下 z (m)",
    "lateral y, kick side + (m)": "左右 y、蹴り足側が + (m)",
    "foot velocity elevation": "足先速度の仰角",
    "foot velocity azimuth (kick side)": "足先速度の方位(蹴り足側)",
    "contact offset, vertical": "接触点のずれ: 上下",
    "contact offset, lateral (kick side)": "接触点のずれ: 左右(蹴り足側)",
    "foot speed": "足先速度の大きさ",
    "restitution e": "反発係数 e",
    "friction mu": "摩擦係数 μ",
    "effective mass": "有効質量",
    "Contact-model sensitivity (baseline aimed at the target by contact offset ": "接触モデルの感度(基準: 目標に着弾するよう接触点のずれを ",
    " cm: foot ": " cm に調整、足先 ",
    ", elev ": "、仰角 ",
    "; ball ": "; ボール ",
    "spin ": "スピン ",
    "contact-offset axis ": "接触点のずれの主軸 ",
    "share of landing variance from foot-velocity direction (%)": "着弾の分散のうち足先速度の向きが占める割合 (%)",
    "tilt of landing ellipse (deg)": "着弾楕円の傾き (deg)",
    "Ellipse tilt from two sources (swing-plane tilt ": "2つの要因による楕円の傾き (スイング面の傾き ",
    ") vs observed (dashed)": ") と実測(破線)",
    # imu3d animation
    "Side view (camera)": "側面(カメラ視点)",
    "Top view": "上面",
    "3D view": "3D(斜め)",
    "kicking leg": "蹴り脚",
    "support leg": "支持脚",
    "time from impact": "インパクトからの時間",
    "slow motion": "スロー再生",
    "IMPACT": "インパクト",
    # chaincheck
    "thigh peak - shank peak (ms; negative = thigh first)": "大腿ピーク − 下腿ピーク (ms; 負 = 大腿が先)",
    "lag (rank-normal, speed peaks regressed out)": "時間差 (順位正規化、速度ピークの影響を除去)",
    "partial relation within subject": "被験者内の偏関係",
}

# ③ 正規表現(数値つきの文字列)
REGEX = [
    (re.compile(r"landing prediction error at most ([\d.]+) m"), r"着弾の予測誤差 \g<1> m 以下"),
    (re.compile(r"(-?\d+(?:\.\d+)?) deg\b"), r"\1°"),
    (re.compile(r"\+/-"), "±"),
    (re.compile(r"\((\d+)y\)"), r"(\1年)"),
    (re.compile(r", (\d+(?:\.\d+)?) y\)"), r", \1年)"),
    (re.compile(r"\btilt\b"), "傾き"),
    (re.compile(r"phi="), "φ="),
    (re.compile(r"\bflex\b"), "屈曲"), (re.compile(r"\babd\b"), "外転"), (re.compile(r"\brot\b"), "回旋"),
    (re.compile(r"\bmu=") , "μ="),
]

_ORDERED = sorted(PHRASES.items(), key=lambda kv: -len(kv[0]))
_font_name: str | None = None
_font_checked = False
_seen_untranslated: set[str] = set()


def translate(text: str) -> str:
    if text in EXACT:
        return EXACT[text]
    out = text
    for en, ja in _ORDERED:
        if en in out:
            out = out.replace(en, ja)
    for rx, rep in REGEX:
        out = rx.sub(rep, out)
    if out == text and re.search(r"[A-Za-z]{3,}[ \-][A-Za-z]{2,}", text):  # 英語の句が残ったとき(被験者名や試行名は対象外)
        _seen_untranslated.add(text)
    return out


def untranslated() -> list[str]:
    """英語が残った文字列(辞書にない)。図の追加時に確認して PHRASES に足す。"""
    return sorted(_seen_untranslated)


def japanese_font() -> str | None:
    """使える日本語フォント名(なければ None)。"""
    global _font_name, _font_checked
    if _font_checked:
        return _font_name
    _font_checked = True
    from matplotlib import font_manager as fm

    for d in ("/mnt/c/Windows/Fonts", "C:/Windows/Fonts"):  # WSL・Windows に入っている日本語フォントを登録
        for f in ("meiryo.ttc", "YuGothR.ttc", "msgothic.ttc", "NotoSansJP-VF.ttf"):
            p = Path(d) / f
            if p.exists():
                try:
                    fm.fontManager.addfont(str(p))
                except Exception:  # noqa: BLE001
                    pass
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "Noto Sans JP", "Yu Gothic", "Meiryo", "MS Gothic", "TakaoGothic"):
        if name in have:
            _font_name = name
            break
    return _font_name


def enabled() -> bool:
    return os.environ.get("POSE2D_NO_JA", "") not in ("1", "true", "True")


def to_japanese(fig) -> bool:
    """図の文字を日本語にする(その場で書き換える。保存後は使わないこと)。フォントがなければ False。"""
    from matplotlib.text import Text

    font = japanese_font()
    if font is None:
        return False
    for t in fig.findobj(Text):
        s = t.get_text()
        if s:
            t.set_text(translate(s))
        t.set_fontfamily(font)
    for ax in fig.axes:  # 文字の目盛り(大腿・下腿など)は保存時に作り直されるので、固定の目盛りとして設定し直す
        for axis in (ax.xaxis, ax.yaxis):
            labs = [t.get_text() for t in axis.get_ticklabels()]
            new = [translate(s) if re.search(r"[A-Za-z]{2,}", s) and "$" not in s else s for s in labs]
            if new != labs:
                axis.set_ticks(axis.get_ticklocs())
                axis.set_ticklabels(new)
    return True
