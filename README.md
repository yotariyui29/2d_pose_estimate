# 2d_pose_estimate — 高速度カメラ映像(+Xsens)の運動学と着弾解析

`動画 → 2D運動学(pose2d) → [Xsensと合わせて下半身3D(imu3d)] → インパクト時の要因 ↔ 着弾位置(accuracy)`。
旧 `3d_xsens_hscamera`(→ `src/imu3d/`, 詳細は [docs/imu3d.md](docs/imu3d.md))と、着弾解析用の `accuracy` フォルダ(→ `data/`、構成は [data/README.md](data/README.md))を1つに統合した。
解析の引き継ぎ(結果・除外・運用)は [docs/handoff.md](docs/handoff.md)、日報は [docs/daily/](docs/daily/)。

## 構成
```
run.py                 ランチャー: run / calib / analyze / factors / landing / imu3d
src/pose2d/            動画→2D運動学、力学量、要因表(kickfactors)、着弾の図(landing)
src/imu3d/             Xsens+2D→3D、OpenSim出力(旧 3d_xsens_hscamera)
config/                sessions.csv impacts.csv landing.csv(2D・着弾) / trials.csv sensors.csv paths.toml(Xsens・3D)
assets/                OpenSimモデル
data/                  着弾図(landing/)・Xsens(xsens/)・2D映像(video/)・被験者表(subjects.csv)。構成と git に入れる範囲は data/README.md
output/<session>/…     2D出力   output/imu3d/…  3D出力   output/analysis/…  集計・着弾図
requirements.txt (2D・解析)  requirements-opensim.txt (3D/OpenSim出力。無くても2D・解析は動く)
```
- PCごとの違いは git管理外の `config/sessions.local.csv`(動画フォルダ)と `config/paths.local.toml`(Xsensの場所 `xsens_root` など)に書く。
- 3D: `python run.py imu3d --trials kurogi_t3,kurogi_t4 [--no-opensim]`(2D出力は `pose2d_root=output` から読む)。
- **kurogi 以外の被験者の3D**: `python run.py imu3d --init` が、`data/xsens/<名前>/` の試行を `config/trials.csv` に自動追加する(Xsensのインパクト検出から start_s を決める。`xsens_path` 列に各試行のXsensフォルダ)。対象は kataoka・kojima。Xsens は Quat が無ければ Roll/Pitch/Yaw、フレーム欠落は補間、`stop` が無い・姿勢が無い計測は最も静止した0.5秒を直立とみなす。2D運動学は `--window` で広い窓(例 `-0.9 0.5`)にしておく。精度(2D位置誤差の中央値 膝/足首/つま先): kataoka 約42/58/72mm、kojima 約35/53/79mm、kurogi 約48/72/85mm。
- **Xsens を使えない試行・値の除外**: `config/xsens_exclude.csv`(scope 列)。`all` = Xsens を使わない(yamada 全試行、kataoka task1・2・10、kurogi task7)、`posture` = 直立基準の姿勢角・スイング面・3D(imu3d・OpenSim・アニメーション)は使わず、インパクト時刻に対する相対値(ピークの時間差・角速度ピーク)だけ使う(fujii: stop と task を分けずに記録し計測方法にも誤りがあったため)。
- **アニメーション**: `python run.py animate [--trials t1,t2] [--speed 0.2]` で、`output/imu3d/<試行>/animation.mp4`(英語)と `animation_ja.mp4`(日本語)を出す(側面・上面・3D、蹴り脚の軌跡、5倍スロー)。`pipeline` の最後の段階でも実行される。

## 移植・起動
```bash
pip install -r requirements.txt        # 旧環境(my_env等)を有効化してから
python run.py run --session kataoka
```
- パスは `F:\...` / `/mnt/f/...` / 相対(このフォルダ基準)のどれでも書け、実行環境に合わせて自動変換される。
- 被験者ごとの値は **すべて `config/sessions.csv`** に集約(コードは触らない)。

## config/sessions.csv (1行=1セッション)
| 列 | 内容 |
|---|---|
| session | 実行時の名前(`--session`) |
| subject / height_m / mass_kg | 被験者名・身長・体重(結果の summary.json/csv に記録) |
| kick_side | right / left |
| fps | **撮影fps**(動画メタは再生用なので使わない) |
| video_dir | 試行動画のフォルダ(`cali*` で始まるファイルは除外)。`;` 区切りで複数指定可(成功/失敗フォルダなど。無いフォルダは警告してスキップ、動画名の重複はエラー) |
| output_dir | 空欄なら `output/<session>/` |
| calib_json | 3点クリックで作ったキャリブ(`calib/<session>.json`) |
| origin_x, origin_y, scale_x_mm_px, scale_y_mm_px | 旧subjects.csv互換のスケール直指定(入っていれば calib_json より優先) |
| mirror_x | 1でx符号反転 |

旧 subjects.csv の kataoka/yamada/kojima/fujii/kurogi は登録済み(身長は未入力)。kurogiは全試行2000fps、他の被験者は1000fps。試行ごとに違う場合は `--fps` で上書き。

## 手順
1. 新被験者は sessions.csv に行を追加
2. キャリブ(定規動画。原点→横1.1m端→縦1.0m端の順にクリック、Enter確定)
   `python run.py calib --session S --video <cali.avi>`  (GUI不可なら `--points x,y x,y x,y`)
   → 傾き・縦横スケール差を吸収した `calib/S.json` ができる。旧スケール値を使うなら不要。
3. 解析 `python run.py run --session S [--trial T] [--impact N] [--redo]`
   `--window T0 T1` でインパクト基準の出力区間(秒)を指定(例 `--window -0.3 0.1`)。フィルタは全区間にかけてから切り出す。
   インパクトは省略時ボール動き出しから自動。`config/impacts.csv`(session,trial,impact_frame)で固定可(旧 yolo/config/events.csv から移行済み。動画のフレーム番号)。overlay.mp4 で目視確認すること。
4. 出力 `output/S/<trial>/`: `raw_px.csv` `kinematics.csv`(m, s, deg) `summary.json` `plot.png` `overlay.mp4`、`output/S/summary.csv`

## 一括処理(全被験者・全試行)

```
python run.py run --session all --window -0.3 0.1 --skip-done
```
- `--session` は `all`(sessions.csv の全員) / `kataoka,yamada`(複数) / 単独。1試行が失敗しても続行し、最後に失敗・スキップ一覧を表示。
- `--skip-done`: `summary.json` がある試行を飛ばす(中断後の再開用)。`--redo` は検出からやり直す。
- `--window` を使う試行は `config/impacts.csv` にインパクトが必要(未登録はスキップして一覧に出す)。
- 出力: 各試行の `output/<session>/<trial>/`、被験者ごとの `output/<session>/summary.csv`、全体の `output/summary_all.csv`。
- モデルは試行ごとに読み直さず使い回す。1試行は共用PCで約8分(kurogi 2000fps, 窓0.4秒)。

## 集計・図 (`python run.py analyze`)
全試行の `kinematics.csv` から、表と図を `output/analysis/` に出力(`--session a,b` で絞込み)。
- `tables/`: `ball_speed_table.csv`(試行別) `ball_speed_by_subject.csv` `metrics.csv`(全指標) `report.md`
- `subjects/<被験者>/`・`compare/`(被験者間): `fig3_momentum_shank`(全身運動量・蹴り脚下腿角速度) `fig4_segment_energy`(6セグメント) `fig5_boxplots`(ボール速度・E_b・W_p・EI、compareのみ) `effective_mass` `extra`
- 定義(3D側 `energy_flow.py` と同じ): ボール速度=インパクト後0.01〜0.02sの位置の直線近似、E_b=½·0.43·v²、W_p=蹴り脚(大腿+下腿+足)のインパクト前0.1sの Σ|dE|(全身版は `W_p_body_J`)、EI=E_b/W_p。セグメントは阿江ら(1992)。
- 有効質量: 股関節固定の平面3リンク(大腿-下腿-足)のKhatib、作用点=踵-母趾中点、方向=その速度方向。3D版(全身/蹴り脚)とは別の2D近似値。
- 蹴り方向が+xになるよう自動で左右反転する。頭頂が未検出の古い `raw_px.csv` では頭位置を体幹方向から推定(`metrics.csv` の `notes`)。
- ボールは torch 不要の ONNX(YOLOX)で検出。姿勢推定済みの試行には、ボールだけを追加検出する(`ball_checked` が目印、`--redo` で全体をやり直し)。

## 着弾との突き合わせ (`python run.py factors`)
`config/landing.csv`(session,trial,y_m,z_m: ゴール平面の着弾。y>0 は被験者から見て右側、z>0 は上)と各試行の `kinematics.csv` から、インパクト時の要因表 `output/analysis/tables/kick_factors.csv` を出す。着弾が5試行以上そろった被験者は、着弾分布の長軸の傾きと各要因のSpearman相関(`kick_factors_corr.csv`)も出る。
- 要因: 重心-ボール・支持足-ボールの前後距離、体幹前傾、大腿/下腿角、足部ピッチ、足先速度の仰角、接触位置(ボール中心に対する足の位置)、スイング円弧の最下点からインパクトまでの時間(`arc_phase_ms`, +=上昇中に接触)、ボール出射仰角(矢状面)。前方が +x。
- 着弾の y は `land_y_kickside_m`(蹴り足側が +)にも変換する(右足はそのまま、左足は符号反転)。
- 着弾の図: `python run.py landing [--session S]` で `config/landing.csv` からゴール平面の図(等角線1〜15°・試行番号・95%楕円)を `output/analysis/landing/<session>.png` に、楕円の傾き・ばらつきを `tables/landing_summary.csv` に出力(傾き: 蹴り足側yが大きいほどzも大きい=+)。着弾を新しく入れるときは `landing.csv` に行を足す(列: session,trial,y_m,z_m,subject_id,kick_side,experience_y。元の図 `data/landing/ID○○_ball.png` を読み取る場合は ±数cmの誤差)。被験者が2名以上なら、蹴り足側yをそろえて重ねた `all_subjects_kickside.png`(被験者別の楕円の傾き)も出る。
- `factors` は着弾が3試行以上そろうと `analysis/landing/<session>_factors.png`(要因×着弾の散布図)も出す。Xsens の要因(`xs_*`)は `data/xsens/<名前>/` から自動で加わる(除外する試行は `config/xsens_exclude.csv`)。スイング面モデルとの照合(`swing_plane_check`)も出る。
- `python run.py contact`: 足-ボール接触の剛体衝突モデル(+飛翔)で、足先速度の向き・接触点のずれ・反発係数などが着弾をどれだけ動かすか(感度)と、被験者ごとの着弾のばらつきを「足先速度の向き(運動学)で説明できる分」と「残り(接触点のずれに相当)」に分けた結果を `analysis/tables/contact_*.csv`、`analysis/landing/contact_*.png` に出す。
- **結果を別のPCへ渡す**: 共用PCで `python run.py export --with-kinematics`(表・図・試行ごとの要約と kinematics.csv を `results/` に集める。約70MB)→ `git add results; git commit -m "results"; git push`。別のPCで `git pull` → `python run.py import-results` で `output/` に戻り、`factors`・`contact`・`chain` などを再実行できる。動画・raw_px.csv・重畳動画は入れない。
- **図の日本語版**: すべての図(`analyze`・`factors`・`landing`・`contact`・`chain`・`run` の plot.png・`imu3d` の check.png)は、英語版 `x.png` に加えて日本語版 `x_ja.png` も同時に出力する(`pose2d/figio.py`・`pose2d/i18n.py`)。英語で描いた図の文字を、保存時に辞書(`i18n.PHRASES`)で置き換える方式。新しい図の文字を足したときは辞書にも足す(辞書にない英語の句は、コマンド終了時に「[情報] 日本語版で未訳の図の文字」と表示される)。日本語フォントは IPAexGothic / Noto Sans JP / Yu Gothic / Meiryo など(無ければ日本語版は作らず警告)。`POSE2D_NO_JA=1` で日本語版を作らない。
- `python run.py chain`: 要因の頑健性チェック(Xsens と着弾だけで動く)。運動連鎖のタイミング(大腿-下腿のピーク差)を別の定義・生ジャイロで測り直した結果、速さを除いた偏相関、姿勢(支持脚の大腿の傾きなど)を測る時刻を ±40ms ずらした結果と1人ずつ除いた結果を出す(`analysis/tables/chain_check.csv`)。新しい被験者で検証するときは、定義(L0、支持脚の大腿の傾き×上下z)を先に決めてそれだけを見る。
- `python run.py sequential [--session S] [--perm 10000]`: 試行間の依存(前の試行の着弾のずれが次にどう効くか)。`config/landing.csv` だけで動く(試行番号 = 実行順と仮定)。被験者ごとに平均を引いた着弾 (y: 蹴り足側+, z) について、ラグ1自己相関 ACF(1)(左右・上下・分布の長軸/短軸方向。正=補正不足、0=最適、負=補正しすぎ)と補正係数行列 K(次の誤差=K×今の誤差。非対角=クロス補正)を求め、**被験者内で試行順をシャッフルした帰無分布**と比べる(試行が少ないと ACF(1) は約 -1/(n-1) だけ負に偏るため)。出力: `analysis/tables/sequential_acf.csv`・`sequential_K.csv`、`analysis/landing/sequential_{series,acf,K}.png`。y と z の相関が強い被験者(条件数>10)の K は不安定で、p とシャッフル範囲で見る。**傾向を除いた版**も同時に出す(`*_detrend.csv/png`、傾向の大きさは `sequential_trend.csv`): 試行番号に対する直線を引いた残差で同じ解析。直線を引いた残差は独立な雑音でも ACF(1) が余計に負になる(n=10 で平均 -0.20)ため、帰無分布は残差のシャッフルではなく Freedman-Lane 法(直線+シャッフルした残差を直線で引き直す)で作る。方法の出典: van Beers ら 2013 PLoS ONE(ダーツのラグ1自己相関)、Cheng & Sabes 2006(状態空間モデル)。
- `python run.py mediation [--session S] [--perm 2000]`: 補正の「つまみ」探し(`factors` の後に実行。`output/analysis/tables/kick_factors.csv` を使う)。前の誤差 e(n) → 次の運動学の量 k(n+1) → 次の着弾 e(n+1) の経路を、被験者内で順位正規化して求める。a = e(n) → k(n+1)(**k(n)を一定にした**偏回帰。平均への回帰を補正と取り違えない)、b = k(n+1) と e(n+1) の相関、間接効果 = a×b。検定は被験者内で試行順をシャッフル(試行内の k と e の対応は保つ)。グループ(ALL、EXP=8年以上)は被験者の平均で、つまみ全体の max-T 補正の p(`p_ind_adj`)も出す。出力: `analysis/tables/mediation.csv`、`analysis/landing/mediation_{a,indirect}.png`。組の数が被験者あたり5〜10と少なく検出力は低い。Xsens 由来のつまみは使える被験者が2〜3人。
- `python run.py score [--session S] [--boot 4000]`: 着弾からタスク精度スコア(CE=定誤差、VE=変動誤差、TE=総合誤差。距離11mの角度にも換算)を定義して試算し、着弾図の読み取り誤差(乱数)と試行の標本誤差(ブートストラップ)による不確かさ、新規計測の要件(入力の測定精度・VEを比べるのに要る試行数)も出す。左右 y・上下 z の2軸ごとのスコアと区間(`accuracy_score_axes.csv`、`accuracy_score_axes.png`)、目標の高さの感度、軸ごとの必要精度も出る。`analysis/tables/accuracy_*.csv`、`analysis/landing/accuracy_{score,score_axes,score_uncertainty,requirements}.png`。定義・結果・要件・要判断事項は [docs/accuracy_score_definition.md](docs/accuracy_score_definition.md)。
- `python run.py trialdep-figures`: 試行間の依存の結果を直感的に読める図にまとめる(`sequential`・`mediation` の後)。`analysis/landing/trialdep_paths.png`(着弾を試行順に矢印でつないだ図)、`trialdep_lag.png`(今回のずれ × 次回のずれ。傾き0=無関係、−1=完全に反対へ戻す)、`trialdep_summary.png`(ACF(1)と、媒介解析の p 値の QQ プロット)。
- `python run.py pipeline [--session S] [--steps run,analyze,factors,landing,sequential,mediation,trialdep,contact,imu3d] [--skip-done] [--no-overlay] [--no-opensim]`: 上の段階を順に実行(1段階が失敗しても続行)。3D(imu3d)には 2D運動学を `--window -0.9 0.5` で出力しておく必要がある。

## overlay.mp4
全身(頭・体幹・左右の腕・左右の脚・足先/踵)の骨格と関節点を部位別に色分け(左上に凡例)。低信頼の点は輪だけ表示。
頭・鼻の点は `--redo` で再検出した結果にのみ出る(古い raw_px.csv には含まれない)。

## 設定 `config/pipeline.toml`
- `detect.backend`: `rtmlib`(足先/踵あり) / `yolo`(COCO17: 足先なし)。ボール検出は ultralytics 必須
- `process.cutoff_hz`: 数値 or `"auto"`(Winter残差解析)。ボールはローパスしない

## 注意
- x: 定規の横方向, y: 上向き(m)。レンズ歪み補正は未実装。定規は蹴り足の動く矢状面で撮る
- テスト: `python tests/test_pipeline.py` / `python tests/test_sessions.py`

## GPU
- `pipeline.toml` の `device = "auto"`: onnxruntime が CUDA を使えれば自動でGPU、無ければCPU(起動時に `[情報] 姿勢推定デバイス` を表示)。
- GPU機では `pip uninstall onnxruntime && pip install onnxruntime-gpu==1.20.2`(CUDA 12系 + cuDNN 9 が必要)。ボール検出(ultralytics)はCUDA版torchで自動的にGPUを使う。
