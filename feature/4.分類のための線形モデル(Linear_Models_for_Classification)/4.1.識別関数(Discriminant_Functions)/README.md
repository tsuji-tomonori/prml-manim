# 4.1 識別関数

一本の境界を回す、点の影を落とす方向を変える、間違えた点で境界を更新する。PRML 4.1 を、問いから始まる10個の視覚実験に作り直した約9分5秒の動画です。Manim Community を使用し、元のファイル名と `PRML41DiscriminantFunctions` クラスを維持しています。

[音声付き動画（480p15）](media/videos/prml_4_1_discriminant_functions/480p15/PRML41DiscriminantFunctions.mp4) / [台本](narration_script.md) / [全文の読み確認](reading_check.md)

## 構成

| Scene | 尺（秒） | 視覚的アイデア | 原文（本文ページ） |
|---|---:|---|---|
| 1 境界の向きと位置 | 52.200 | 重みの回転・バイアスの移動・符号による分類 | pp.181–182、(4.4)–(4.5)、図4.1 |
| 2 スコアと距離 | 43.600 | 垂線を動かす、重みを倍にする、距離を正規化 | pp.181–182、(4.5)–(4.8)、図4.1 |
| 3 多クラスのスコア | 56.200 | 半平面の共通部分、点の移動と最大値、凸領域 | pp.182–184、(4.9)–(4.12)、図4.2–4.3 |
| 4 最小二乗の弱点 | 69.667 | 正しい側の点を遠ざけて再学習、中央クラスの消失 | pp.184–187、(4.13)–(4.19)、図4.4–4.5 |
| 5 Fisher の射影 | 67.333 | 同じ軸を回し、影・平均・評価値を連動 | pp.186–189、(4.20)–(4.30)、図4.6 |
| 6 最小二乗との関係 | 50.067 | 特別な目標値、全体平均を通る境界、軸の一致 | pp.189–190、(4.31)–(4.38) |
| 7 多クラス Fisher | 52.200 | 平均の三角形、中心からの矢印、固有値の棒 | pp.191–192、(4.39)–(4.51) |
| 8 パーセプトロン | 68.600 | 実際の誤分類点で3回更新、誤り数を連動 | pp.192–195、(4.52)–(4.56)、図4.7 |
| 9 収束の条件 | 45.333 | 複数の解、XOR と積特徴への移動 | pp.194–196、収束定理、固定特徴の限界 |
| 10 三つの見方 | 40.000 | 同一座標上で学習の意味を振り返る | pp.184–196、4.2への接続 |

原文PDFのページ番号は本文ページに20を足したものです。4.1には表はありません。図4.8の歴史的ハードウェア写真は転載せず、数学的な内容を中心に扱います。

## 数値と原文の対応

- 境界・点・スコア・射影・学習結果は NumPy で計算します。原図のデータや図版は複製していません。
- 最小二乗は定数列付き設計行列を `np.linalg.lstsq` で解きます。点の移動中も毎回再学習し、単に二本の境界を補間する描画にはしていません。表示する2クラス出力は和が1でも範囲外になり、3クラスの対称例では中央のスコアが常に0.2、両端の最大スコアが0.4以上となります。
- Fisher は青30点・橙22点、seed=4105。集団内行列は正定値で、暗黙の正則化は加えません。ばらつきは原文(4.24)と同じ、点数で割らない二乗偏差和です。評価値は平均差方向0.067797、Fisher方向0.305348です。
- 特別な最小二乗の目標は青52/30、橙−52/22。重みは `S_W⁻¹(m₁−m₂)` の正の倍数となり、(4.30)の向きと符号を除いて一致します。射影軸の一致と、しきい値の選び方を分けて説明します。
- 多クラス例は4次元、3クラス、各24点、seed=4107。画面の平均の三角形は平面内の模式表示です。固有値は実データから計算しています。`W` の列を射影軸として `y=Wᵀx` に統一し、(4.51)は `Tr[(WᵀS_W W)⁻¹(WᵀS_B W)]` と表示します。原文の(4.39)と(4.51)の行列の向きの不一致をそのまま持ち込んでいません。
- パーセプトロンは `φ=(1,x₁,x₂)`、学習率1。必ず誤分類を確認してから更新し、この8点では3回で分離します。更新途中の連続変形は離散更新を見せるための補間です。収束保証は使用する特徴空間での線形分離可能性が条件です。
- XOR の積特徴は独自の補助例です。ロジスティック回帰との実験比較、歴史的人物・装置の紹介、各導出の全行は省略しています。

## 字幕・音声・同期

`narration_content.py` に10シーン・57 beat・114文を持ち、display と speech を分離しています。字幕内の数式は MathTex です。1.1と同じ実測かな文字高、本体中心線、左右の余白を使い、空白グリフが高さを壊すケースも補正しました。

VOICEVOX の全 speech を `audio_query`（speaker=23）へ送り、修正前後の読みを `reading_check.md` / `.json` に保存しました。15文を修正しています。読みの全文目視は音声の通し聴取とは区別しています。

文単位の音声を合成し、PCMの長さを manifest に保存します。字幕と動作は同じ文時刻を参照し、15fpsの累積フレーム境界へ合わせます。古い台本や変更されたWAVはハッシュで検出し、描画を停止します。実測タイムラインは `media/prml41_timeline.json` へ出力されます。

## 再生成

既存の venv を使い、このディレクトリで実行します。Engine は `http://127.0.0.1:50021`。接続できなければ停止し、コンテナは操作しません。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python narration_content.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_4_1_discriminant_functions.py PRML41DiscriminantFunctions
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_media.py
/home/t-tsuji/project/prml-manim/.venv/bin/python review_video.py
```

`make_voicevox_narration.py --from-scene scene04` で音声生成を再開できます。文音声キャッシュは worktree 内 `.working/voicevox-lines/` です。動画レビューは全beat・全記号字幕・3シーンの同期前後画像を `media/review/` に出力します。

## 3Blue1Brown から参考にした手法

ManimGL のコードを移植せず、演出を CE の ValueTracker と updater で実装しています。

- [GeometricInterpretation](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2016/eola/chapter7.py)：内積を垂線と射影の長さで見せ、図から式へ進む。境界からの距離と Fisher の影へ適用。
- [break_down_cost_function / TwoVariableInputSpace](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/nn/part2.py)：目標との差と損失、どちらへ動かすかという問いを結ぶ。最小二乗の弱点とパーセプトロン更新へ適用。
- [ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py)：状態を一か所に置き、境界・矢印・影・スコアを連動。同じ座標で前後を比較。

## 音声クレジット

VOICEVOX:WhiteCUL（ノーマル、speaker 23）。Engine 0.25.2、話速1.08、抑揚0.95。

## 最終動画の検証

- `py_compile` 成功、数値・音声整合の8検証群に合格。
- 全編レンダリング成功（59 animations）。H.264、854×480、15fps、映像545.199344秒、AAC音声545.237333秒、差0.037989秒。
- `silencedetect=noise=-45dB:d=3` は0件。`volumedetect` は平均−26.3 dB、最大−6.1 dB。
- 全114字幕の寸法を検査。全57 beat、記号入り字幕全7文、3シーンの同期画像9枚、合計73枚を目視し、重なり・はみ出しを修正して再出力。
- シーン1・5・8で、PCMの文頭時刻と動作の開始・中間・終了画像を照合。
- 全114文のAPI読みを修正後に再取得して確認。display中の指定した記号の読み仮名は検索結果0件。

数値は `validation_numerics.json`、字幕寸法は `validation_captions.json`、媒体情報・SHA-256は `validation_media.json`、画像抽出時刻は `validation_video.json` に保存しています。全編の通し聴取、全フレーム目視、高解像度版のレンダリングは未実施です。
