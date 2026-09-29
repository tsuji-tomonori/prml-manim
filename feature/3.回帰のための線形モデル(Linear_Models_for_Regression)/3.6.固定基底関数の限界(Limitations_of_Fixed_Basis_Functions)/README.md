# PRML 3.6 固定基底関数の限界

問いを出し、数値実験を動かし、式で確かめる **約10分31秒** の動画です。
Manim Community と VOICEVOX:WhiteCUL（ノーマル、speaker 23）を使用しています。

[動画（480p15）](media/videos/prml_3_6_fixed_basis_limitations/480p15/PRML36FixedBasisLimitations.mp4) ／ [全台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## シーン構成

尺と開始時刻は合成した WAV の実測値です。

| シーン | 問い | 開始（秒） | 尺（秒） | 視覚的アイデア | 原文参照 |
|---|---|---:|---:|---|---|
| scene01 | 曲線を曲げられるのに、何が限界？ | 0.000 | 86.133 | 重みのつまみと3つの基底・予測曲線を連動 | §3.6 pp.172–173; 復習 Eqs.(3.3), (3.15) |
| scene02 | 入力が増えると、山はいくつ必要？ | 86.133 | 93.267 | 5点→25点→125点、対数軸上で次元を増やす | §3.6 p.173; 関連 Fig.1.21 |
| scene03 | 二つの座標。でも、動く自由は一つ？ | 179.400 | 87.667 | 曲線上の点を動かし、曲線を直線へほどく | §3.6 p.173（多様体） |
| scene04 | 山を置く場所を、データから選ぶ | 267.067 | 87.133 | 81中心から近傍を抽出し、12中心をデータへ移す | §3.6 p.173（局所基底） |
| scene05 | 点が広がっていても、予測に効く方向は？ | 354.200 | 99.467 | 応答方向を回し、色の予測とRMSEを連動 | §3.6 p.173（関連方向） |
| scene06 | 基底そのものに、向きを選ばせる | 453.667 | 85.200 | 内部の向き・位置・幅と外側の重みを分けて操作 | §3.6 p.173; 関連 Eq.(3.6) |
| scene07 | 学ぶのは、重みか、見る場所か | 538.867 | 91.933 | 同じ平面で曲線・格子・局所配置・方向を振り返る | §3.6 pp.172–173 |

## 視覚補足と復習

- scene02 冒頭（86.133–96.800秒、10.667秒）：`復習: 1.4 次元の呪い`。元動画の青い箱と紫へ変わる奥行きの配色で5→25→125を再現し、黄色の中心を基底数へ結びます。
- scene03 冒頭（179.400–191.933秒）：既存の点の説明に `復習: 1.4 多様体` を付記。字幕と音声にも節番号を入れ、独立カードは追加しません。文の変更で1.600秒増えました。
- scene05 の特徴の式の直後（394.067–405.133秒、11.067秒）：共通図法 V08a。本文を一時置換し、長さ1の黄色の方向、青い点と垂線、緑の影を示します。点 `(1,0)` の影を数直線の `1/√2` へ移し、内積から特徴への対応を見せます。

既存の説明と図を残し、補足の前後で同じ本文へ戻ります。局所基底とシグモイドは既存の操作で説明できているため、補足を重ねていません。

## 原文との対応

Bishop, *Pattern Recognition and Machine Learning* (2006), §3.6、印刷 pp.172–173（PDF pp.192–193）を読み、構成しました。
**3.6 節には番号付きの式・図・表はありません。** 復習する線形結合は p.138 Eq.(3.3)、最小二乗解は p.142 Eq.(3.15)、シグモイドは p.139 Eq.(3.6) に対応します。
局所Gaussianの式は p.139 Eq.(3.4) を二次元へ拡張した説明用の式です。幅は原文の s と区別して h と表記します。
格子は §1.4 p.35 Fig.1.21 の考え方にも関連しますが、原図を複製していません。

- 固定基底の利点、次元とともに増える基底数、データ多様体、目標に関係する少数の方向を扱います。
- `B=K^D` は全入力空間を各方向 K 個の局所基底で覆う例です。全種類の基底についての普遍則ではありません。B は説明用の中心数で、Eq.(3.3) のバイアスを含む M と区別します。
- 最小二乗の逆行列表示には列フルランクの条件があります。実計算は `numpy.linalg.lstsq` を使います。
- 曲線をほどく動き、中心の移動、応答方向の回転は自由度を示す演出です。学習アルゴリズムの反復経路ではありません。
- データから中心を選ぶ方法と、基底の内部パラメータまで最適化する方法を区別しています。内部まで学ぶ場合、学習全体は線形最小二乗にはなりません。
- 原文が挙げる動径基底関数ネットワーク、サポートベクトルマシン、関連ベクトルマシン、ニューラルネットワークへの動機を紹介します。各手法の学習手順はこの動画の範囲外です。

## 独自実験

- 最初の回帰: 25点、3つのガウス基底と定数項。中心 `[-0.65, 0, 0.65]`、幅0.23、ノイズ標準偏差0.035、seed 3601。最小二乗RMSEは0.0362119183。
- 曲線: `(s, 0.68 sin(2.4s))`、72点、小さな座標ノイズ（標準偏差0.022、seed 3603）。9×9格子のうちデータまでの距離0.18未満は17中心。データの順序から12中心を選びます。
- 関連方向: 正方形内の70点（seed 3605）。目標は `sigmoid(3(x1+x2)/sqrt(2))`。水平から45度へ回すと、実験点上のRMSEが0.2203162714から0へ変わります。既知の生成関数への一致であり、未知データでの汎化性能ではありません。
- シグモイド: `phi_j(x)=sigmoid(a_j^T x+b_j)` の向き、位置、傾きと、外側の重みを別々に操作します。

## 実装と再生成

- `prml_3_6_fixed_basis_limitations.py`: 公開クラス `PRML36FixedBasisLimitations` を維持した7シーン。
- `basis_model.py`: NumPy の再現可能な数値実験。
- `narration_content.py`: 59 beat・116文の `display` / `speech` の正本。
- `caption_layout.py` / `scene_support.py`: 1.1と同じ実測文字高・数式本体の中心合わせ、PCM文境界に基づく同期。
- `make_voicevox_narration.py`: 文単位合成、台本とWAVのハッシュ検査、シーン単位での再開。
- `check_narration_readings.py` / `reading_check.md` / `reading_check.json`: 全文の修正前後の読み。
- `verify_numerics.py` / `verify_video.py` / `validation.json`: 数値・字幕・音声・映像・同期検証と実測結果。
- `assets/voicevox/`: 新構成の7 WAVとmanifest。古い音声は置換済み。

このディレクトリで実行します。既存venv、Noto Sans CJK JP、LaTeX、dvisvgm、ffmpegを使用します。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python write_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py --captions
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_3_6_fixed_basis_limitations.py PRML36FixedBasisLimitations
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py --output /tmp/prml36-review
```

Engineは `http://127.0.0.1:50021`。話速1.08、抑揚0.95です。接続できなければ停止し、コンテナは操作しません。
`make_voicevox_narration.py --from-scene scene04` で途中再開できます。
音声は文単位で合成し、各beat末尾に約0.35〜0.42秒だけ余白を加えます。
音声と台本が不整合なら描画を停止します。タイムラインは `media/prml36_timeline.json` に出力します。

`verify_video.py` は画像を抽出し `validation.json` の目視状態を pending にします。画像を見て確認した後に、確認した範囲を記録してください。
全編の通し聴取と全フレームの目視は実施していません。

## 最終検証

`py_compile` と既存の6カテゴリの数値・音声・字幕検証に成功。指定のキャッシュ無効化・flush付き480pフルレンダリングを2回実施し、最終版は74 animationsでした。

| 項目 | 結果 |
|---|---|
| 映像 | H.264、854×480、15fps、630.799022秒 |
| 音声 | AAC、630.826667秒（差0.027645秒） |
| 元尺との比較 | 607.066011秒 → 630.799022秒、+23.733011秒（+3.909461%） |
| 3秒以上の無音 | -45dBしきい値で0件 |
| 音量 | 平均 -26.5 dB、最大 -5.6 dB |
| 字幕 | 116文、最大幅9.3891・高さ0.7519、安全領域内。読み仮名の検索0件 |
| 読み確認 | 全116文のaudio_query取得、新規4文・変更1文の結果を目視確認 |
| 既存音声 | 変更していない111文のPCMが元WAVと完全一致 |
| 最終動画の目視 | 67枚。全7シーン、追加2場面の前後・各動作、参照ラベルを確認 |
| 同期 | 追加2場面の全10動作に発声あり。シーン尺の最大誤差7.316e-11秒 |

初回65枚を目視し、最終版67枚を再抽出して画像ハッシュを比較。差分と追加画像を再目視しました。数式の切り替えは旧表示を消してから新表示を出し、文字の重なりを防いでいます。
新規の誤読は見つからず、多様体は既存の修正に合わせて speech を「たようたい」にしています。
全編の通し聴取と全フレームの目視は未実施です。実測値・抽出時刻・動作とPCMの照合は `validation.json` を参照してください。

## 3Blue1Brownから取り入れた手法

[3b1b/videos](https://github.com/3b1b/videos) の `_2023/clt/main.py`（数値・つまみ・図形を同じ状態へ結び付ける）と `_2021/quick_eigen.py` の `ShowSquishingAndStretching`（方向ベクトルと変形の連動）、
[3b1b/manim](https://github.com/3b1b/manim) の `value_tracker.py` / `mobject_update_utils.py` を参照しました。
ManimGLのコードは取り込まず、Manim CEのValueTrackerとupdaterで実装しています。

## 音声クレジット

VOICEVOX:WhiteCUL
