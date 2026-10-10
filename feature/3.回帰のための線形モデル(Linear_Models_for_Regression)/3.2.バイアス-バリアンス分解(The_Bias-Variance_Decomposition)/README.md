# PRML 3.2 バイアス–バリアンス分解

## 物語の設計

全編の問いは「同じ入力 $x=0.25$ で、安定した予測を選ぶほど正解から遠ざかるのはなぜか」。seed=3202 の同じ訓練集合 0番・2番を使う。弱い正則化（$\ln\lambda=-3$）では予測が 0.830・1.457 に割れる。強い正則化（$\ln\lambda=3$）では 0.402・0.299 に集まるのに、真の条件付き平均は 1.000。冒頭でどちらを信じるか予想する間を置き、結末で同じ入力・同じ訓練集合に戻る。全編は約8分46秒。

| 場面 | 前の疑問から始める実験と発見 |
|---|---|
| 1 | 2組で予測を切り替え、安定だけでは正しさを測れない違和感を置く |
| 2 | 「正解」とは何か。1.5 の復習を使い、条件付き平均と残るノイズを見つける |
| 3 | 1本の偶然をどう見抜くか。100組を学び直し、予測の平均を作る |
| 4 | 何が違うのか。平均と正解の距離、各予測と平均の距離を別々に動かす |
| 5 | 2つを足してよいのか。符号付きの交差項が平均で消えることを確かめる |
| 6 | つまみで両方を小さくできるか。$\lambda$ を往復して交換関係を見る |
| 7 | 1点の発見は全体でも成り立つか。入力平均とテスト誤差で釣り合いを調べる |
| 8 | 冒頭の $x=0.25$ に戻って値を照合し、実際には1組しかない問題を次節へ渡す |

グラフの背面に装飾の枠・台・影を置かず、データの距離・二乗面積・つまみだけに視線を集める。既存の 1.5 条件付き平均の復習と、3組×2入力の2段階平均の補足は物語の途中に残す。数値は自作実験から算出し、原典の式 (3.36)–(3.47) と図 3.5–3.6 の意味に合わせる。

[動画（480p15）](media/videos/prml_3_2_bias_variance_decomposition/480p15/PRML32BiasVarianceDecomposition.mp4) / [台本](narration_script.md) / [全文の読み確認](reading_check.md)

## 構成

| Scene | 開始 | 尺 | 視覚的アイデア | 原文参照 |
|---|---:|---:|---|---|
| scene01: 安定した予測ほど、外れる？ | 0.000秒 | 56.800秒 | 同じ入力の 0番・2番を切り替え、強い抑制の意外な外れを示す | pp.147–148 / Eq.(3.38) |
| scene02: そもそも、正解はどこ？ | 56.800秒 | 79.067秒 | 散らばる観測から条件付き平均を探し、1.5 の図を復習 | p.148 / Eqs.(3.36)–(3.37) |
| scene03: 1本の偶然を、どう見抜く？ | 135.867秒 | 61.667秒 | 25点×100組を学び直し、20本を表示して100本の平均へ | pp.148–151 / Fig.3.5 / Eq.(3.45) |
| scene04: 外れ方は、何が違う？ | 197.533秒 | 60.267秒 | 1入力の100予測を数直線へ移し、平均のずれと揺れを別々に動かす | p.149 / Eq.(3.40) |
| scene05: 2つの面積を、足してよい？ | 257.800秒 | 62.200秒 | 交差項の符号付き差を打ち消し、二乗誤差を2つに分ける | pp.148–149 / Eqs.(3.38)–(3.40) |
| scene06: つまみで、両方を減らせる？ | 320.000秒 | 59.800秒 | λのつまみを往復し、曲線群・平均・B²・Vを連動 | pp.147,149–152 / Fig.3.5 / Eq.(3.27)（前節） |
| scene07: 1点の発見は、全体でも？ | 379.800秒 | 83.200秒 | 入力の平均とテスト誤差で釣り合いを調べ、3組×2入力を補足 | pp.149–151 / Eqs.(3.41)–(3.47) / Fig.3.6 |
| scene08: 最初の予測に、戻ろう | 463.000秒 | 63.467秒 | 冒頭と同じ入力・訓練集合・数値へ戻り、1組しかない現実から3.3へ | pp.151–152 / §3.3 への接続 |

## 数値実験

NumPy seed=3202。入力は区間[0,1]の一様分布、真の回帰関数は sin(2πx)、観測ノイズは平均0、標準偏差0.25の正規分布。
独立な100組の訓練集合を生成し、各25点に24個のGaussian基底（中心は等間隔、幅0.08）と定数項を使って学習します。
正則化は原文(3.27)に合わせ、定数項を含む全25係数に適用します。λの連続変形中も、その時点の正規方程式の解を固有値分解から計算し直します。

入力積分は独立な2000点で近似。テストMSEは別の1000点で、100モデルの誤差を平均します。
61候補の中でB²+Vが最小となるのは ln λ=0.1。B²=0.00394894、V=0.01737765、noise=0.0625、推定期待損失=0.08382659、テストMSE=0.08862283です。
有限サンプルの値であり、原著の図の数値を転写したものではありません。最適値はモデル・データ・候補グリッドに依存します。

Scene 2の12観測、Scene 4の群れの移動・拡大、Scene 5の符号付き4本の棒は補助例・模式図です。Scene 8 の事後分布の式は次節への予告であり、表示した2本を事後分布から抽出したものではありません。
曲線の描画値はクリップしていません。Scene 3・6は20本を表示し、黄色の平均には100本すべてを使います。
データ集合の切り替え時の点・線の補間は見せ方であり、中間フレームの点で再学習した結果ではありません。

## ファイルと再生成

- `prml_3_2_bias_variance_decomposition.py`: 公開クラス `PRML32BiasVarianceDecomposition`、8シーン。
- `bias_variance_model.py`: 再現可能なデータ・正則化回帰・分解・テスト評価。
- `video_support.py`: 1.1から移植した日本語とMathTexの高さ・中心合わせ、PCMによる文同期。
- `narration_content.py`: 125文のdisplay/speech、62 beat。
- `export_narration_script.py`, `narration_script.md`: manifest の実測時刻から台本を再生成。
- `make_voicevox_narration.py`: WhiteCUL合成、ハッシュ整合確認、シーン別再開。整合済みの音声は再利用。
- `check_narration_readings.py`, `reading_check.md`, `reading_check.json`: 修正前後のAPI読み全文。
- `assets/voicevox/`: 8 WAVとmanifest。物語化に合わせ全8 WAVを更新。余剰WAVなし。
- `verify_visual_aids.py`: 補足例の数値、全beatの音声時刻、追加場面の操作前後のフレーム抽出。フルレンダリング後に実行。
- `verify_numerics.py`: 回帰解、分解恒等式、テストMSE、描画範囲、音声・字幕整合の検証。

このディレクトリで実行:

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
OPENBLAS_NUM_THREADS=1 /home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
OPENBLAS_NUM_THREADS=1 /home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_3_2_bias_variance_decomposition.py PRML32BiasVarianceDecomposition
OPENBLAS_NUM_THREADS=1 /home/t-tsuji/project/prml-manim/.venv/bin/python verify_visual_aids.py
```

VOICEVOXは http://127.0.0.1:50021 、WhiteCUL ノーマル（speaker=23）、話速1.08・抑揚1.08。
途中再開は `make_voicevox_narration.py --from-scene scene04`。動画は音声が未生成・台本と不一致の場合に停止します。

## 参考資料

PRML (2006) §3.2、印刷pp.147–152 / PDF pp.167–172、式(3.36)–(3.47)、図3.5–3.6（節内に表なし）。
[3b1b/videos の MeanAndStandardDeviation](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/clt/main.py) の、平均線・偏差の正方形・分布の連続変形を参考にしました。
[3b1b/manim の ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py) を参照し、共有状態から図・数値を更新する考え方をManim CEで実装しています。ManimGLコードは取り込んでいません。

## 音声クレジット

ナレーション: VOICEVOX:WhiteCUL

## 今回の視覚補足と復習

| 位置 | 尺 | 内容 |
|---|---:|---|
| 74.933–86.800秒、scene02-03-01の直前 | 11.867秒 | R: 1.5 の条件付き密度・期待二乗損失を復習し、平均を $h(x)$ へつなぐ |
| 413.400–426.467秒、scene07-04-02の直後 | 13.067秒 | V07b: 3組×2入力で予測の平均、二乗偏差、入力の平均を順に動かす |

既存の補足図の内容と文IDを維持し、音声の実測時刻に合わせて物語へつないだ。補足図の説明用数値は列平均 2・2、列分散 $8/3$・$2/3$、全体 $5/3$。本編の自作実験と混ぜない。

## 今回の検証

- 全 Python の `py_compile`、既存数値検証、補足の小例・音声時刻検証に成功。分解恒等式の最大誤差は $2.08\times10^{-17}$。冒頭と結末の同じ入力・訓練集合の表示値も再計算して照合。
- 全8シーン・62 beat・125文で台本、`display`/`speech`、`audio_query` の読み記録、manifest、WAVのハッシュ・PCM時刻・尺が一致。全125字幕の最大幅9.045、高さ0.292（Manim単位）。個数・番号・倍率の漢数字残りは0件。
- キャッシュ無効で全編480pを再レンダリングし、73アニメーションを完了。最終MP4は854×480 / 15fps、映像526.466667秒、音声526.506667秒、差0.040000秒。
- `ffprobe`、−45dBでの`silencedetect`、`volumedetect`、映像・音声の全編デコードを実施。3秒以上の無音0件、平均−26.5dB、最大−5.3dB、デコードエラーなし。
- 最終版から補足検証用56枚と物語・つまみ操作用40枚を抽出し、うち57枚を接触シートで目視。移動・強調の途中と直後、字幕と図の対応、枠・重なり・式の可読性を確認。初回で見つけた数値ラベルと補足表の一時的な文字重なりを修正後に再確認。

全編の通し聴取、全フレームの目視、高解像度での全編レンダリング、視聴者による理解度評価は行っていない。

[今回の作業完了レポート](../../../reports/working/20261011-0544-prml-3-2-story.md)
