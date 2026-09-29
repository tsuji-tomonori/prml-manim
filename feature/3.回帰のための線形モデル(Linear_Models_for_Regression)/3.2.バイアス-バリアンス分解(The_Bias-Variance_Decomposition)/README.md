# PRML 3.2 バイアス–バリアンス分解

同じ学習法でデータを取り直したら、予測はどれくらい変わるか。点と曲線の実験から、平均のずれ・予測の揺れ・観測ノイズの三つを分ける、9分33.4秒の日本語動画です。

[動画（480p15）](media/videos/prml_3_2_bias_variance_decomposition/480p15/PRML32BiasVarianceDecomposition.mp4) / [台本](narration_script.md) / [全文の読み確認](reading_check.md)

## 構成

| Scene | 開始 | 尺 | 視覚的アイデア | 原文参照 |
|---|---:|---:|---|---|
| scene01: 同じ方法なのに、予測が変わる？ | 0.000秒 | 45.400秒 | 同じ軸で点と学習曲線を取り替える問い | pp.147–148 / Eq.(3.38) |
| scene02: 観測の真ん中を狙う | 45.400秒 | 85.533秒 | 観測の散らばり、動く予測点、距離の二乗の面積。1.5の条件付き平均を復習 | p.148 / Eqs.(3.36)–(3.37) |
| scene03: データを百回取り直したら | 130.933秒 | 70.667秒 | 曲線を重ねて100本の平均へ。表示は20本 | pp.148–151 / Fig.3.5 / Eq.(3.45) |
| scene04: ずれを、二つの距離に分ける | 201.600秒 | 71.467秒 | 交点を数直線へ移し、群れの移動と拡大を分離 | p.149 / Eq.(3.40) |
| scene05: 平均を足して引くと、式も分かれる | 273.067秒 | 71.533秒 | 平均を足して引き、符号付きの差が相殺する様子 | pp.148–149 / Eqs.(3.38)–(3.40) |
| scene06: 正則化のつまみで、全体を動かす | 344.600秒 | 71.333秒 | λスライダーで曲線群・平均・B²・Vを連動 | pp.147,149–152 / Fig.3.5 / Eq.(3.27)（前節） |
| scene07: 入力全体で足すと、予測誤差になる | 415.933秒 | 94.933秒 | 入力を走査し、数値計算した誤差曲線とノイズの差へ。3組×2入力の二段階平均 | pp.149–151 / Eqs.(3.41)–(3.47) / Fig.3.6 |
| scene08: この平均は、実際に計算できる？ | 510.867秒 | 62.533秒 | 100組から手元の1組へ。係数の事後平均につなぐ | pp.151–152 / §3.3 への接続 |

## 数値実験

NumPy seed=3202。入力は区間[0,1]の一様分布、真の回帰関数は sin(2πx)、観測ノイズは平均0、標準偏差0.25の正規分布。
独立な100組の訓練集合を生成し、各25点に24個のGaussian基底（中心は等間隔、幅0.08）と定数項を使って学習します。
正則化は原文(3.27)に合わせ、定数項を含む全25係数に適用します。λの連続変形中も、その時点の正規方程式の解を固有値分解から計算し直します。

入力積分は独立な2000点で近似。テストMSEは別の1000点で、100モデルの誤差を平均します。
61候補の中でB²+Vが最小となるのは ln λ=0.1。B²=0.00394894、V=0.01737765、noise=0.0625、推定期待損失=0.08382659、テストMSE=0.08862283です。
有限サンプルの値であり、原著の図の数値を転写したものではありません。最適値はモデル・データ・候補グリッドに依存します。

Scene 2の12観測、Scene 4の群れの移動・拡大、Scene 5の符号付き4本の棒、Scene 8のベイズ平均への接続は補助例・模式図です。
曲線の描画値はクリップしていません。Scene 3・6は20本を表示し、黄色の平均には100本すべてを使います。
データ集合の切り替え時の点・線の補間は見せ方であり、中間フレームの点で再学習した結果ではありません。

## ファイルと再生成

- `prml_3_2_bias_variance_decomposition.py`: 公開クラス `PRML32BiasVarianceDecomposition`、8シーン。
- `bias_variance_model.py`: 再現可能なデータ・正則化回帰・分解・テスト評価。
- `video_support.py`: 1.1から移植した日本語とMathTexの高さ・中心合わせ、PCMによる文同期。
- `narration_content.py`: 125文のdisplay/speech、62 beat。
- `make_voicevox_narration.py`: WhiteCUL合成、ハッシュ整合確認、シーン別再開。整合済みの音声は再利用。
- `check_narration_readings.py`, `reading_check.md`, `reading_check.json`: 修正前後のAPI読み全文。
- `assets/voicevox/`: 8 WAVとmanifest。今回の追加でscene02/07を更新し、他6本を保持。余剰WAVなし。
- `verify_visual_aids.py`: 補足例の数値、全beatの音声時刻、追加場面の操作前後のフレーム抽出。フルレンダリング後に実行。
- `verify_numerics.py`: 回帰解、分解恒等式、テストMSE、描画範囲、音声・字幕整合の検証。

このディレクトリで実行:

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
OPENBLAS_NUM_THREADS=1 /home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
OPENBLAS_NUM_THREADS=1 /home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_3_2_bias_variance_decomposition.py PRML32BiasVarianceDecomposition
OPENBLAS_NUM_THREADS=1 /home/t-tsuji/project/prml-manim/.venv/bin/python verify_visual_aids.py
```

VOICEVOXは http://127.0.0.1:50021 、WhiteCUL ノーマル（speaker=23）、話速1.08・抑揚0.95。
途中再開は `make_voicevox_narration.py --from-scene scene04`。動画は音声が未生成・台本と不一致の場合に停止します。

## 参考資料

PRML (2006) §3.2、印刷pp.147–152 / PDF pp.167–172、式(3.36)–(3.47)、図3.5–3.6（節内に表なし）。
[3b1b/videos の MeanAndStandardDeviation](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/clt/main.py) の、平均線・偏差の正方形・分布の連続変形を参考にしました。
[3b1b/manim の ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py) を参照し、共有状態から図・数値を更新する考え方をManim CEで実装しています。ManimGLコードは取り込んでいません。

## 音声クレジット

ナレーション: VOICEVOX:WhiteCUL

## 今回の視覚補足と復習

| 位置 | 尺 | 内容と共通の見せ方 |
|---|---:|---|
| 63.000–75.133秒、scene02-03-01の直前 | 12.133秒 | R: 1.5の青い条件付き密度・黄色の予測線を再現。損失の谷底へ動かし、緑の条件付き平均を h(x) へ対応づける |
| 456.067–470.933秒、scene07-04-02の直後 | 14.867秒 | V07b: 3組×2入力の青い予測表→黄色の二乗偏差・面積→緑の列平均→入力間の平均。説明用の等確率例と一般の p(x) の重みを区別 |

元の546.400秒から573.400秒へ、+27.000秒（+4.941%）。計画の約12秒の二段階平均を、予測平均と二乗偏差の順序が追える14.867秒へ調整しました。
補足の例は列平均2・2、列分散8/3・2/3、全体5/3であり、本編の実験結果とは別の説明用データです。
復習の密度は1.5と同じ平均0・標準偏差0.6の例です。

## 最終検証

- `py_compile`、既存6分類の数値検証、補足の小例・音声時刻検証、指定フルレンダリングに成功（73 animations）。分解誤差最大2.081668e-17。
- 映像573.399678秒、音声573.440000秒、差0.040322秒。3秒以上の無音0件（−45dB）、平均−26.5dB、ピーク−5.8dB。
- 全125文のAPI読みを取得し、新規5文を確認。新規誤読なし。「節」の読みをspeechで「せつ」と明示。既存120文のPCMは元とバイト単位で一致。
- 全125字幕を生成して配置制限内、数式の読み仮名の残留0件。台本・manifest・読み記録が一致。
- 最終MP4の34フレームを確認（全追加場面の前後・全操作、既存8シーン）。初回目視済み画像と同一の25枚、変更後再目視9枚。平均結果の表示開始時の重なりを解消。
- 追加2場面でPCM発声区間・操作時刻・画像を照合。全62beatの音声と描画開始の最大差3.98e-11秒。全編通し聴取・全フレーム目視は未実施。
- [今回の作業レポート](../../../reports/working/20260930-0529-prml-3-2-visual-aid-recap.md)。
- [元動画の制作レポート](../../../reports/working/20260927-1836-prml-3-2-3b1b-remake.md)。
