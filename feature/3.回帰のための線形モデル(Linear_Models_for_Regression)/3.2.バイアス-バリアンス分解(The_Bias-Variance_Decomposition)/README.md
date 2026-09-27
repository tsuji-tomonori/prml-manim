# PRML 3.2 バイアス–バリアンス分解

同じ学習法でデータを取り直したら、予測はどれくらい変わるか。点と曲線の実験から、平均のずれ・予測の揺れ・観測ノイズの三つを分ける、9分6.4秒の日本語動画です。

[動画（480p15）](media/videos/prml_3_2_bias_variance_decomposition/480p15/PRML32BiasVarianceDecomposition.mp4) / [台本](narration_script.md) / [全文の読み確認](reading_check.md)

## 構成

| Scene | 開始 | 尺 | 視覚的アイデア | 原文参照 |
|---|---:|---:|---|---|
| scene01: 同じ方法なのに、予測が変わる？ | 0.000秒 | 45.400秒 | 同じ軸で点と学習曲線を取り替える問い | pp.147–148 / Eq.(3.38) |
| scene02: 観測の真ん中を狙う | 45.400秒 | 73.400秒 | 観測の散らばり、動く予測点、距離の二乗の面積 | p.148 / Eqs.(3.36)–(3.37) |
| scene03: データを百回取り直したら | 118.800秒 | 70.667秒 | 曲線を重ねて100本の平均へ。表示は20本 | pp.148–151 / Fig.3.5 / Eq.(3.45) |
| scene04: ずれを、二つの距離に分ける | 189.467秒 | 71.467秒 | 交点を数直線へ移し、群れの移動と拡大を分離 | p.149 / Eq.(3.40) |
| scene05: 平均を足して引くと、式も分かれる | 260.933秒 | 71.533秒 | 平均を足して引き、符号付きの差が相殺する様子 | pp.148–149 / Eqs.(3.38)–(3.40) |
| scene06: 正則化のつまみで、全体を動かす | 332.467秒 | 71.333秒 | λスライダーで曲線群・平均・B²・Vを連動 | pp.147,149–152 / Fig.3.5 / Eq.(3.27)（前節） |
| scene07: 入力全体で足すと、予測誤差になる | 403.800秒 | 80.067秒 | 入力を走査し、数値計算した誤差曲線とノイズの差へ | pp.149–151 / Eqs.(3.41)–(3.47) / Fig.3.6 |
| scene08: この平均は、実際に計算できる？ | 483.867秒 | 62.533秒 | 100組から手元の1組へ。係数の事後平均につなぐ | pp.151–152 / §3.3 への接続 |

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
- `narration_content.py`: 120文のdisplay/speech、60 beat。
- `make_voicevox_narration.py`: WhiteCUL合成、ハッシュ整合確認、シーン別再開。
- `check_narration_readings.py`, `reading_check.md`, `reading_check.json`: 修正前後のAPI読み全文。
- `assets/voicevox/`: 8 WAVとmanifest。旧音声は全8本を置換し、余剰WAVなし。
- `verify_numerics.py`: 回帰解、分解恒等式、テストMSE、描画範囲、音声・字幕整合の検証。

このディレクトリで実行:

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
OPENBLAS_NUM_THREADS=1 /home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
OPENBLAS_NUM_THREADS=1 /home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_3_2_bias_variance_decomposition.py PRML32BiasVarianceDecomposition
```

VOICEVOXは http://127.0.0.1:50021 、WhiteCUL ノーマル（speaker=23）、話速1.08・抑揚0.95。
途中再開は `make_voicevox_narration.py --from-scene scene04`。動画は音声が未生成・台本と不一致の場合に停止します。

## 参考資料

PRML (2006) §3.2、印刷pp.147–152 / PDF pp.167–172、式(3.36)–(3.47)、図3.5–3.6（節内に表なし）。
[3b1b/videos の MeanAndStandardDeviation](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/clt/main.py) の、平均線・偏差の正方形・分布の連続変形を参考にしました。
[3b1b/manim の ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py) を参照し、共有状態から図・数値を更新する考え方をManim CEで実装しています。ManimGLコードは取り込んでいません。

## 音声クレジット

ナレーション: VOICEVOX:WhiteCUL

## 最終検証

- 120文のAPI読みを修正前後で全文確認。全7記号字幕を含む57枚を最終MP4から目視。
- 指定フルレンダリング、py_compile、6分類の数値・整合検証が成功。
- 映像546.400000秒、音声546.432000秒、差0.032000秒。
- 3秒以上の無音0件（−45dB）、平均-26.5 dB、ピーク-5.7 dB。
- 3シーンでPCM発声開始・文境界・動作前後の画像を照合。全編の通し聴取は未実施。
- [作業レポート](../../../reports/working/20260927-1836-prml-3-2-3b1b-remake.md)
