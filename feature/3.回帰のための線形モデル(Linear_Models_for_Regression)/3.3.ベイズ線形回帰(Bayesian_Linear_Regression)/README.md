# 3.3 ベイズ線形回帰

PRML Chapter 3 の 3.3 節 Bayesian Linear Regression を、高校数学から大学初級の線形代数につなげる形で視覚化した Manim アニメーションと台本です。

## ファイル

- `prml_3_3_bayesian_linear_regression.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

```bash
uv run manim -pql prml_3_3_bayesian_linear_regression.py PRML33BayesianLinearRegression
```

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_3_3_bayesian_linear_regression.py PRML33BayesianLinearRegression
```

生成済み動画:

```text
media/videos/prml_3_3_bayesian_linear_regression/480p15/PRML33BayesianLinearRegression.mp4
```

## 制作方針

- PRML の図を直接複製せず、切片と傾きだけの自作データで重み空間と回帰グラフを対応させる。
- 重みの事前分布、尤度、事後分布を楕円と帯で見せ、ベイズ更新を直感化する。
- 逐次的にデータを追加し、重み空間の楕円と予測帯が同時に狭くなる様子を見せる。
- 予測分布では平均曲線と不確実性の帯を分け、観測ノイズと重み不確実性の二成分を説明する。
- 最後は 3.4 節 Bayesian Model Comparison への橋渡しで締める。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
