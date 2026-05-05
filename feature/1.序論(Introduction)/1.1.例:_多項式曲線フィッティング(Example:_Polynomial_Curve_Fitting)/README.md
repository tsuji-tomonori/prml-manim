# 1.1 例: 多項式曲線フィッティング

PRML Chapter 1 冒頭から 1.1 節を、高校数学向けに翻訳した Manim アニメーションと台本です。

## ファイル

- `prml_1_1_polynomial_curve_fitting.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

```bash
uv run manim -pql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
```

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
```

生成済み動画:

```text
media/videos/prml_1_1_polynomial_curve_fitting/480p15/PRML11PolynomialCurveFitting.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Chapter 1 Introduction 冒頭: pattern recognition, handwritten digit, training/test/generalization
- Section 1.1: Polynomial Curve Fitting
- Fig. 1.1 から Fig. 1.8
- Eq. (1.1) から Eq. (1.4)
- Table 1.1, Table 1.2

## 制作方針

- PRML の図を直接複製せず、同じ構造を自作データと自作レイアウトで再構成する。
- 行列、偏微分、尤度、確率密度はこの回では主役にしない。
- `training set` は「練習問題」、`test set` は「初見問題」、`over-fitting` は「練習問題の丸暗記」として導入する。
- 最後は「不確かさを扱うために確率論へ進む」という 1.2 節への橋渡しで締める。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
