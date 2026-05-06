# 3.1 線形基底関数モデル

PRML Chapter 3 の 3.1 節 Linear Basis Function Models を、高校数学から大学初年級向けに翻訳した Manim アニメーションと台本です。

## ファイル

- `prml_3_1_linear_basis_function_models.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

```bash
uv run manim -pql prml_3_1_linear_basis_function_models.py PRML31LinearBasisFunctionModels
```

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_3_1_linear_basis_function_models.py PRML31LinearBasisFunctionModels
```

生成済み動画:

```text
media/videos/prml_3_1_linear_basis_function_models/480p15/PRML31LinearBasisFunctionModels.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_3_1_linear_basis_function_models.py PRML31LinearBasisFunctionModels
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 3.1: Linear Basis Function Models
- Eq. (3.1)-(3.3): basis functions and linearity in parameters
- Eq. (3.4)-(3.6): Gaussian and sigmoidal basis functions
- Eq. (3.7)-(3.17): Gaussian noise, least squares, design matrix, normal equations
- Eq. (3.22)-(3.23): sequential learning and LMS
- Eq. (3.24)-(3.30): regularized least squares and q regularization

## 制作方針

- PRML の図を直接複製せず、同じ構造を自作データと自作レイアウトで再構成する。
- `linear` は「入力に対して線形」ではなく「重み `w` に対して線形」であることを最初に明確にする。
- 基底関数は、固定した特徴量を作る部品として見せ、重み付き和で曲線ができる様子を動かす。
- 最小二乗は、残差の二乗和だけでなく、ガウスノイズの最尤推定としても見せる。
- 逐次学習は、一点ずつの誤差が重み更新に入る流れをアニメーションで示す。
- 正則化は、大きすぎる重みを抑えることで曲線の暴れを小さくする説明にする。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
