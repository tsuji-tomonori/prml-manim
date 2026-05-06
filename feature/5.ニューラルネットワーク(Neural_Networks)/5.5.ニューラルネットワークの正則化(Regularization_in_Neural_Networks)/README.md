# 5.5 ニューラルネットワークの正則化

PRML Chapter 5 の 5.5 節を、重み減衰、早期終了、不変性、タンジェント伝播、変換データ、畳み込みネットワーク、ソフト重み共有の流れとして解説する Manim アニメーションと台本です。

## ファイル

- `prml_5_5_regularization_in_neural_networks.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_5_5_regularization_in_neural_networks.py PRML55RegularizationInNeuralNetworks
```

生成済み動画:

```text
media/videos/prml_5_5_regularization_in_neural_networks/480p15/PRML55RegularizationInNeuralNetworks.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_5_5_regularization_in_neural_networks.py PRML55RegularizationInNeuralNetworks
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 5.5: Regularization in Neural Networks
- Figure 5.9, Figure 5.10: hidden units and local minima
- Eq. (5.112): weight decay
- Section 5.5.1: consistent Gaussian priors
- Figure 5.11: prior hyperparameters for layers, weights, and biases
- Section 5.5.2, Figure 5.12, Figure 5.13: early stopping
- Section 5.5.3: invariances and four approaches
- Section 5.5.4, Eq. (5.125) to Eq. (5.128): tangent propagation
- Section 5.5.5: training with transformed data and Tikhonov regularization
- Section 5.5.6: convolutional networks
- Section 5.5.7: soft weight sharing

## 制作方針

- PRML の図を直接複製せず、同じ構造を自作データと自作レイアウトで再構成する。
- 重み減衰は `lambda` スライダーと係数バーの縮小で、曲線の暴れが抑えられる様子として見せる。
- 早期終了は訓練誤差と検証誤差の分岐、重み空間の停止点を同時に見せる。
- 不変性は「データ拡張」「正則化項」「特徴抽出」「構造」の四分類として整理する。
- タンジェント伝播は、入力変換の接線方向と `J tau` の小ささを可視化する。
- 畳み込みネットワークは、局所受容野、重み共有、サブサンプリングを一つの流れで示す。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
