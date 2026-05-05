# 2.3 ガウス分布

PRML Chapter 2 の 2.3 節を、高校数学向けに翻訳した Manim アニメーションと台本です。

## ファイル

- `prml_2_3_gaussian_distribution.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_2_3_gaussian_distribution.py PRML23GaussianDistribution
```

生成済み動画:

```text
media/videos/prml_2_3_gaussian_distribution/480p15/PRML23GaussianDistribution.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_2_3_gaussian_distribution.py PRML23GaussianDistribution
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 2.3: The Gaussian Distribution
- Section 2.3.1: Conditional Gaussian distributions
- Section 2.3.2: Marginal Gaussian distributions
- Section 2.3.3: Bayes' theorem for Gaussian variables
- Section 2.3.4: Maximum likelihood for the Gaussian
- Section 2.3.6: Bayesian inference for the Gaussian
- Section 2.3.7: Student's t-distribution
- Section 2.3.9: Mixtures of Gaussians
- Fig. 2.6 から Fig. 2.8、Fig. 2.12、Fig. 2.15、Fig. 2.16
- Eq. (2.42) から Eq. (2.64)、Eq. (2.118) から Eq. (2.124)、Eq. (2.138) から Eq. (2.143)

## 制作方針

- PRML の図を直接複製せず、同じ概念構造を自作データと自作レイアウトで再構成する。
- 一変量では平均と分散を動かし、パラメータの意味を先に視覚化する。
- 多変量では楕円、固有ベクトル、共分散行列の形を同じ画面で対応させる。
- 条件付き分布と周辺分布は、式展開を追いすぎず「ガウスのまま残る」性質を強調する。
- 最尤推定とベイズ推定は、点推定と分布推定の違いが見えるようにする。
- 後半は t 分布と混合ガウスへ橋渡しし、ガウス分布の限界も明示する。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
