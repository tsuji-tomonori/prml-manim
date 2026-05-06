# 4.3 確率的識別モデル

PRML Chapter 4 の 4.3 節を、事後確率を直接モデル化する考え方、固定基底関数、ロジスティック回帰、IRLS、多クラス softmax、probit、canonical link の流れとして解説する Manim アニメーションと台本です。

## ファイル

- `prml_4_3_probabilistic_discriminative_models.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_4_3_probabilistic_discriminative_models.py PRML43ProbabilisticDiscriminativeModels
```

生成済み動画:

```text
media/videos/prml_4_3_probabilistic_discriminative_models/480p15/PRML43ProbabilisticDiscriminativeModels.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_4_3_probabilistic_discriminative_models.py PRML43ProbabilisticDiscriminativeModels
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 4.3: Probabilistic Discriminative Models
- Section 4.3.1: Fixed basis functions
- Figure 4.12: 固定非線形基底関数と特徴空間の線形境界
- Section 4.3.2: Logistic regression
- Eq. (4.87) から Eq. (4.91): ロジスティック回帰、尤度、交差エントロピー、勾配
- Section 4.3.3: Iterative reweighted least squares
- Eq. (4.92) から Eq. (4.103): Newton-Raphson、Hessian、IRLS
- Section 4.3.4: Multiclass logistic regression
- Eq. (4.104) から Eq. (4.110): softmax、1-of-K 符号化、多クラス交差エントロピー
- Section 4.3.5: Probit regression
- Eq. (4.111) から Eq. (4.117): noisy threshold、probit、外れ値、ラベル反転
- Section 4.3.6: Canonical link functions
- Eq. (4.118) から Eq. (4.124): exponential family と canonical link による勾配の単純化

## 制作方針

- 4.2 の生成モデルとの対比を冒頭に置き、識別学習では `p(C|x)` を直接合わせることを示す。
- 固定基底関数では、入力空間で非線形な境界が特徴空間では直線になる様子を、点の移動で見せる。
- ロジスティック回帰は確率の曲線と決定境界を同時に見せ、分類モデルであることを明確にする。
- IRLS は式だけでなく、境界付近の点ほど重み `R = y(1-y)` が大きいことをアニメーション化する。
- 多クラスでは softmax の正規化と 1-of-K の交差エントロピーを、棒グラフと三クラス平面で示す。
- probit は noisy threshold と CDF の直感に絞り、外れ値への尾部差も比較する。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
