# 4.5 ベイズロジスティック回帰

PRML Chapter 4 の 4.5 節を、ラプラス近似によるベイズロジスティック回帰として解説する Manim アニメーションと台本です。

## ファイル

- `prml_4_5_bayesian_logistic_regression.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_4_5_bayesian_logistic_regression.py PRML45BayesianLogisticRegression
```

生成済み動画:

```text
media/videos/prml_4_5_bayesian_logistic_regression/480p15/PRML45BayesianLogisticRegression.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_4_5_bayesian_logistic_regression.py PRML45BayesianLogisticRegression
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 4.5: Bayesian Logistic Regression
- Section 4.5.1: Laplace approximation
- Eq. (4.140): Gaussian prior over weights
- Eq. (4.141)-(4.142): posterior and log posterior
- Eq. (4.143)-(4.144): Gaussian posterior approximation
- Section 4.5.2: Predictive distribution
- Eq. (4.149)-(4.150): mean and variance of `a=w^T phi`
- Eq. (4.151)-(4.155): predictive distribution approximation

## 制作方針

- PRML の図を直接複製せず、自作の 2 クラスデータと自作レイアウトで再構成する。
- 重みの事後分布、MAP、ヘッセ行列、予測積分の関係を順に見せる。
- MAP 確率面とベイズ予測確率面を並べ、決定境界は同じでも不確かな領域の確率が 0.5 側へ戻ることを示す。
- 厳密な最適化手順よりも、ラプラス近似が予測分布へ入る直感を優先する。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
