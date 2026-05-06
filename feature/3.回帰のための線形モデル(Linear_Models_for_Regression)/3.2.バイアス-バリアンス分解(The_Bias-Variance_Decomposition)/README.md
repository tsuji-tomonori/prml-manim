# 3.2 バイアス-バリアンス分解

PRML Chapter 3 の 3.2 節 The Bias-Variance Decomposition を、高校数学から大学初級程度の学習者向けに翻訳した Manim アニメーションと台本です。

## ファイル

- `prml_3_2_bias_variance_decomposition.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

```bash
uv run manim -pql prml_3_2_bias_variance_decomposition.py PRML32BiasVarianceDecomposition
```

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_3_2_bias_variance_decomposition.py PRML32BiasVarianceDecomposition
```

生成済み動画:

```text
media/videos/prml_3_2_bias_variance_decomposition/480p15/PRML32BiasVarianceDecomposition.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_3_2_bias_variance_decomposition.py PRML32BiasVarianceDecomposition
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 3.2: The Bias-Variance Decomposition
- Eq. (3.36)-(3.44): squared loss から bias^2 / variance / noise への分解
- Fig. 3.5: 正則化係数と曲線群・平均予測の関係
- Fig. 3.6: bias^2、variance、sum の定量的な trade-off

## 制作方針

- PRML の図を直接複製せず、自作データ、Gaussian basis、独自配色で再構成する。
- `bias` は「平均予測と真の回帰関数 h(x) のズレ」、`variance` は「個々の訓練データ集合に対する予測の散らばり」として導入する。
- `lambda` を動かすことで、強い正則化では高 bias / 低 variance、弱い正則化では低 bias / 高 variance になる様子を見せる。
- 最後は、分解は洞察には有用だが実用上は多数の独立訓練データ集合を仮定する点に限界があり、ベイズ線形回帰へ進む流れで締める。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
