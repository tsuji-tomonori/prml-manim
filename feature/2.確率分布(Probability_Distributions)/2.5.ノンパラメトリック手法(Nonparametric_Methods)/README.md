# 2.5 ノンパラメトリック手法

PRML Chapter 2 の 2.5 節を、ヒストグラム、カーネル密度推定、K 近傍法の「局所的に数える」発想として解説する Manim アニメーションと台本です。

## ファイル

- `prml_2_5_nonparametric_methods.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_2_5_nonparametric_methods.py PRML25NonparametricMethods
```

生成済み動画:

```text
media/videos/prml_2_5_nonparametric_methods/480p15/PRML25NonparametricMethods.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_2_5_nonparametric_methods.py PRML25NonparametricMethods
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 2.5: Nonparametric Methods
- Figure 2.24: Histogram density estimation
- Eq. (2.241): histogram density
- Eq. (2.246): local density estimate
- Section 2.5.1: Kernel density estimators
- Eq. (2.249), Eq. (2.250): kernel density estimator
- Figure 2.25: bandwidth `h` による平滑化の違い
- Section 2.5.2: Nearest-neighbour methods
- Eq. (2.253) から Eq. (2.256): KNN 分類への拡張
- Figure 2.26 から Figure 2.28: K による平滑化と分類境界

## 制作方針

- PRML の図を直接複製せず、同じ構造を自作データと自作レイアウトで再構成する。
- 密度推定の厳密な収束条件よりも、「近くの点を数え、体積で割る」という共通の直感を優先する。
- `h` と `K` は、曲線や近傍半径が動くパラメータとして見せる。
- 最後に、訓練データを保持する計算コストと、以降のより柔軟なモデルへの橋渡しを入れる。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
