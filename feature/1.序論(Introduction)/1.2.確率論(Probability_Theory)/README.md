# 1.2 確率論

PRML Chapter 1 の 1.2 節を、高校数学向けに翻訳した Manim アニメーションと台本です。

## ファイル

- `prml_1_2_probability_theory.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_1_2_probability_theory.py PRML12ProbabilityTheory
```

生成済み動画:

```text
media/videos/prml_1_2_probability_theory/480p15/PRML12ProbabilityTheory.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_1_2_probability_theory.py PRML12ProbabilityTheory
```

## 原文参照

主に PRML Chapter 1 Section 1.2 Probability Theory の以下を参照しています。

- 1.2.1 Probability densities
- 和の法則、積の法則、ベイズの定理
- 期待値、分散
- 1.1 の曲線フィッティングを確率モデルとして読む導入

## 制作方針

- PRML の図を直接複製せず、同じ構造を自作データと自作レイアウトで再構成する。
- 確率を「可能性の重み」として導入し、式だけでなく棒・表・面積・重心で意味を見せる。
- 1.1 の曲線フィッティングから自然に接続し、次の最尤推定・ベイズ推論へ橋渡しする。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
