# 1.4 次元の呪い

PRML Chapter 1 の 1.4 節を、Manim アニメーションと VOICEVOX ナレーションで説明する動画です。

## ファイル

- `prml_1_4_curse_of_dimensionality.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

ナレーションを再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから実行します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_1_4_curse_of_dimensionality.py PRML14CurseOfDimensionality
```

生成される低解像度動画:

```text
media/videos/prml_1_4_curse_of_dimensionality/480p15/PRML14CurseOfDimensionality.mp4
```

高品質で出力する場合:

```bash
uv run manim --disable_caching --flush_cache -pqh prml_1_4_curse_of_dimensionality.py PRML14CurseOfDimensionality
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Chapter 1 Introduction
- Section 1.4: The Curse of Dimensionality
- 高次元空間における体積、格子数、データの疎性に関する説明

## 制作方針

- PRML の図を直接複製せず、単位超立方体を前提にした自作の数値例で再構成する。
- 「次元が増えると、同じ細かさの格子数が指数的に増える」ことを最初に示す。
- `0.1^(1/D)` と `0.8^D` を、局所領域の幅と境界近くの体積として視覚化する。
- 近傍点が遠くなること、空の箱が増えることを、近傍法や表形式の学習の限界として説明する。
- 最後は、特徴選択、次元削減、正則化、モデルの仮定が汎化を支えるという流れで 1.5 節へ橋渡しする。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
