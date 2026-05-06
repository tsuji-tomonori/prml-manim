# 3.6 固定基底関数の限界

PRML Chapter 3 の 3.6 節 Limitations of Fixed Basis Functions を、日本語ナレーション付き Manim アニメーションとして構成したものです。

## ファイル

- `prml_3_6_fixed_basis_limitations.py`: Manim アニメーション実装
- `narration_script.md`: 日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_3_6_fixed_basis_limitations.py PRML36FixedBasisLimitations
```

映像だけを確認する場合は、音声生成なしでもレンダリングできます。

```bash
uv run manim -ql prml_3_6_fixed_basis_limitations.py PRML36FixedBasisLimitations
```

生成済み動画:

```text
media/videos/prml_3_6_fixed_basis_limitations/480p15/PRML36FixedBasisLimitations.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_3_6_fixed_basis_limitations.py PRML36FixedBasisLimitations
```

## 制作方針

- 固定基底の利点を先に示し、その後に限界を「幅」「高次元」「空白領域」「データ形状」の順に分解する。
- PRML の図を直接複製せず、RBF 曲線、格子、データの帯を自作して説明する。
- 最後は、特徴量を固定せずデータに合わせる発想として、カーネル法、スパースな基底選択、ニューラルネットワークへの橋渡しで締める。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
