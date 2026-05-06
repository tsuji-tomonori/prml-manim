# 5.4 ヘッセ行列

PRML Chapter 5 の 5.4 節 "The Hessian Matrix" を、誤差面の曲率、固有値、近似、逆ヘッセ行列、Hessian-vector product の直感として解説する Manim アニメーションと台本です。

## ファイル

- `prml_5_4_hessian_matrix.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_5_4_hessian_matrix.py PRML54HessianMatrix
```

生成済み動画:

```text
media/videos/prml_5_4_hessian_matrix/480p15/PRML54HessianMatrix.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_5_4_hessian_matrix.py PRML54HessianMatrix
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 5.2.2: Local quadratic approximation
- Eq. (5.28) から Eq. (5.40): 局所二次近似、ヘッセ行列の固有値、正定値性
- Section 5.4: The Hessian Matrix
- Eq. (5.78): 重みに関する二階微分
- Section 5.4.1: Diagonal approximation
- Section 5.4.2: Outer product approximation
- Section 5.4.3: Inverse Hessian
- Section 5.4.4: Finite differences
- Section 5.4.5: Exact evaluation of the Hessian
- Section 5.4.6: Fast multiplication by the Hessian

## 制作方針

- PRML の図を直接複製せず、同じ構造を自作レイアウトで再構成する。
- ヘッセ行列を「二階微分の表」ではなく、誤差面の曲率を読む道具として説明する。
- 固有値と正定値性は、楕円等高線と曲率方向のアニメーションで示す。
- 対角近似と外積近似は、計算を軽くする代わりに捨てている情報があることを明示する。
- 最後に、正確な二階 backprop と Hessian-vector product を、必要な計算対象に応じて使い分ける流れへまとめる。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
