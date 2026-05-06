# 4.1 識別関数

PRML Chapter 4 の 4.1 節 "Discriminant Functions" を、線形決定境界、最小二乗分類、Fisher の線形識別、パーセプトロン更新の流れとして解説する Manim アニメーションと台本です。

## ファイル

- `prml_4_1_discriminant_functions.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_4_1_discriminant_functions.py PRML41DiscriminantFunctions
```

生成済み動画:

```text
media/videos/prml_4_1_discriminant_functions/480p15/PRML41DiscriminantFunctions.mp4
```

高品質で出力する場合:

```bash
uv run manim --disable_caching --flush_cache -qh prml_4_1_discriminant_functions.py PRML41DiscriminantFunctions
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 4.1: Discriminant Functions
- Eq. (4.4)-(4.8): 二クラス線形識別関数の幾何
- Fig. 4.1: 線形決定境界、法線、符号付き距離
- Eq. (4.9)-(4.12): 多クラス線形識別関数と凸な決定領域
- Eq. (4.15)-(4.17): 最小二乗分類の閉形式解
- Fig. 4.4-Fig. 4.5: 最小二乗分類の外れ値感度と失敗例
- Eq. (4.20)-(4.30): Fisher の線形識別と Fisher criterion
- Eq. (4.31)-(4.38): Fisher と特別な target coding の最小二乗の関係
- Eq. (4.39)-(4.51): 多クラス Fisher と最大 `K-1` 個の線形特徴
- Eq. (4.52)-(4.56): パーセプトロンモデル、criterion、更新則
- Fig. 4.7: パーセプトロン更新の図解

## 制作方針

- PRML の図を直接複製せず、同じ概念構造を自作データと自作レイアウトで再構成する。
- 線形境界の幾何を、`w`、`w0`、符号付き距離の動きとして見せる。
- 多クラス分類は、二値識別器の組み合わせではなく `argmax_k y_k(x)` の同時比較として説明する。
- 最小二乗分類は、閉形式解の利点と、分類損失としての外れ値感度を対比する。
- Fisher は「識別器そのもの」ではなく「重なりを小さくする射影方向」として説明する。
- パーセプトロンは、誤分類点だけで境界が動くオンライン更新として見せる。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
