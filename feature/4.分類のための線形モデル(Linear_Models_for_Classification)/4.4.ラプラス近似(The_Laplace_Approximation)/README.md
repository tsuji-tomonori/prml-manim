# 4.4 ラプラス近似

PRML Chapter 4 の 4.4 節を、高校数学から大学初年級の数学へつなぐ視点で説明する Manim アニメーションと日本語ナレーションです。

## ファイル

- `prml_4_4_laplace_approximation.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_4_4_laplace_approximation.py PRML44LaplaceApproximation
```

生成済み動画:

```text
media/videos/prml_4_4_laplace_approximation/480p15/PRML44LaplaceApproximation.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_4_4_laplace_approximation.py PRML44LaplaceApproximation
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 4.4: The Laplace Approximation
- Section 4.4.1: Model comparison and BIC
- Fig. 4.14
- Eq. (4.125) から Eq. (4.139)

## 制作方針

- PRML の図を直接複製せず、同じ概念構造を自作の分布、曲線、幾何表現で再構成する。
- 1 変数では、非対称な分布のモードを探し、対数密度の二次近似からガウス近似へ変わる流れを見せる。
- 多変数では、Hessian が精度行列、逆行列が共分散になることを楕円の幅と向きで説明する。
- 証拠近似では、最大値だけでなく「幅」も効くことを示し、Occam factor と BIC へ接続する。
- 最後に、正規化定数が不要な利点と、局所・ガウス近似である限界を明示する。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
