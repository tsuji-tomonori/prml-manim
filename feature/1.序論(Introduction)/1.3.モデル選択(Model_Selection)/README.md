# 1.3 モデル選択

PRML Chapter 1 の 1.3 節 Model Selection を、高校数学向けに翻訳した Manim アニメーションと台本です。

## ファイル

- `prml_1_3_model_selection.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

```bash
uv run manim -pql prml_1_3_model_selection.py PRML13ModelSelection
```

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_1_3_model_selection.py PRML13ModelSelection
```

生成済み動画:

```text
media/videos/prml_1_3_model_selection/480p15/PRML13ModelSelection.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_1_3_model_selection.py PRML13ModelSelection
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 1.3: Model Selection
- Fig. 1.18: S-fold cross-validation
- Eq. (1.73): AIC

## 制作方針

- PRML の図を直接複製せず、同じ構造を自作データと自作レイアウトで再構成する。
- `model complexity` は「モデルの複雑さ」、`validation set` は「検証データ」、`test set` は「最終評価用データ」として導入する。
- 訓練誤差だけでは over-fitting を見抜けないことを、訓練誤差と初見データ誤差の対比で見せる。
- Cross-validation は S=4 の保持ブロックを動かして、平均スコアを作る流れを見せる。
- 最後は「高次元では複雑さとデータ量の問題がさらに厳しくなる」という 1.4 節への橋渡しで締める。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL

