# 5.3 誤差逆伝播

PRML Chapter 5 の 5.3 節を、誤差逆伝播を「重み更新法」ではなく「勾配評価のための局所的なメッセージパッシング」として解説する Manim アニメーションと台本です。

## ファイル

- `prml_5_3_error_backpropagation.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_5_3_error_backpropagation.py PRML53ErrorBackpropagation
```

生成済み動画:

```text
media/videos/prml_5_3_error_backpropagation/480p15/PRML53ErrorBackpropagation.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_5_3_error_backpropagation.py PRML53ErrorBackpropagation
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 5.3: Error Backpropagation
- Section 5.3.1: Evaluation of error-function derivatives
- Eq. (5.48), Eq. (5.49): forward propagation の局所計算
- Eq. (5.50)-(5.53): `partial E_n / partial w_ji = delta_j z_i`
- Eq. (5.54): 出力ユニットのデルタ
- Eq. (5.55), Eq. (5.56): 隠れユニットの backpropagation formula
- Fig. 5.7: 隠れユニットへのデルタ逆伝播
- Section 5.3.2: `tanh` 隠れ層を持つ単純な例
- Eq. (5.65)-(5.67): 出力デルタ、隠れデルタ、各層の重み微分
- Section 5.3.3: backpropagation の計算効率と有限差分による実装確認

## 制作方針

- backpropagation を最適化法全体ではなく、勾配評価の手順として説明する。
- `delta_j` の定義、重み微分の局所積、隠れユニットの逆向き集約を順に見せる。
- 数式は画面に置き、ナレーションは式の意味を補足する。
- PRML の図を直接複製せず、同じ構造を自作ネットワーク図と自作レイアウトで再構成する。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
