# 1.5 決定理論

PRML Chapter 1 の 1.5 節を、高校数学向けに翻訳した Manim アニメーションと台本です。

## ファイル

- `prml_1_5_decision_theory.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_1_5_decision_theory.py PRML15DecisionTheory
```

生成済み動画:

```text
media/videos/prml_1_5_decision_theory/480p15/PRML15DecisionTheory.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_1_5_decision_theory.py PRML15DecisionTheory
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 1.5: Decision Theory
- Section 1.5.1: Minimizing the misclassification rate
- Section 1.5.2: Minimizing the expected loss
- Section 1.5.3: The reject option
- Section 1.5.4: Inference and decision
- Section 1.5.5: Loss functions for regression
- Fig. 1.24 から Fig. 1.28
- Eq. (1.77) から Eq. (1.90)

## 制作方針

- PRML の図を直接複製せず、同じ構造を自作曲線と自作レイアウトで再構成する。
- 分類では「事後確率を比べる」だけでなく、決定境界を動かして誤分類面積が変わる様子を見せる。
- 損失行列では、医療診断の例で「正解率最大」と「期待損失最小」が違う判断になり得ることを見せる。
- 棄却オプションは、不確かな入力を専門家に回す決定として説明する。
- 回帰では、二乗損失の最適予測が条件付き平均になることを、条件付き分布と予測線で見せる。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
