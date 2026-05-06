# 1.6 情報理論

PRML Chapter 1 の 1.6 節を、高校数学から追える日本語の Manim アニメーションと台本として構成した解説動画です。

## ファイル

- `prml_1_6_information_theory.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

```bash
uv run manim -pql prml_1_6_information_theory.py PRML16InformationTheory
```

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_1_6_information_theory.py PRML16InformationTheory
```

生成済み動画:

```text
media/videos/prml_1_6_information_theory/480p15/PRML16InformationTheory.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_1_6_information_theory.py PRML16InformationTheory
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 1.6: Information Theory
- Section 1.6.1: Relative entropy and mutual information
- Eq. (1.92) から Eq. (1.121)
- Fig. 1.30

## 制作方針

- PRML の図を直接複製せず、同じ概念を自作データと自作レイアウトで再構成する。
- エントロピーを「驚きの平均」と「平均符号長」の両方から説明する。
- KL ダイバージェンスは距離ではなく、近似分布で代用したときの余分な情報量として説明する。
- 最後は相互情報量を「観測による不確かさの減少」として締める。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
