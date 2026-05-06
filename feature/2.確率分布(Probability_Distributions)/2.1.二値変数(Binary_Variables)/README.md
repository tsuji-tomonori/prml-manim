# 2.1 二値変数

PRML Chapter 2, Section 2.1 Binary Variables を、日本語ナレーション付きの Manim 解説動画として構成したものです。

## ファイル

- `prml_2_1_binary_variables.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_2_1_binary_variables.py PRML21BinaryVariables
```

生成済み動画:

```text
media/videos/prml_2_1_binary_variables/480p15/PRML21BinaryVariables.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_2_1_binary_variables.py PRML21BinaryVariables
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照する構成です。

- Section 2.1: Binary Variables
- Bernoulli distribution, Eq. (2.1)
- Maximum likelihood estimate for Bernoulli parameter, Eq. (2.2)
- Beta distribution and conjugacy, Eq. (2.13) to Eq. (2.17)
- Sequential Bayesian updating and predictive distribution, Eq. (2.18)

## 制作方針

- 二値変数を「1 か 0」の具体例から入り、Bernoulli 分布の式へ接続する。
- 最尤推定は、観測列から尤度曲線が立ち上がり、山が `m / N` に来る様子で説明する。
- Beta 分布は、`a` と `b` を疑似カウントとして見せ、観測によって曲線が逐次更新される形で表現する。
- 最後は Bayesian predictive distribution を、次の観測が 1 になる確率として説明し、2.2 の多項変数へ橋渡しする。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
