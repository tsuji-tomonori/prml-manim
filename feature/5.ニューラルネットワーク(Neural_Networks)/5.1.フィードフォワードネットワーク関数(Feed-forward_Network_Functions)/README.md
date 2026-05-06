# 5.1 フィードフォワードネットワーク関数

PRML Chapter 5 の 5.1 節を、固定基底関数モデルから多層パーセプトロンへ拡張する流れとして解説する Manim アニメーションと台本です。

## ファイル

- `prml_5_1_feed_forward_network_functions.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_5_1_feed_forward_network_functions.py PRML51FeedForwardNetworkFunctions
```

生成済み動画:

```text
media/videos/prml_5_1_feed_forward_network_functions/480p15/PRML51FeedForwardNetworkFunctions.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_5_1_feed_forward_network_functions.py PRML51FeedForwardNetworkFunctions
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 5.1: Feed-forward Network Functions
- Eq. (5.1): 固定基底関数の線形結合
- Eq. (5.2) から Eq. (5.4): 入力から隠れユニット、隠れユニットから出力活性への変換
- Eq. (5.5) から Eq. (5.7): 出力活性化と全体のネットワーク関数
- Eq. (5.8) から Eq. (5.10): バイアスの吸収と一般的な feed-forward topology
- Figure 5.1: 2 層ネットワークの構造
- Figure 5.2: 一般的な feed-forward topology
- Figure 5.3: 多層パーセプトロンの関数近似能力
- Section 5.1.1: Weight-space symmetries

## 制作方針

- PRML の図を直接複製せず、同じ構造を自作レイアウトで再構成する。
- 固定基底関数モデルとの連続性を最初に示し、隠れユニットを「学習される基底関数」として説明する。
- feed-forward は確率的グラフィカルモデルではなく、閉路のない決定的な関数評価であることを明示する。
- 出力活性化は、回帰・複数二値分類・多クラス分類の用途別に整理する。
- 5.1.1 の重み空間対称性は、同じ入出力関数を表す重みが複数あるという直感に絞って扱う。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
