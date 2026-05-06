# 5.2 ネットワーク学習

PRML Chapter 5 の 5.2 節を、誤差関数、確率的な出力解釈、重み空間の最適化、局所二次近似、勾配情報、バッチ更新とオンライン更新として解説する Manim アニメーションと台本です。

## ファイル

- `prml_5_2_network_training.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_5_2_network_training.py PRML52NetworkTraining
```

生成済み動画:

```text
media/videos/prml_5_2_network_training/480p15/PRML52NetworkTraining.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_5_2_network_training.py PRML52NetworkTraining
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 5.2: Network Training
- Eq. (5.11) から Eq. (5.14): 回帰と二乗和誤差
- Eq. (5.18) から Eq. (5.25): 出力活性化と誤差関数の対応
- Figure 5.5: 重み空間上の誤差面、局所最小、大域最小、勾配
- Eq. (5.26) から Eq. (5.27): 停留点と反復更新
- Section 5.2.2, Figure 5.6: 局所二次近似、ヘッセ行列、楕円等高線
- Section 5.2.3: 勾配情報を使う理由
- Eq. (5.41) から Eq. (5.43): 勾配降下、バッチ法、オンライン勾配降下

## 制作方針

- PRML の図を直接複製せず、重み空間の等高線や誤差曲線を自作レイアウトで再構成する。
- 誤差逆伝播の詳細計算は 5.3 に回し、この節では「なぜ勾配が必要か」までを扱う。
- 出力活性化と誤差関数は、確率モデルから自然に対応する組として示す。
- 単純な勾配降下だけでなく、バッチ更新とオンライン更新の違いをアニメーションで比較する。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
