# 2.4 指数型分布族

PRML Chapter 2 の 2.4 節 "The Exponential Family" を、Manim と VOICEVOX で解説動画化するための制作単位です。

## ファイル

- `prml_2_4_exponential_family.py`: Manim アニメーション実装
- `narration_script.md`: 日本語ナレーション台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_2_4_exponential_family.py PRML24ExponentialFamily
```

生成済み動画:

```text
media/videos/prml_2_4_exponential_family/480p15/PRML24ExponentialFamily.mp4
```

高品質で出力する場合:

```bash
uv run manim --disable_caching --flush_cache -qh prml_2_4_exponential_family.py PRML24ExponentialFamily
```

## 内容

- 指数型分布族の標準形
- `h(x)`, `g(eta)`, `u(x)`, `eta` の役割
- ベルヌーイ分布の log-odds 表現
- 分散固定の一変量ガウス分布の自然パラメータ
- 十分統計量によるデータ要約
- 最尤推定条件と平均合わせの見方
- 共役事前分布と足し算としてのベイズ更新

## 制作方針

- 式をいきなり暗記させず、部品の役割を色分けして見せる。
- ベルヌーイとガウスの例で、いつものパラメータから自然パラメータへの見方をつなぐ。
- 十分統計量、最尤推定、共役事前分布を、同じ「足し合わせる統計量」の流れとして見せる。
- PRML の式は参照しつつ、図表は動画用に自作する。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
