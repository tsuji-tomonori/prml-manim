# 5.6 混合密度ネットワーク

PRML Chapter 5 の 5.6 節を、高校数学向けに翻訳した Manim アニメーションと台本です。

## ファイル

- `prml_5_6_mixture_density_networks.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_5_6_mixture_density_networks.py PRML56MixtureDensityNetworks
```

生成済み動画:

```text
media/videos/prml_5_6_mixture_density_networks/480p15/PRML56MixtureDensityNetworks.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_5_6_mixture_density_networks.py PRML56MixtureDensityNetworks
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 5.6: Mixture Density Networks
- Fig. 5.18: ロボットアームの逆問題
- Fig. 5.19: 順問題と逆問題に対する二乗誤差回帰
- Fig. 5.20: MDN のネットワーク構造
- Fig. 5.21: 混合係数、平均、条件付き密度、近似モード
- Eq. (5.148): 入力依存のガウス混合による `p(t|x)`
- Eq. (5.149)-(5.152): `pi_k(x)`, `sigma_k(x)`, `mu_k(x)` の制約変換
- Eq. (5.153)-(5.157): 負の対数尤度、責務、出力活性への誤差信号
- Eq. (5.158)-(5.160): 条件付き平均と分散

## 制作方針

- PRML の図を直接複製せず、同じ概念構造を自作データと自作レイアウトで再構成する。
- 逆問題の多峰性を先に見せ、一点予測がなぜ不十分かを直感化する。
- `p(t|x)` を、入力で変わる混合係数、平均、分散の組として表示する。
- ネットワークの生出力から、softmax、exp、identity で確率分布の制約を満たす流れを明示する。
- 条件付き平均と近似モードを同じ密度地図上で比較し、用途に応じた要約量選択を強調する。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
