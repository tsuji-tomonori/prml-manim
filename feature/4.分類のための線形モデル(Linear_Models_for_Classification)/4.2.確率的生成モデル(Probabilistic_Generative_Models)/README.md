# 4.2 確率的生成モデル

PRML Chapter 4 の 4.2 節を、クラス条件付き密度、事前確率、ベイズの定理、共有共分散ガウス、softmax、最尤推定、naive Bayes、指数型分布族への一般化として解説する Manim アニメーションと台本です。

## ファイル

- `prml_4_2_probabilistic_generative_models.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_4_2_probabilistic_generative_models.py PRML42ProbabilisticGenerativeModels
```

生成済み動画:

```text
media/videos/prml_4_2_probabilistic_generative_models/480p15/PRML42ProbabilisticGenerativeModels.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_4_2_probabilistic_generative_models.py PRML42ProbabilisticGenerativeModels
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 4.2: Probabilistic Generative Models
- Eq. (4.57)-(4.61): 二クラス posterior、log odds、logistic sigmoid
- Eq. (4.62)-(4.63): 多クラス posterior と softmax
- Eq. (4.64)-(4.70): 共有共分散ガウスから線形スコアが出る導出
- Figure 4.9: logistic sigmoid
- Figure 4.10: 共有共分散ガウスの線形 posterior
- Figure 4.11: 共有共分散と非共有共分散による線形・二次境界の違い
- Eq. (4.71)-(4.80): 共有共分散ガウスの最尤推定
- Eq. (4.81)-(4.82): binary discrete features と naive Bayes
- Eq. (4.83)-(4.86): 指数型分布族での一般化

## 制作方針

- PRML の図を直接複製せず、同じ概念構造を自作データと自作レイアウトで再構成する。
- 生成モデルを「密度を作る -> prior を掛ける -> Bayes で posterior にする」という処理の流れとして見せる。
- 共有共分散ガウスでは、二次項が相殺されて境界が線形になる点を中心に置く。
- 事前確率は重みの向きではなく bias を変え、境界を平行移動することをスライダーで示す。
- 離散特徴と指数型分布族は詳細な証明よりも、「対数を取ると線形スコアになる」直感を優先する。
- 最後に、生成モデルの密度仮定が悪いと posterior も悪くなり得るため、次節の確率的識別モデルへ橋渡しする。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
