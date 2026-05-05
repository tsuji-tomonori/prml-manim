# 2.2 多項変数

PRML Chapter 2 の 2.2 節を、高校数学向けに翻訳した Manim アニメーションと台本です。

## ファイル

- `prml_2_2_multinomial_variables.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_2_2_multinomial_variables.py PRML22MultinomialVariables
```

生成済み動画:

```text
media/videos/prml_2_2_multinomial_variables/480p15/PRML22MultinomialVariables.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_2_2_multinomial_variables.py PRML22MultinomialVariables
```

## 原文参照

主に PRML Chapter 2 Section 2.2 Multinomial Variables の以下を参照しています。

- 1-of-K 表現とカテゴリ分布
- データ集合のカウント `m_k` と十分統計量
- 多項分布と最尤推定 `mu_k^ML=m_k/N`
- Dirichlet 分布と共役事前分布

## 制作方針

- PRML の図を直接複製せず、同じ構造を自作データと自作レイアウトで再構成する。
- 式だけでなく、状態カード、one-hot ベクトル、カウント棒グラフ、単体の点群で意味を見せる。
- 2.1 の beta-Bernoulli の更新から 2.2 の Dirichlet-multinomial 更新へ自然に接続し、2.3 のガウス分布へ橋渡しする。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
