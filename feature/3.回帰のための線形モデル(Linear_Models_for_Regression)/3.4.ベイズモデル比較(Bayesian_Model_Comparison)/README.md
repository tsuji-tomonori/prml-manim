# 3.4 ベイズモデル比較

PRML Chapter 3 の 3.4 節 Bayesian Model Comparison を、日本語ナレーション付きの Manim アニメーションとして構成した動画制作単位です。

## ファイル

- `prml_3_4_bayesian_model_comparison.py`: Manim アニメーション実装
- `narration_script.md`: 原文参照付きの日本語台本
- `make_voicevox_narration.py`: VOICEVOX Engine からシーン別ナレーション WAV を生成するスクリプト
- `assets/voicevox/`: 生成済みナレーション WAV と `manifest.json`

## レンダリング

Manim は `pangocairo` などの system dependency を必要とします。`uv sync` が `No package 'pangocairo' found` で失敗する場合は、OS 側で Cairo/Pango/pkg-config 関連パッケージを入れてから再実行してください。
`MathTex` の数式描画には LaTeX と `dvisvgm` も必要です。

```bash
uv run manim -pql prml_3_4_bayesian_model_comparison.py PRML34BayesianModelComparison
```

ナレーション入りで再生成する場合は、VOICEVOX Engine を `http://127.0.0.1:50021` で起動してから、先に音声を生成します。

```bash
python3 make_voicevox_narration.py
uv run manim --disable_caching --flush_cache -ql prml_3_4_bayesian_model_comparison.py PRML34BayesianModelComparison
```

生成済み動画:

```text
media/videos/prml_3_4_bayesian_model_comparison/480p15/PRML34BayesianModelComparison.mp4
```

高品質で出力する場合:

```bash
uv run manim -pqh prml_3_4_bayesian_model_comparison.py PRML34BayesianModelComparison
```

## 原文参照

主に `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` の以下を参照しています。

- Section 3.4: Bayesian Model Comparison
- Eq. (3.76): モデル事後確率
- Eq. (3.77): モデルエビデンス
- Eq. (3.80): モデル事後オッズ
- Eq. (3.81): モデル平均

## 制作方針

- PRML の図を直接複製せず、自作の多項式データと模式図で再構成する。
- `model evidence` は「エビデンス」または「周辺尤度」、`Bayes factor` は「エビデンス比」として導入する。
- エビデンスを「最大尤度だけ」ではなく「重み空間全体での平均的な説明力」として見せる。
- オッカム因子は、広すぎる重み空間のうちデータを説明する領域が小さい、という面積比の直感で表現する。
- 最後は「積分をどう近似して計算するか」という 3.5 節への橋渡しで締める。

## 音声クレジット

- ナレーション: VOICEVOX:WhiteCUL
