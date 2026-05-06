# PRML 5.7 ベイズニューラルネットワーク

Christopher M. Bishop, *Pattern Recognition and Machine Learning* 5.7 "Bayesian Neural Networks" の解説動画です。

## 目的

- ニューラルネットワークの重みを一点推定ではなく確率分布として扱う考え方を説明する。
- 事前分布、尤度、事後分布、予測分布のつながりを視覚化する。
- ラプラス近似とエビデンスによる複雑さ制御の直感を示す。

## 構成

| scene | 内容 |
|---|---|
| scene01 | 点推定から重み分布への導入 |
| scene02 | 重みの事前分布 |
| scene03 | 尤度による候補重みの重み付け |
| scene04 | ベイズ則と事後分布 |
| scene05 | 予測分布と不確かさ |
| scene06 | ラプラス近似とヘッセ行列 |
| scene07 | エビデンスと複雑さ |
| scene08 | まとめ |

## 生成手順

VOICEVOX Engine が `http://127.0.0.1:50021` で起動している状態で、次を実行します。

```bash
python3 make_voicevox_narration.py
.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_5_7_bayesian_neural_networks.py PRML57BayesianNeuralNetworks
```

音声生成を途中から再開する場合:

```bash
python3 make_voicevox_narration.py --from-scene scene05
```

## 主な成果物

```text
prml_5_7_bayesian_neural_networks.py
make_voicevox_narration.py
narration_script.md
assets/voicevox/manifest.json
media/videos/prml_5_7_bayesian_neural_networks/480p15/PRML57BayesianNeuralNetworks.mp4
```

## VOICEVOX クレジット

ナレーション音声には VOICEVOX:WhiteCUL を使用します。
