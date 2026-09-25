# 1.2 確率論 — 面積から、予測の分布へ

約10分12秒、全10シーンの日本語解説。箱と果物の推理から始め、確率の和と積、ベイズの定理、密度、期待値・共分散、最尤推定、ベイズ曲線フィッティングまでつなげます。

[動画（480p15、音声付き）](media/videos/prml_1_2_probability_theory/480p15/PRML12ProbabilityTheory.mp4) / [原文参照付き台本](narration_script.md) / [全文の読み確認](reading_check.md)

## 構成

| Scene | 秒 | 視覚的な問い・操作 | 原文（印刷ページ） |
|---|---:|---|---|
| 1 | 45.07 | オレンジはどちらの箱から来た？ 箱の選択確率のつまみ | pp.12–13,17、Fig.1.9 |
| 2 | 72.07 | 面積の縦×横と、面積の足し合わせ | pp.13–17、Figs.1.10–1.11、式(1.5)–(1.13) |
| 3 | 64.33 | オレンジだけ残す→同じ高さの帯→全体を1に正規化 | pp.15–17、式(1.12)–(1.23)、独立 |
| 4 | 60.87 | 区間を掃く、密度を狭める、座標を伸ばして高さを下げる | pp.17–19、Fig.1.12、式(1.24)–(1.32) |
| 5 | 64.93 | 確率を動かして重心と広がりを比較、点群の共分散 | pp.19–20、式(1.33)–(1.42) |
| 6 | 62.73 | 平均・標準偏差のつまみで密度と対数尤度を連動 | pp.24–27、Figs.1.13–1.14、式(1.46)–(1.56) |
| 7 | 48.27 | 二点標本の取り直しと4,000回の推定分散の平均 | pp.27–28、Fig.1.15、式(1.57)–(1.59) |
| 8 | 57.20 | 三回表のコイン：尤度×事前→事後→次回の予測 | pp.21–24、式(1.43)–(1.45) |
| 9 | 66.60 | 入力に沿うノイズ密度、二乗誤差、事前を強めると係数が縮む | pp.28–30、Fig.1.16、式(1.60)–(1.67) |
| 10 | 70.07 | 事後標本の曲線を平均し、予測分散を二つの成分に分ける | pp.30–32、Fig.1.17、式(1.68)–(1.72) |

## 数値例と原文との対応

- 箱の例は自作です。赤=(リンゴ1,オレンジ3)、青=(リンゴ4,オレンジ1)、赤の事前確率0.3。オレンジの確率は0.365、赤の事後確率は45/73≃0.616438。
- 連続密度は区間の面積が確率です。変数変換の例は x∼Uniform(0,1), y=2x。幅2、高さ1/2で面積1を保ちます。
- 共分散ゼロと独立を区別します。左右対称な x と y=x²−1 は、共分散がゼロでも独立ではありません。
- ガウスの観測点、二点標本、回帰のノイズと事後標本は `probability_model.py` で再現できます。二点標本の最初の図は見やすい幅の3組を選択し、累積平均は選別せず4,000組を使用します。
- コインの事前は Beta(2,2)、三回表の後は Beta(5,2)。次回の表の確率は5/7。これは原文p.23の説明を具体化した自作例で、1.2でベータ分布の一般論は導出しません。
- 回帰は三次多項式、自作8点、α=0.015、β=1/0.15²。切片も含めて事前を適用し、λ=α/β。最後の帯は予測平均±予測標準偏差であり、観測ノイズだけの帯とは区別します。分散を足し、平方根を取って幅にします。
- 行列 S は係数の事後共分散です。S⁻¹=αI+βΣφ(xₙ)φ(xₙ)ᵀ を用います。PDFの式(1.72)は第2因子の添字が欠けて見えるため、§3.3の式(3.54)とも照合し、訓練入力同士の外積という整合した形で実装・数値検証しています。
- 原文は印刷pp.12–32を抽出・照合しました。この範囲に表はありません。歴史記事、bootstrapの詳細、多変量ガウスの規格化定数の導出は動画では省略しています。

## 3Blue1Brownから参考にした演出

ManimGLのコードは移植せず、Manim Communityで作り直しています。

- [BayesDiagram / BayesTheoremOnProportions](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2019/bayes/part1.py): 条件に合う面積を残し、分母を新しい全体として捉える。
- [MeanAndStandardDeviation / BuildUpGaussian](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/clt/main.py): 同じ軸の上で分布・重心・偏差・面積を動かす。
- [VariableC](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/gauss_int/herschel.py): つまみ・数値・曲線を同じ状態から更新する。
- [ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py): 連動する表示の状態を一か所に保持する。

## 再生成

このディレクトリで実行します。VOICEVOX Engineは別途 `http://127.0.0.1:50021` で稼働させます。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_2_probability_theory.py PRML12ProbabilityTheory
```

音声合成は `--from-scene scene05` などで再開できます。台本・WAVのハッシュが一致しないと動画生成を停止します。字幕は `narration_content.py` のdisplay、音声はspeechを使用。`video_support.py` は1.1の日本語/MathTexの実測文字高・本文中心の合わせ方と文PCM同期を継承しています。各文の開始・終了、動作区間は生成時に `media/prml12_timeline.json` へ保存します。

## 音声クレジットと検証範囲

ナレーション: **VOICEVOX:WhiteCUL**（ノーマル、speaker=23）。Engine 0.25.2、話速1.08。APIの全文読みを確認し、修正前後を `reading_check.md` / `reading_check.json` に保存しています。API読み確認は音声の通し聴取を意味しません。

最終版: 映像612.131秒、音声612.160秒（差0.029秒）。3秒以上の無音0件、平均−26.7dB、ピーク−6.1dB。数値5テスト、各シーン4枚・全13記号字幕を含む59枚の目視、3シーンのPCM発声・動作照合を実施。全編の通し聴取と全フレームの目視は未実施です。

[制作・検証レポート](../../../reports/working/20260925-2231-prml-1-2-3b1b-remake.md)に原文・参照コード・修正内容・実測時刻を記録しています。
