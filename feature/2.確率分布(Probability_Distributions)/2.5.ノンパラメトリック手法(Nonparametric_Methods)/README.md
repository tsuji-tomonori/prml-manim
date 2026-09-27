# PRML 2.5 ノンパラメトリック手法

「次の測定値はどこに出る？」から始め、近くの点を数える発想をヒストグラム、カーネル密度推定、近傍分類へつなぐ日本語動画です。約9分5秒、Manim Community / VOICEVOX:WhiteCUL（ノーマル、speaker 23）。

[音声付き480p15動画](media/videos/prml_2_5_nonparametric_methods/480p15/PRML25NonparametricMethods.mp4)

## シーン構成

| シーン | 尺（秒） | 視覚的な実験 | PRML 印刷ページ・式・図 |
|---|---:|---|---|
| scene01 この点を生んだ分布は？ | 50.667 | 点→単一ガウス→二峰分布。区間確率を面積で示す | p.120 / §2.5 冒頭 |
| scene02 棒の高さより、面積を数える | 56.600 | ビン幅と区切りの位置を動かし、棒と面積を再計算 | pp.120–122 / (2.241), Fig.2.24 |
| scene03 動く窓で、局所的に数える | 68.467 | 窓の位置・幅・点数・推定密度を連動。二項分布の収束を描く | pp.122–123 / (2.242)–(2.246) |
| scene04 窓を動かす？ 点に窓を置く？ | 54.533 | 動く窓から点中心の箱へ視点を切り替える | p.123 / (2.247)–(2.249) |
| scene05 小さな山を足して、密度を作る | 61.733 | 各点の寄与を累積し、幅と合成曲線を往復させる | pp.123–124 / (2.250)–(2.252), Fig.2.25 |
| scene06 同じ個数が入るまで、窓を広げる | 63.267 | 第K近傍まで拡大する窓。実際の密度と動的な縦軸 | pp.124–125 / (2.246), Fig.2.26 |
| scene07 近い点の色で、クラスを予測する | 63.200 | 円・近傍線・票数・事後確率を連動。ベイズ則で約分 | pp.125–126 / (2.253)–(2.256), Fig.2.27(a) |
| scene08 多数決が、境界を描く | 63.267 | 実計算したK=1・5・11の分類領域を同じ平面で切替 | pp.126–127 / Fig.2.27(b), Fig.2.28 |
| scene09 柔軟さと、データを持ち続けるコスト | 63.533 | 元データへの参照、ビン数の増加、探索木の経路 | pp.121,123–124,127 / §2.5 まとめ |

## ファイル

- `prml_2_5_nonparametric_methods.py`：公開クラス `PRML25NonparametricMethods` を維持した9シーン。
- `nonparametric_model.py`：seed=2509、独立に生成した60点、20点の分類例、ヒストグラム・KDE・KNNの実計算。
- `narration_content.py` / `narration_script.md`：59 beat、118文の字幕 display と音声 speech、原文参照。
- `caption_layout.py` / `narrated_scene.py`：1.1・1.6から引き継いだ文字高・数式本体中心の調整、文PCMに基づく同期。
- `check_narration_readings.py` / `reading_check.md` / `reading_check.json`：全文のAPI読みと修正前後。
- `make_voicevox_narration.py` / `assets/voicevox/`：9 WAV、ハッシュと文IDを検査するmanifest、シーン単位の再開。
- `verify_numerics.py`：正規化、局所窓と箱カーネルの一致、第K近傍、二項分布、ベイズ則、音声整合性。
- `review_video.py`：各beat・全記号字幕・3シーンの同期画像、ffprobe、無音・音量の検査。

## 再生成

このディレクトリで実行します。VOICEVOX Engine は利用者側で起動します。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python narration_content.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_2_5_nonparametric_methods.py PRML25NonparametricMethods
/home/t-tsuji/project/prml-manim/.venv/bin/python review_video.py
```

`--from-scene scene06` などで音声生成を再開できます。台本・字幕・WAVハッシュが一致しない映像生成は停止します。文ごとの時刻は `media/prml25_timeline.json` に保存します。
.voicevox-cache、mediaの途中生成物、Tex、texts、画像、__pycache__ はGit管理外です。最終MP4だけを480p15から管理します。

## 原文との照合と説明の条件

PRML印刷pp.120–127（PDF pp.140–147）をpdftotextで読み、図2.24–2.28も画像で照合しました。本節に表はありません。図は複製せず、生成分布・観測点・分類点を自作しています。図2.28のoilデータは使用していません。

- 2006年版の式(2.243)にある指数 `1−K` は `N−K`、式(2.250)のガウス正規化係数の指数 `1/2` は一般のD次元では `D/2` に訂正。[著者の公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf) のp.10を確認しました。
- 推定値は未知の真の密度と区別するため `p̂` と表記します。幅・位置を動かす途中の値も再計算します。ヒストグラムの個数は点が境界を越えると離散的に変化します。
- ガウスのデータ生成では混合成分も独立に抽選し、点をクリップしません。このseedの60点が表示範囲内にあることを検証します。
- ヒストグラムは端のビンが短くなった場合も実際の幅で正規化します。
- KDEは全実数空間で面積1です。映像の横軸0〜1は表示範囲であり、密度の支持域の制約ではありません。
- KNN密度は半径の下限・密度の切り詰めを廃止。K=3,7,25のピークを全て示すよう縦軸の上限を連動させます。目盛りも実値を表示します。
- K=1で標本と評価点が一致すれば発散します。密度の比較ではK≥3を使い、分類ではK=1も使います。
- KNN密度は全空間の積分が発散する局所推定量です。一次元の遠方では `K/(2N|x|)` に漸近します。
- 分類は等しい誤分類損失、ユークリッド距離、同じ近傍領域、経験的クラス事前確率を用います。円の縦横の縮尺は同じです。
- 近傍数は整数です。密度のつまみは丸めた整数を計算・表示へ一貫して使います。分類領域は320×192の格子点で実計算した近似表示です。Kによる境界の単調な平滑化を一般には保証しません。
- 最近傍の誤り率の上界は独立同分布など通常の条件を仮定した無限標本極限の結果で、有限データに対する保証ではありません。証明は省略しています。
- KDE/KNNの一致性にはNを増やすだけでなく、領域を適切に縮め、局所の個数を増やす条件が必要です。収束証明は省略しています。
- 木構造による探索の改善はデータや次元に依存します。カーネル幅・近傍数の最適化や高速探索の実装は、この動画の範囲外です。

## 3Blue1Brownから取り入れた手法

[3b1b/videos](https://github.com/3b1b/videos) と [3b1b/manim](https://github.com/3b1b/manim) の演出を参照し、Manim CEで実装しました。ManimGLのコード・素材は取り込んでいません。

- `_2023/clt/main.py` / `BuildUpGaussian`, `get_variable_display`：一つのtrackerに分布・幅・数値・目印を結びつける。
- `_2023/convolutions2/continuous.py` / `TransitionToContinuousProbability`：高さから区間の面積へ視線を移す。
- `_2019/bayes/part1.py` / `CreateFormulaFromDiagram`：図の個数や領域からベイズ則を組み立てる。
- `manimlib/mobject/value_tracker.py`, `mobject_update_utils.py`：状態を一か所に置き、updaterで図を再計算する設計。

## 最終版の検証

映像545.266011秒／音声545.301333秒（差0.035322秒）。H.264＋AAC、854×480、15fps。3秒以上の無音0件、平均音量−26.5 dB、最大−6.0 dB。

py_compileと数値検証を通過。最終版の各beat59枚・全記号字幕19枚・同期比較6枚、合計84枚を画像で確認しました。全文118文のAPI読み確認、3シーンのPCMとアニメーションの時刻照合も実施。全編の通し聴取・全フレームの個別目視は未実施です。詳しい数値と抽出時刻は`validation_results.json`、制作記録は[作業レポート](<../../../reports/working/20260927-1723-prml-2-5-3b1b-remake.md>)を参照してください。

## 音声クレジット

ナレーション：VOICEVOX:WhiteCUL。全文のAPI読み確認と音響検査は、全編の通し聴取とは異なります。
