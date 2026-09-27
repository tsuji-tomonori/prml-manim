# PRML 4.5 ベイズロジスティック回帰

8分1秒・8シーンの日本語動画です。重みの候補の広がりが、分類の予測確率へどう伝わるかを追います。Manim Community と VOICEVOX:WhiteCUL（ノーマル、speaker 23）を使用。

[動画（480p15）](media/videos/prml_4_5_bayesian_logistic_regression/480p15/PRML45BayesianLogisticRegression.mp4) ／ [全台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## シーン

| シーン | 開始秒 | 尺（秒） | 視覚的アイデア | 原文参照 |
|---|---:|---:|---|---|
| scene01 一本の曲線を、どこまで信じる？ | 0.000 | 53.800 | 切片・傾きのつまみから候補曲線の束へ | pp.217–218 / §4.5, Eq.(4.145); sigmoid: Eq.(4.59) |
| scene02 観測は、重みの候補をどう絞る？ | 53.800 | 55.067 | 観測を加え、同じ重み軸上で分布を更新 | pp.217–218 / Eq.(4.140)–(4.142) |
| scene03 山の頂上と曲がり方を残す | 108.867 | 59.400 | 谷底の二次近似をガウスの山へ戻す | p.218 / Eq.(4.142)–(4.144); §4.4 |
| scene04 二つの重みなら、不確かさは楕円 | 168.267 | 66.800 | 重みの点と予測曲線を連動させ、楕円をたどる | p.218 / Eq.(4.140), (4.143)–(4.144) |
| scene05 重みの雲を、一本の数直線へ | 235.067 | 61.267 | 楕円の候補点をロジット軸へ写す | pp.218–219 / Eq.(4.145)–(4.150) |
| scene06 平均してから曲げる？　曲げてから平均？ | 296.333 | 61.600 | 積の面積を掃き、分散と平均確率を連動 | pp.218–219 / Eq.(4.145), (4.147), (4.151) |
| scene07 積分を、縮める係数に置き換える | 357.933 | 61.467 | シグモイドと正規CDF、近似と数値積分を比較 | pp.219–220 / Eq.(4.152)–(4.155); Fig.4.9（p.211） |
| scene08 境界は同じでも、確率は変わる | 419.400 | 61.600 | 同じ座標でMAPからベイズ予測へ変形 | p.220 / Eq.(4.153)–(4.155), §4.5.2 最終段落 |

## 原文と数値例

Bishop (2006), *Pattern Recognition and Machine Learning*, 本文 pp.217–220（PDF pp.237–240）の4.5節を抽出して参照しました。式 (4.140)–(4.155) が対象です。この節に固有の図・表はなく、Fig.4.9（p.211）を補助参照しています。デルタ関数を用いる (4.146)–(4.148) は、候補点を射影する図で説明します。

- [公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)に従い、式 (4.143) の左辺は共分散の逆行列 S_N⁻¹。Φ は標準正規CDF（inverse probit）として扱います。
- 自作の二値観測10点。入力とラベルは `bayesian_model.py` に固定し、事前は平均0、精度0.55の等方ガウス。候補の標本は seed=45 で10個。
- 場面2–3は切片を0に固定した傾き1変数のモデルです。事後密度は [-15,15] の12,001点で数値正規化。MAP=0.915592、ラプラス近似の分散=0.315006。
- 場面4以降は特徴 φ=(1,x)ᵀ、切片と傾きの2変数モデルです。MAP=(-0.514355,1.045037)、共分散は `numerical_validation.json` に保存。
- 楕円は計算した共分散から描いたガウス等密度線です。観測数の整数間は次の観測の対数尤度を連続的に導入する表示です。
- 場面6–7では平均ロジット2を固定し、分散だけを動かします。分散9でMAP確率0.880797、ガウスで平均した確率0.717424。
- 予測積分は100点のGauss–Hermite求積。独立な台形積分との差は4.53×10⁻¹⁰。κ近似の誤差は平均2、分散0〜9の37点で最大0.001711。範囲外の一様な誤差保証ではありません。
- 場面8の移行中は共分散を0倍から1倍に補間します。0.5の境界は x=0.492188。0.8の交点はMAPで1.818739、予測近似で2.159244。
- 事後分布のガウス化と、予測平均のκ近似を区別します。「データから遠ければ必ず分散が増える」という一般化はしません。

## ファイル

- `prml_4_5_bayesian_logistic_regression.py`: クラス `PRML45BayesianLogisticRegression` を維持した動画本体。
- `bayesian_model.py` / `verify_numerics.py` / `numerical_validation.json`: NumPy数値モデル、独立な検証と結果。
- `narration_content.py` / `export_narration_script.py`: 96文の display / speech と実測尺付き台本の出力。
- `scene_support.py`: 1.1の実測かな高・数式本体中心・左右余白・文PCM同期を、この節内に継承。
- `make_voicevox_narration.py` / `assets/voicevox/`: 8 WAVとmanifest。文音声のキャッシュはworktree内 `.working/voicevox-lines/`。
- `check_narration_readings.py` / `reading_check.md` / `reading_check.json`: 全文の修正前後のAPI読み。
- `verify_caption_layout.py` / `caption_validation.json`: 全字幕の寸法。
- `verify_video.py` / `video_validation.json`: 映像・音声検査、画像の時刻とハッシュ、3シーンのPCM同期。画像自体は `media/review/` に生成しGit管理しません。

## 再生成

このディレクトリで実行します。VOICEVOX Engine は http://127.0.0.1:50021 を使用。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -W error -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_4_5_bayesian_logistic_regression.py PRML45BayesianLogisticRegression
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

音声は `--from-scene scene04` などで再開できます。台本とWAVのハッシュが一致しないと描画を停止します。

## 演出の参照

- [ProbabilityBar / BayesDiagram](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2019/bayes/part1.py): 状態・確率表示・色付き数式の対応。
- [IntroduceSigmoid](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/nn/part1.py): 入力の位置とシグモイド上の確率を対応させる説明順。
- [AntiDerivative](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/gauss_int/integral.py): 走査する境界と積分面積を連動させる。
- [ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py) / [updater](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/mobject_update_utils.py): 一つの状態から、つまみ・数値・点・曲線・分布を更新。

ManimGLのコードは持ち込まず、Manim CEで実装。1.1の字幕・同期・音声ハッシュと、3.3の共分散楕円・射影の実装方針を参照しました。3.4・3.5の積分と分布の説明も参照。

## 音声

VOICEVOX:WhiteCUL（ノーマル、speaker=23）、Engine 0.25.2、話速1.08、抑揚0.95。全96文のAPI読みを確認。「黄色→オオショク」「0.8→レイテンワチ」をspeech側だけ修正し、再取得後の読みを確認しました。

全編の通し聴取、全フレームの目視、高解像度版のレンダリングは実施していません。

## 最終検証

構文・数値・96字幕の寸法検査が成功。全編レンダリングは51 animations。
映像481.000秒、音声481.024秒、差0.024秒。3秒以上の無音0件、平均−26.6 dB、最大−6.8 dB。
各シーン6枚以上、全15記号字幕を含む69画像を確認。3シーンで文PCMと操作の前後を照合しました。

[作業レポート](../../../reports/working/20260928-0238-prml-4-5-3b1b-remake.md)に検証条件・表示修正・残る制約を記録しています。
