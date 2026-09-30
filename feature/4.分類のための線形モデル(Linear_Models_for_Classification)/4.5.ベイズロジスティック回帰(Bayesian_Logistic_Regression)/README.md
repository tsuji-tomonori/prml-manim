# PRML 4.5 ベイズロジスティック回帰

8分36秒・8シーンの日本語動画です。重みの候補の広がりが、分類の予測確率へどう伝わるかを追います。Manim Community と VOICEVOX:WhiteCUL（ノーマル、speaker 23）を使用。

[動画（480p15）](media/videos/prml_4_5_bayesian_logistic_regression/480p15/PRML45BayesianLogisticRegression.mp4) ／ [全台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## シーン

| シーン | 開始秒 | 尺（秒） | 視覚的アイデア | 原文参照 |
|---|---:|---:|---|---|
| scene01 一本の曲線を、どこまで信じる？ | 0.000 | 64.800 | 切片・傾きのつまみから候補曲線の束へ | pp.217–218 / §4.5, Eq.(4.145); sigmoid: Eq.(4.59) |
| scene02 観測は、重みの候補をどう絞る？ | 64.800 | 55.067 | 観測を加え、同じ重み軸上で分布を更新 | pp.217–218 / Eq.(4.140)–(4.142) |
| scene03 山の頂上と曲がり方を残す | 119.867 | 70.733 | 谷底の二次近似をガウスの山へ戻す | p.218 / Eq.(4.142)–(4.144); §4.4 |
| scene04 二つの重みなら、不確かさは楕円 | 190.600 | 78.867 | 重みの点と予測曲線を連動させ、楕円をたどる | p.218 / Eq.(4.140), (4.143)–(4.144) |
| scene05 重みの雲を、一本の数直線へ | 269.467 | 62.200 | 楕円の候補点をロジット軸へ写す | pp.218–219 / Eq.(4.145)–(4.150) |
| scene06 平均してから曲げる？　曲げてから平均？ | 331.667 | 61.600 | 積の面積を掃き、分散と平均確率を連動 | pp.218–219 / Eq.(4.145), (4.147), (4.151) |
| scene07 積分を、縮める係数に置き換える | 393.267 | 61.467 | シグモイドと正規CDF、近似と数値積分を比較 | pp.219–220 / Eq.(4.152)–(4.155); Fig.4.9（p.211） |
| scene08 境界は同じでも、確率は変わる | 454.733 | 61.600 | 同じ座標でMAPからベイズ予測へ変形 | p.220 / Eq.(4.153)–(4.155), §4.5.2 最終段落 |

## 視覚補足と復習

| 位置 | 尺 | 内容・見せ方 |
|---|---:|---|
| scene01 / 34.600–45.600秒 | 11.000秒 | 復習: 3.3。元節の候補6色と赤い平均を使い、候補の直線からS字の確率群へ。各候補を変換してから平均 |
| scene03 / 139.867–151.200秒 | 11.333秒 | 復習: 4.4。赤い二次の谷とガウス、金色の幅。曲率1→4で分散1→0.25、記号AからHへ |
| scene04 / 243.800–255.867秒 | 12.067秒 | 補足 V08d。青い列×紫の行を黄色の4セルへ。係数0.25を掛けた緑の精度を、紫の事前精度に加えて逆行列へ |
| scene05 / ガウス密度の出現時 | 独立カードなし | 2.3の線形変換を字幕・音声で参照。表示中は入力つまみを退避 |

3カードは計34.400秒。2.3の参照を既存文へ織り込んだ分は+0.933秒です。元映像481.000秒から516.333秒（+7.346%）。補足は本文を一時置換する黄枠カードに統一し、小窓への詰め込みを避けました。

3.3の候補線は説明用の6組へ縮約し、元の候補色と平均の赤を維持しています。4.4の曲率は正の一変数例です。行列の例は今回の学習結果とは別の説明用数値で、加算するのは共分散ではなく精度です。

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
- `narration_content.py` / `export_narration_script.py`: 103文の display / speech と実測尺付き台本の出力。
- `scene_support.py`: 1.1の実測かな高・数式本体中心・左右余白・文PCM同期を、この節内に継承。
- `make_voicevox_narration.py` / `assets/voicevox/`: 8 WAVとmanifest。文音声のキャッシュはworktree内 `.working/voicevox-lines/`。
- `check_narration_readings.py` / `reading_check.md` / `reading_check.json`: 全文の修正前後のAPI読み。
- `verify_caption_layout.py` / `caption_validation.json`: 全字幕の寸法。
- `verify_video.py` / `video_validation.json`: 映像・音声検査、画像の時刻とハッシュ、既存3箇所・追加2箇所のPCM同期。画像自体は `media/review/` に生成しGit管理しません。

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

音声は変更シーンだけ生成し、既存manifestでハッシュを確認した変更前WAVから未変更文のPCMを再利用します。`--from-scene scene04` などで再開できます。台本とWAVのハッシュが一致しないと描画を停止します。

## 演出の参照

- [ProbabilityBar / BayesDiagram](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2019/bayes/part1.py): 状態・確率表示・色付き数式の対応。
- [IntroduceSigmoid](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/nn/part1.py): 入力の位置とシグモイド上の確率を対応させる説明順。
- [AntiDerivative](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/gauss_int/integral.py): 走査する境界と積分面積を連動させる。
- [ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py) / [updater](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/mobject_update_utils.py): 一つの状態から、つまみ・数値・点・曲線・分布を更新。

ManimGLのコードは持ち込まず、Manim CEで実装。1.1の字幕・同期・音声ハッシュと、3.3の共分散楕円・射影の実装方針を参照しました。3.4・3.5の積分と分布の説明も参照。

## 音声

VOICEVOX:WhiteCUL（ノーマル、speaker=23）、Engine 0.25.2、話速1.08、抑揚0.95。全103文のAPI読みを取得し、追加・変更した8文を目視確認。既存の「黄色→オオショク」「0.8→レイテンワチ」の修正を保持。今回の追加文は「縦横→ジュウオウ」をspeech側で「たてよこ」に直し、APIで再確認しました。変更のない95文は元のPCMとバイト単位で一致し、scene02・06・07・08はWAV全体が一致しています。

全編の通し聴取、全フレームの目視、高解像度版のレンダリングは実施していません。

## 最終検証

構文・既存数値・追加例・103字幕の寸法検査が成功。最終全編レンダリングは64 animations。
映像516.333011秒、音声516.373333秒、差0.040322秒。3秒以上の無音0件、平均−26.6 dB、最大−6.8 dB。
字幕の最大幅9.37890625、最大高0.69953125。全8シーンと追加・変更場面の前後・途中を含む99画像を確認。最終出力では同一88枚をハッシュ照合し、差分11枚を再目視しました。
追加2箇所・既存3箇所の文PCMと操作を照合。新規動作開始と文開始の差は5.334 ms、2.666 msで1フレーム以内です。

[今回の作業レポート](../../../reports/working/20260930-0933-prml-4-5-visual-aid-recap.md)に採否・変更理由・検証範囲を記録しています。
