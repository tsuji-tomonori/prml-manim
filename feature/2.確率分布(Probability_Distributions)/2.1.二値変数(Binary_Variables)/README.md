# PRML 2.1 二値変数

「３回とも表。次も必ず表？」から始める、約9分57秒・9シーンの日本語解説です。
観測を動かし、問いを確かめ、数式へつなぎます。Manim Community と VOICEVOX:WhiteCUL（ノーマル、23）を使用します。

[動画（音声付き480p15）](media/videos/prml_2_1_binary_variables/480p15/PRML21BinaryVariables.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## 構成

| Scene | 秒 | 視覚的アイデア | 原文の印刷ページ・式・図 |
|---|---:|---|---|
| 1 ３回とも表 | 60.067 | コイン→0と1→確率のつまみ→指数のスイッチ | pp.68–70、(2.1)–(2.2),(2.8) |
| 2 平均はどこ | 60.267 | 確率を重みとする支点、平均からの距離、分散の山 | p.69、(2.3)–(2.4) |
| 3 データを説明する確率 | 68.800 | 8観測の積から尤度、つまみで頂上へ、順序を交換 | pp.69–70、(2.5)–(2.8) |
| 4 回数だけ数える | 62.333 | 6通りの列→組合せ係数→動く二項分布 | pp.70–71、図2.1、(2.9)–(2.12) |
| 5 確率自体の不確かさ | 70.600 | 同じ軸で一様→偏り→集中→両端、区間の面積 | pp.70–72、図2.2、(2.13)–(2.16) |
| 6 表を一つ観測 | 67.733 | 各位置の高さに尤度を掛け、面積を1へ戻す | pp.71–73、図2.3、(2.17)–(2.18) |
| 7 順番に学ぶ | 61.400 | 観測・二つのカウント・曲線の同期、まとめた更新と一致 | pp.72–73、(2.18) |
| 8 次の一回 | 73.800 | 候補の確率×密度の面積、事後平均、事前と観測の重み | pp.73–74、(2.19)–(2.20) |
| 9 必ず不確かさは減る？ | 72.333 | 意外な裏で分散増加、全分散を積み上げる | p.74、(2.21)–(2.24) |

## 原文と数値の扱い

Bishop (2006), *Pattern Recognition and Machine Learning*, §2.1 / §2.1.1、印刷pp.68–74（PDF pp.88–94）を抽出して照合しました。図2.1–2.3は構造を参考にして、自作データから描いています。この節に表はありません。歴史コラムとガンマ関数の導出は省略しています。

- 自作観測は `1,0,1,1,0,1,0,1`。N=8、m=5、l=3。一定のμのもとで各試行が独立という条件です。
- ベルヌーイ分布は式(2.2)、最尤推定は(2.7)–(2.8)、事後分布は(2.18)、予測は(2.19)–(2.20)。旧READMEの式番号を訂正しました。
- 尤度の横軸はμ、二項分布の横軸は回数m。固定した順序付き観測列の尤度に、二項係数は付けません。
- 事前は Beta(3,2)、8観測後は Beta(8,5)。予測は8/13≃0.615385、最尤推定は5/8=0.625です。
- 表を1回見た更新では Beta(3,2)×μ の面積が0.6。1/0.6倍して Beta(4,2)へ正規化します。
- 両端へ寄る Beta(0.7,0.7) はμ∈[0.005,0.995]の曲線を描き、両端での発散を矢印と注記で示します。有限の高さへ切り詰めた密度としては扱いません。
- ベータ分布のa,bは正の実数。擬似観測の重みとして解釈でき、密度の指数はa−1,b−1です。結果xの分散と、未知のμの分散を区別します。
- 逐次更新の確定状態は整数カウントです。途中の連続変形は、その前後をつなぐ演出です。観測比を固定してNを増やす場面も連続補間で、実験の両端はN=8と80です。
- 3回とも表の予測4/5は、一様事前Beta(1,1)の場合です。主例のBeta(3,2)とは画面・音声で区別します。
- 不確かさの反例は Beta(9,1)→裏→Beta(9,2)。分散は0.0081818→0.0123967へ増えます。表と裏を予測確率0.9/0.1で平均した事後分散は0.00743802。事後平均の分散0.000743802を足すと元の分散に戻ります。
- 式(2.21),(2.24)は画面に示します。(2.22)–(2.23)の積分定義は、データ全体について平均するという説明へまとめました。毎回の分散減少を主張しません。

## ファイルと再生成

- `prml_2_1_binary_variables.py`：公開クラス `PRML21BinaryVariables` を維持した9シーン。
- `binary_model.py` / `verify_numerics.py`：NumPy計算、列の全列挙・数値積分・共役性・全分散・音声整合の6検証。
- `narration_content.py`：54段階、133文のdisplay / speech、原文参照の正本。実行すると台本を出力します。
- `caption_layout.py` / `narrated_scene.py`：1.1から継承した日本語/MathTexの文字高・本体中心・間隔と、文PCM同期。
- `make_voicevox_narration.py`：9 WAV、文時刻、台本/WAVハッシュを生成。`--from-scene scene06` で再開可能。
- `check_narration_readings.py` / `reading_check.md` / JSON：修正前後を保存した全文API読み。
- `review_video.py`：全54段階・全12記号字幕・4場面の同期画像を抽出。
- `validate_video.py` / `validation_results.json`：映像・音響・PCM同期の検証と最終実測値。

このディレクトリで実行します。VOICEVOX Engineは `http://127.0.0.1:50021` で稼働している必要があります。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python narration_content.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_2_1_binary_variables.py PRML21BinaryVariables
/home/t-tsuji/project/prml-manim/.venv/bin/python review_video.py
/home/t-tsuji/project/prml-manim/.venv/bin/python validate_video.py
```

音声と台本が一致しない場合は描画を停止します。各文・視覚操作の時刻は `media/prml21_timeline.json` に出力します。最終MP4以外のmedia生成物と`.working/`はGit管理しません。

## 3Blue1Brownから取り入れた演出

[3b1b/videos](https://github.com/3b1b/videos) と [3b1b/manim](https://github.com/3b1b/manim) のコードを読み、Manim CEで演出を作り直しました。ManimGLコード・素材は取り込んでいません。

- `_2020/beta/beta1.py` / `ShowBinomialFormula`：並び方を数え、同色の因子を式へ結ぶ。
- 同 `PreviewBeta`：観測が増えるたびにカウントと分布を同時に更新する。
- `_2020/beta/beta3.py` / `ShowBayesianUpdating`：各位置へ尤度を掛けて曲線を縮め、面積を1へ戻す。同じ座標系で更新を追う。
- `_2020/beta/beta2.py` / `SumToIntegral`：細かな面積の和を積分へつなぐ。
- `manimlib/mobject/value_tracker.py`：一つの状態からつまみ、数値、曲線を連動させる。

参照commit: videos `ae2b911326b8255dfae790b6e8764a8a400c5875`、manim `fafa083a4fb274bba9cabde0b6e2f50ba6da0622`。

## 読み確認と検証範囲

Engine 0.25.2、WhiteCULノーマル23、話速1.08。全133文のAPI読みを照合しました。「表→ひょう」「値→ね」「0.625→零点六百二十五」「四分→よんぷん」「五分→ごふん」「節→ふし」をspeech側で修正しています。46文のspeech表記を変更し、再取得した読みを確認しました。うち2文は正しく読めていた「表側」、1文は「二値」の読みを明示したもので、46件すべてが別の誤読を表すわけではありません。

最終動画はH.264 / AAC、854×480、15 fps。映像597.333011秒、音声597.376000秒、差0.042989秒です。3秒以上の無音（−45 dB）は0件、平均音量−26.4 dB、最大−7.3 dB。`py_compile`と6件の数値・音声整合テストが成功しました。

各シーン6段階と記号入り字幕12箇所、同期確認8枚の計74枚を抽出し、13枚の一覧画像で目視しました。scene01・03・06・08で文PCMの開始と動作区間、途中画像を照合しました。全133字幕の配置計算とdisplay内の記号読み仮名検索も実施しました。

[検証値と抽出時刻](validation_results.json) ／ [作業レポート](../../../reports/working/20260927-1449-prml-2-1-3b1b-remake.md)

全編の通し聴取、音素単位の強制アラインメント、全フレームの目視、高解像度版のレンダリングは実施していません。

ナレーション：**VOICEVOX:WhiteCUL**。
