# PRML 3.3 ベイズ線形回帰

二点からどの直線を信じるかを問い、重みの分布を更新して予測分布へ進む、約9分57秒の日本語動画です。最後に等価カーネルと外挿の限界まで扱います。

[音声付き動画（480p15）](media/videos/prml_3_3_bayesian_linear_regression/480p15/PRML33BayesianLinearRegression.mp4) / [全文台本](narration_script.md) / [全文の読み確認](reading_check.md)

## 構成と原文

Bishop (2006), *Pattern Recognition and Machine Learning*, §3.3、印刷pp.152–161（PDF pp.172–181）を抽出・照合しました。図3.7–3.11は画像でも確認。本節に表はありません。

| シーン | 尺（秒） | 視覚的な操作 | 原文 |
|---|---:|---|---|
| 点が二つ。どの直線を信じる？ | 58.533 | 切片と傾きのつまみ、候補の直線 | pp.152–155、(3.48)–(3.54)、図3.7 |
| 一本の直線を、平面の一点へ | 59.000 | 係数点と直線の対応、事前精度の往復 | pp.152–154、(3.48)、(3.52)、図3.7 |
| 一点の観測は、地図をどう絞る？ | 63.933 | 尤度の帯に沿う移動、円から楕円へ更新 | pp.153–155、(3.49)–(3.54)、図3.7 |
| 点を増やすと、何が残る？ | 67.000 | 2→4→8→20点、同じ軸で分布を更新 | pp.153–155、(3.50)–(3.54)、図3.7 |
| 正則化は、事前分布の強さ | 67.133 | αとβを別々に操作、残差と係数の二乗 | pp.153,156、(3.52)–(3.56) |
| 予測の幅は、二つの揺れの和 | 71.733 | 入力を走査、関数の帯に観測分散を加える | pp.156–158、(3.57)–(3.59)、図3.8 |
| 曲線でも、同じ計算が使える | 71.067 | 9基底、1→2→4→25点、曲線全体の揺れ | pp.157–158、(3.58)–(3.59)、図3.8–3.9 |
| 予測は、観測値の重み付き和 | 70.533 | 入力とカーネルの断面を連動、負の重み | pp.159–161、(3.60)–(3.65)、図3.10–3.11 |
| 細い帯は、いつ信頼できる？ | 67.933 | 観測域の外を走査し、帯が細くなる理由を示す | pp.158,160–161、(3.59)、(3.63)、(3.65) |

## 数値実験の条件

PRMLの図を複製せず、NumPyで自作データと分布を計算しています。

- 直線: `t=0.15+0.65x+ε`、ノイズ標準偏差0.2、20点、seed 3303。事前精度α=2、観測精度β=25を基本とします。
- 曲線: `sin(2πx)+ε`、25点、入力seed 331、ノイズseed 332。中心を[0,1]へ等間隔に配置した9個のガウス基底、幅0.14。定数基底は加えていません。
- 途中の点追加は、新しい点の尤度の指数を0から1へ連続的に増やす演出です。整数のNで通常の事後分布に一致し、表示する点数は追加済みの整数です。
- 分布の等高線はマハラノビス半径0.6、1.2、1.8です。確率がその内側だけにあるという意味ではありません。シーン4では重み平面の表示範囲を拡大し、そのシーン内では同じ縮尺を使います。
- 候補の直線・曲線は実際のガウス乱数から計算します。同じ乱数を共有して更新を追いやすくします。曲線標本間の連続変形は独立な標準正規乱数の正弦・余弦混合を使用し、各時点の各曲線の周辺分布を保ちます。
- 帯は点ごとの平均±1標準偏差。緑の帯は関数の事後不確かさ、橙は新しい観測の予測です。分散を加えてから平方根を取ります。曲線全体が帯に収まる確率の主張ではありません。
- ガウス事前でのMAPは事後平均に一致し、正則化係数はλ=α/β。切片も正則化します。(3.56)は比例形とq>0を示し、規格化定数の導出は省略します。
- (3.48)は一般のガウス事前、(3.49)は事後分布です。実験と更新式は(3.52)の平均ゼロ・等方的事前を用います。βは既知として固定した計算です。βも未知の場合のStudentのt予測は結論のみ紹介します。
- 等価カーネルは訓練入力にも依存し、負にもなります。(3.64)の重み和1は定数を再現できる条件で扱います。全係数を縮める今回の事前では一般に1になりません。無正則化・定数基底・列フルランクの場合との数値比較を検証コードに含めています。
- 遠方でガウス基底がゼロへ近づくと、平均は0、予測分散はβ⁻¹へ戻ります。観測がない場所ほど常に帯が広がる、とは説明しません。

## 実装と再生成

元のファイル名 `prml_3_3_bayesian_linear_regression.py` とクラス `PRML33BayesianLinearRegression` を維持しています。

- `bayesian_model.py`: データ、事後分布、予測、標本、等価カーネル。
- `narration_content.py`: 9シーン・54 beat・108文のdisplay/speechと原文参照。
- `scene_support.py`: 1.1の文字高・数式本体中心・余白・文音声同期を節内に継承。
- `make_voicevox_narration.py`: 文単位のPCM合成、9 WAV、ハッシュ照合、途中再開。
- `check_narration_readings.py`: 全文のaudio_queryと修正前後の読みをMarkdown/JSONに保存。
- `verify_numerics.py`: 逐次更新、分散、MAP、カーネル、外挿、定数再現、音声整合。
- `review_video.py`: 全beat・全記号字幕・3シーンの同期前後画像、発声開始を抽出。

このディレクトリで実行します。VOICEVOX Engineは `http://127.0.0.1:50021` を使用します。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python narration_content.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_3_3_bayesian_linear_regression.py PRML33BayesianLinearRegression
/home/t-tsuji/project/prml-manim/.venv/bin/python review_video.py
```

`make_voicevox_narration.py --from-scene scene04` で再開できます。台本やWAVが不整合なら描画を停止します。Engineに接続できない場合は停止し、コンテナは操作しません。読みを変えた場合は再度全文確認し、音声合成後に台本を出力します。

字幕はdisplayだけから生成し、数式をMathTexで描きます。日本語かなの実測文字高に合わせ、添字を除く数式本体の中心線をそろえます。文のPCM尺と15fpsの累積境界を共用し、字幕・動作・音声のずれを防ぎます。実測時刻は `media/prml33_timeline.json`（Git管理外）へ出力します。

全文の読み確認で7文を修正しました。値→ね（2文）、行→くだり、黄色→おうしょく、積の和→せきのやわ、負→まけ、節→ふし。修正後の読みを再取得して照合しています。全編の通し聴取は実施していません。

## 3Blue1Brownから参考にした演出

ManimGLのコードを移植せず、Manim Communityで実装しています。

- [HeartOfBayesTheorem](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2019/bayes/part1.py): 観測が候補の空間を制限する図から、条件付き分布の式へ進む順序。
- [BuildUpGaussian / get_variable_display](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/clt/main.py): つまみ、数値、分布の形を同じ状態で更新する演出。
- [ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py): CEのValueTrackerとupdaterで、係数点・直線・分布・帯を連動させる。

## 実施した検証

- 全Pythonの `py_compile`、数値検証7項目、9 WAVと108文のハッシュ・時刻整合が成功。
- 逐次更新と一括更新の最大誤差 `5.00e-16`、MAPとリッジの誤差 `6.66e-16`、等価カーネルによる平均の誤差 `1.28e-14`。
- 指定の `--disable_caching --flush_cache -ql` で全編を描画。最終版は854×480、15fps、H.264/AAC、13,701,607 bytes。
- 映像596.866344秒、音声596.906667秒、差0.040323秒。`silencedetect=noise=-45dB:d=3` は0件。平均音量−26.7 dB、最大−7.2 dB。
- 全54 beat、記号字幕15場面、同期前後6枚の計75枚を抽出して確認。最終修正後は70枚が確認済み画像とバイト単位で同一、変化した5枚を再目視。各シーン6枚以上を含みます。
- 全108字幕の大きさは最大幅9.3875（上限12.9）、最大高0.7027（上限0.9）。数式と日本語の位置、重なりを抽出画像でも確認。
- シーン2・6・8でPCMの発声開始と操作区間を照合し、前後画像で分布・予測帯・カーネルの変化を確認。
- 全文の読みを再取得し、追加の誤読を発見せず。字幕データの記号読みカタカナは0件。

全編の通し聴取、全フレームの目視、高解像度レンダリングは未実施です。

## 音声クレジット

VOICEVOX:WhiteCUL（ノーマル、speaker 23）。Engine 0.25.2、話速1.08、抑揚0.95。
