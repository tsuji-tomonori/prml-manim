# PRML 5.4 ヘッセ行列

同じ傾きなら同じ一歩でよいか、という問いから、曲率・近似・逆行列・方向微分を追う9シーン、11分9秒の動画です。Manim Community と VOICEVOX:WhiteCUL（ノーマル、speaker 23）を使用します。

[動画（480p15）](media/videos/prml_5_4_hessian_matrix/480p15/PRML54HessianMatrix.mp4) ／ [全台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## シーン構成

| シーン | 開始秒 | 尺（秒） | 視覚的アイデア | 原文参照（印刷頁） |
|---|---:|---:|---|---|
| 1 同じ傾きなら、同じ一歩でよい？ | 0.000 | 66.933 | 同じ接線を持つ放物線の曲率を動かし、一歩の結果を比較 | pp.237–239,249、(5.28)–(5.32), (5.78) |
| 2 曲がり方を、方向ごとに測る | 66.933 | 67.933 | 等高線上で方向を回す。固有方向から鞍点の断面へ | pp.237–239,249–250、(5.28)–(5.40), Fig.5.6 |
| 3 対角だけ残すと、何が消える？ | 134.867 | 68.733 | 非対角成分のつまみと楕円が連動 | p.250、(5.79)–(5.81) |
| 4 外積近似は、何を小さいとみなす？ | 203.600 | 71.067 | 残差を縮め、赤い二階微分項の成分をゼロへ近づける | p.251、(5.82)–(5.85) |
| 5 一点の情報で、逆行列を更新する | 274.667 | 83.600 | 外積を一点ずつ加え、同じ座標上の楕円と逆行列を更新 | p.252、(5.86)–(5.89) |
| 6 少しずらして、二階微分を検算する | 358.267 | 69.133 | 四隅の中心差分、実測した丸め誤差、計算量の比較 | pp.252–253、(5.90)–(5.91) |
| 7 二つの層を、もう一度微分する | 427.400 | 84.000 | 小さなネットワークの接続から、三種類の行列成分へ | pp.253–254、(5.92)–(5.95) |
| 8 行列を作らず、方向の効果だけ知る | 511.400 | 83.200 | 勾配矢印の変化から、前向き・逆向きのR計算へ | pp.254–256、(5.96)–(5.111) |
| 9 必要な情報の量で、計算を選ぶ | 594.600 | 74.333 | 行列とベクトルの成分数、基底方向で列を復元、冒頭の問いへ | pp.249–256、§5.4 全体 |

## 視覚補足と復習（今回の追加）

元の628.466秒から668.933秒へ、40.467秒（6.439%）増えました。すべて本文図を一時置換する黄枠のカードで、字幕・タイトルを残します。

| 位置 | 尺 | 内容・共通の見せ方 |
|---|---:|---|
| scene05、286.000〜297.133秒 | 11.133秒 | 復習 R：4.4 の赤い楕円、金・青の主軸。精度の固有値1→5で標準偏差1→1/√5。A→H、Σ→H⁻¹へ対応 |
| scene07、427.400〜440.533秒 | 13.133秒 | 復習 R：5.3 の箱と配色。値を前へ、感度を後ろへ流し、微分gからHへ |
| scene08、542.867〜559.067秒 | 16.200秒 | 補足 V14b：青・紫の辺、黄色の二本の帯、薄い角の面積、緑の積の微分。y=vzの方向微分へ |

計画の3か所をすべて採用しました。積の面積は、本編の出力式と同じ `vz` に変更し、角の二次の微小量が極限で消える説明を明示しました。5.3 の復習は前後の流れと今回の記号への対応を見せるため、見積もり10秒から実測13.133秒になりました。既存の二階微分、固有方向、非対角、差分、計算量の図には重複する補足を加えていません。

既存108文の字幕・speech・文PCMは改訂前と一致しています。6シーンのWAVはそのまま保持し、scene05・07・08のみ追加音声を組み込みました。

## 原文と数値実験

Bishop (2006), *Pattern Recognition and Machine Learning*, §5.4、印刷 pp.249–256（PDF pp.269–276）を `pdftotext -layout` で抽出して全文照合しました。5.4固有の図・表はありません。Fig.5.1（p.228）の二層ネットワークと、Fig.5.6（p.239）の固有方向を補助参照し、自作の小さな例を計算して描きました。

[著者の公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)を適用しています。(5.83) の外積には転置が必要です。(5.95) の右辺は H を M に直し、j と j′ を交換します。映像では修正版を一隠れユニットへ特殊化した式を示します。

- 導入は説明用の断面 `E(s)=1+s+ks²/2`。原点の傾きは常に1、曲率は k=1〜4。一歩 s=−1 で、E は k=1 なら0.5、k=4なら2になります。
- 等高線は `H=[[3,c],[c,3]]` の二次形式から計算。c=2の固有値は1と5。正定値による局所最小判定は**勾配ゼロの停留点**で用います。ゼロ固有値だけでは判定できません。
- 対角近似で失う結びつきを同じ座標上で見せます。正確な対角成分の抽出と、再帰中の交差項を捨てる O(W) 近似を区別します。
- 非線形の数値例は `x=0.8, w=(u,v)=(0.7,1.2), z=tanh(ux), y=vz, t=0.25`。`hessian_model.py` が解析勾配・ヘッセ行列・外積と残差項を計算します。外積近似の説明で目標だけ動かすのは、残差の効果を見る実験であり訓練ではありません。
- 二値分類の (5.85) では、b は**確率になる前の総入力**の勾配です。多クラスsoftmaxにも対応する近似があり、出力間の結びつきを含む係数行列を用います。本映像は一出力の場合を説明します。
- 逆行列の初期値は `H₀=0.3I`。外積に使う3ベクトルは `(1.4,0.3), (0.2,1.3), (1,−0.8)`。整数間は次の外積を0〜1倍する演出です。更新した行列は外積和＋αI、楕円は `dᵀHd=0.3,0.6,1` の等高線です。確率質量の割合は主張しません。
- 中心差分の混合成分は四隅、対角成分は標準3点差分で検算。幅の二乗の打ち切り誤差と、小さい幅での浮動小数点の丸め誤差を区別します。計算量のグラフは漸近的な増え方の模型で、実測時間ではありません。
- ネットワーク図の a と z は、同じ隠れユニットの総入力と出力です。三ブロックの特殊化は `Hvv=z²`, `Huu=x²[(vh′)²+δvh″]`, `Huv=xh′(vz+δ)`。`δ=y−t`, 出力曲率 `M=1` です。バイアスは対応する入力を1にして扱えます。
- `hvp()` は R{a}, R{z}, R{y}, R{δ}, R{δ_h} を前後に計算し、**H を生成せず**積を求めます。原文の vᵀH と映像の Hv は、H の対称性の下で転置の関係です。動画中の出力重み v と方向ベクトル v の成分 v_u, v_v は別の役割です。
- 正確な評価とは差分幅・外積近似による誤差を導入しない意味であり、浮動小数点の丸め誤差は残ります。計算量は一例あたり、接続・活性化の固定費用を前提とします。

## ファイルと再生成

ファイル名 `prml_5_4_hessian_matrix.py` とクラス `PRML54HessianMatrix` を維持しています。

- `narration_content.py`：9シーン・57 beat・116文の display/speech。
- `export_narration_script.py`：参照・実測尺・全台本をMarkdownへ出力。
- `make_voicevox_narration.py`：文単位のPCM合成、9 WAV、manifest、ハッシュ照合、途中再開。
- `check_narration_readings.py` / `reading_check.md` / `.json`：全116文のAPI読みと修正前後。
- `video_support.py` / `scene_support.py`：1.1のかな文字高・数式本体中心・左右余白・文PCM同期を節内に継承。
- `hessian_model.py` / `verify_numerics.py` / `numerical_results.json`：数値実験と独立な検算。
- `verify_caption_layout.py` / `caption_validation.json`：全文字幕の安全領域検査。
- `verify_visual_aids.py` / `visual_aid_validation.json`：精度と幅、積の微分の極限、追加8文のPCMと動作時刻。
- `review_video.py` / `validate_video.py`：全beat・全記号字幕・同期画像、映像音声・音量・無音・時刻検査。

対象ディレクトリで実行します。Engine に接続できない場合は停止し、コンテナを操作しません。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -W error -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_5_4_hessian_matrix.py PRML54HessianMatrix
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_visual_aids.py
/home/t-tsuji/project/prml-manim/.venv/bin/python validate_video.py
/home/t-tsuji/project/prml-manim/.venv/bin/python review_video.py
```

`make_voicevox_narration.py --from-scene scene04` などで再開できます。ハッシュが一致する生成済みシーンは再生成しません。古い台本・WAV・字幕が混ざると描画を停止します。中間ファイルと抽出画像は `media/`、文音声のキャッシュは worktree 内 `.working/voicevox-lines/` に生成します。Git管理する動画は480p15の最終MP4だけです。

## 演出の参照

ManimGLのコードを持ち込まず、Manim CEで実装しています。

- [TwoGradientInterpretationsIn2D](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/nn/part2.py)：重みの位置、勾配の成分、矢印を対応させる。導入・方向走査・勾配変化へ。
- [ConstructGradientFromAllTrainingExamples](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/nn/part3.py)：一例の寄与を集めて全体の量を作る。外積の逐次追加へ。
- [network.py / backprop](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/nn/network.py)：前向きに保存した量と逆向きの感度を対応させる。R付き計算の演出へ。
- [NameEigenvectorsAndEigenvalues](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2016/eola/chapter10.py)：特別な方向と倍率を色で結ぶ。固有方向と曲率へ。
- [ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py) / [updater](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/mobject_update_utils.py)：一つの状態から数値・曲線・行列・矢印を連動させる。

1.1の字幕・同期・読み確認、3.3の精度と楕円、4.5の曲率から共分散への説明、5.3の文ごとの動作生成を参照しています。

## 音声と検証の範囲

VOICEVOX:WhiteCUL（ノーマル、speaker 23）、Engine 0.25.2、話速1.08、抑揚0.95。全116文のaudio_queryのmora列を目視確認しました。今回の追加文では「値→ネ」を「あたい」、「角→カク」を「すみ」に修正しました。読み指定を文単位に限定し、既存の「対角」の読みを維持しています。従来の「負→マケ」「縦横→ジュウオウ」への修正も維持しています。

音声の聴覚確認、全編の通し聴取、全フレームの目視、高解像度版のレンダリングは実施していません。

## 最終検証

- 全Pythonの `py_compile`、既存数値検証、追加カード検証、116字幕の安全領域検査が成功。
- 全9シーンを指定のキャッシュ無効コマンドでフルレンダリング（116 animations）。
- 映像668.932678秒、音声668.970667秒、差0.037989秒。元映像628.466011秒から40.466667秒（6.438952%）増。
- H.264 / AAC、854×480、15fps。3秒以上の無音0件、平均−26.3 dB、最大−6.2 dB。
- 40例の最大誤差：関数差分2.1811×10⁻⁸、勾配差分9.5476×10⁻¹¹、Hを作らない積4.4409×10⁻¹⁶。
- 逆行列更新の最大誤差6.6613×10⁻¹⁶。復習の楕円の等高線誤差5.5511×10⁻¹⁶。
- 積の面積差をεで割った量と方向微分との差は、ε=0.6→0.1→0.025→0.00001で0.48→0.08→0.02→0.000008へ減少。
- 字幕の最大幅9.353125（上限12.9）、最大高0.762089（上限0.9）。記号のカタカナ読みを `rg` で検索して0件。
- 追加8文すべての字幕開始と動作開始の差は1/30秒以内。PCM発声開始から発声終了まで対応動作の時間内に収まる。
- 最終動画の86画像を直接目視。全57 beat、全6数式字幕、追加3場面の前後6枚と追加8文の序盤・終盤16枚、差分の詳細1枚。確認した画像で重なり・はみ出しなし。
- 3場面でPCM時刻と前後画像を照合。曲率と楕円の幅、前後への伝播、二本の帯と極限への縮小が対応文中に動くことを確認。

[今回の作業レポート](../../../reports/working/20260930-1111-prml-5-4-visual-aid-recap.md)に判断、原文対応、検証結果、未検証範囲を記録しています。
[改訂前の制作レポート](../../../reports/working/20260929-0217-prml-5-4-3b1b-remake.md)も参照できます。
