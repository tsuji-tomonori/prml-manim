# PRML 9.1 K平均クラスタリング

Manim と VOICEVOX:WhiteCUL（ノーマル、speaker 23、話速1.08、抑揚1.08）による約8分50秒の日本語動画。

[動画（480p15）](media/videos/prml_9_1_kmeans_clustering/480p15/PRML91KMeansClustering.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## 物語の設計

全編の問いは「14点の左側に2つとも置いた中心が、どうして左右の群へ動き、分け方が止まるのか」。8.4で隠れた配置を推論した流れから、9章ではラベルのない点を扱うと導入する。冒頭に2つの中心を左側に置き、「右側の点を誰が引き受けるか」を予想させる。最初の割当ては4点対10点で、片方の中心が右へ大きく動く。

点から中心への距離、近い中心への割当て、二乗距離の合計という順に手がかりを増やす。割当てだけを直すEステップと、中心を平均へ動かすMステップを1つの散布図で交互に見せ、ひずみが91.400→28.588→15.420→11.264と下がる。割当てが止まっても最良とは限らないため、初期値を変えた同じ14点で10.560という別の結果を比較する。最後は冒頭と同じ初期中心・同じ14点へ戻り、なぜ移動と停止が起こるかを説明し、境界付近の点を必ず1群に決めることから9.2への問いを残す。

## シーン

| シーン | 開始 / 尺（秒） | 疑問 → 実験 → 発見 | 原典 |
|---|---:|---|---|
| 1 左に2つの中心 | 0.000 / 59.467 | 右の点はどうなる？ → 4対10の予想外の割当て | pp.423–425 |
| 2 近さをどう決める | 59.467 / 56.133 | 点と中心の距離を同一図で比べ、垂直二等分線を見る | p.425、(9.2)、図9.1 |
| 3 何を小さくする | 115.600 / 60.667 | 1.1の二乗誤差を復習し、割当ての表からひずみJへ | pp.424–425、(9.1) |
| 4 まず点を選ぶ | 176.267 / 51.800 | 中心固定で各点の近い方を選び、Jが増えない | p.425、(9.2) |
| 5 中心が動く理由 | 228.067 / 62.200 | 割当て固定で平均へ動き、Jが91.400から28.588へ | p.425、(9.3)–(9.4) |
| 6 もう1周 | 290.267 / 58.533 | 再割当てと平均を交互に行い、11.264で停止 | pp.425–427、図9.1–9.2 |
| 7 止まれば最良か | 348.800 / 58.467 | 同じ14点で初期値を変え、10.560の別解を得る | p.425、p.427 |
| 8 色を少数にする | 407.267 / 58.800 | RGB画素の群分けとパレット圧縮を同じ画像で比較 | pp.428–430、図9.3 |
| 9 最初の点群へ | 466.067 / 63.667 | 左の2中心に戻り、停止と初期値の限界、硬い割当てを確認 | pp.427–428 |

## 視覚補足と章をまたぐ復習

- scene03冒頭の24.133秒で「復習: 1.1 二乗誤差」を表示し、1.1の残差を二乗する図法を、現在の散布図にある点と中心の線へ対応させる。正方形の面積、全点への線、ひずみの合計を順に示す。過去節のコードはimportしない。
- scene02の7.533秒で横と縦のずれを直角の2辺として示し、距離の二乗へ対応させる。scene05の8.000秒で赤4点の横・縦座標の平均を示す。いずれも本文の散布図と同じ画面で行う。
- 3箇所の合計39.667秒は音声実測529.733秒の7.49%。青=入力、黄=演算、緑=結果、紫=比較、赤=誤差。グラフの背面に装飾枠を置かない。

## 原文との照合

Bishop (2006), *Pattern Recognition and Machine Learning* §9.1–9.1.1、印刷pp.424–430（PDF pp.444–450）の全文を確認。図9.1–9.3をPDF画像で確認。[(9.1)–(9.6)を含む公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)には、印刷pp.424–430の訂正項目がない。中心を最も近いものに固定してから各群の平均を取ること、各段階でJが増えないこと、局所解の可能性を原文に合わせる。オンライン版(9.5)、一般距離のKメドイド(9.6)、高速化のデータ構造は本編の問いから外れるため扱わない。

## 数値例と近似の条件

- 点群は本動画用の二次元14点。初期中心は(-2.6,-1.5),(-1.6,-0.4)。ユークリッド距離の二乗で割り当てる。同距離なら番号の小さい中心を選ぶ。空群は生じない。正確な座標と履歴は数値モデルに保存する。
- 冒頭の初期割当ては4対10、J=91.400。M後28.588、次のE後15.420、M後11.264。停止時の中心は(-2.117,-1.083),(1.238,1.013)。別初期値(-2.6,-1.5),(2.4,0.8)ではJ=10.560。この比較は良い局所解を示すが、大域最小の証明ではない。
- 画像例は自作のRGB格子を使う。原典の写真は複製しない。各色成分は0–255の8ビット整数。パレットの理論ビット数は24K+N⌈log₂K⌉で、コンテナ・符号表・圧縮符号化などの費用を含まない。Kによる画質はこの例の二乗色差で測る。空間的な隣接は使わないため、物体領域の認識とは同一視しない。

## ファイルと再生成

| 種類 | ファイル |
|---|---|
| 数値 | [固定した14点とRGB画像](kmeans_model.py)、[独立検証](verify_numerics.py)、[計算結果](numerical_validation.json) |
| 台本 | [表示と読みの定義](narration_content.py)、[実測時刻付き全文](narration_script.md) |
| 音声 | [VOICEVOX生成](make_voicevox_narration.py)、[9本のWAVとmanifest](assets/voicevox/)、[全126文の読み](reading_check.md) |
| 映像 | [Manim本体](prml_9_1_kmeans_clustering.py)、[字幕とPCM同期](scene_support.py) |
| 検証 | [字幕寸法](verify_caption_layout.py)、[動画検証](verify_video.py)、[結果](video_validation.json) |

このディレクトリで実行する。worktreeに仮想環境がなければメインの `/home/t-tsuji/project/prml-manim/.venv` を使う。音声生成時には VOICEVOX Engine 0.25.2 を `localhost:50021` で起動する。自分で起動したEngineは生成後に停止し、辞書は登録しない。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_9_1_kmeans_clustering.py PRML91KMeansClustering
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

音声生成を中断した場合は `make_voicevox_narration.py --from-scene scene05` などから再開できる。台本・WAVのハッシュが合わなければManimは描画を停止する。プレビューや抽出フレームはGit管理外の `media/review/` に置く。

## 検証と限界

- `py_compile`、独立した数値再計算、126文の表示・読み・WAVハッシュ・尺の整合、全字幕の寸法検査に合格した。字幕の最大寸法は幅10.253、高さ0.619（Manim単位）。
- キャッシュを無効にして全9シーンをレンダリングした。MP4は854×480、15 fps、529.733秒。`ffprobe`、3秒以上の無音検出（0件）、音量測定（平均-26.4 dB、最大-5.1 dB）、全編デコードを実施した。126文のWAVと映像音声の相関は最小0.999649、ずれは最大0.0214秒。
- 最終MP4から261フレームを抽出し、強調・移動・色替えの途中を含む48フレームと、修正箇所の前後12フレームを目視した。枠、重なり、字幕と式の可読性、軸の目盛りとラベル、曲線・注記の配置、画像の色を確認した。抽出した全261フレームの個別目視、音声全編の通し聴取、高解像度レンダーは行っていない。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL
