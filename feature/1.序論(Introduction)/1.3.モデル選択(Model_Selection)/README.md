# 1.3 モデル選択

十個の点を通る曲線から始め、未知の点で選ぶ理由を実験で確かめる、約9分の日本語動画です。
Manim Community / 854×480 / 15 fps、VOICEVOX:WhiteCUL（ノーマル、speaker 23）の音声と数式字幕付き。

## シーンと原文

原文は Bishop (2006), *Pattern Recognition and Machine Learning*, §1.3、印刷 pp.32–33
（ローカルPDFの52–53ページ）。Fig.1.18 と Eq.(1.73) が対象で、この節に表はありません。

| シーン | 音声尺 | 視覚的アイデア | 原文参照 |
|---|---:|---|---|
| 点を全部通れば、選んでよい？ | 66.47秒 | 次数のつまみから曲線・残差が連動 | p.32 第1–2段落 |
| 隠していた点で、答え合わせ | 68.40秒 | 検証点を開示し、同じ図を誤差の図へ切り替える | p.32 第2段落 |
| 学ぶ・選ぶ・最後に測る | 67.80秒 | データの役割分けと、固定したモデルの最終評価 | p.32 第2段落 |
| 評価する担当を、順番に回す | 73.47秒 | 四組の役割交替、再学習、残差から各回の点数へ | pp.32–33 / Fig.1.18 |
| 一点ずつ隠すと、何が変わる？ | 61.60秒 | 分割を細かくし、隠す一点と再学習曲線が動く | p.33 / leave-one-out |
| つまみが増えると、何回学ぶ？ | 64.00秒 | 一列→格子→三層、80→240回のカウンタ | p.33 / 組合せ探索 |
| 当てはまりから、自由さを引く | 75.73秒 | 対数尤度からパラメータ数を引き、順位が変わる | p.33 / Eq.(1.73) |
| 選ぶ基準にも、限界がある | 68.93秒 | 一本の最適解から係数の不確かさへ | p.33 最終段落 |

## ファイル

- `prml_1_3_model_selection.py`: 公開クラス `PRML13ModelSelection` を維持した8シーン。
- `model_selection.py`: NumPyによる最小二乗、独立な検証・テスト、交差検証、AIC。
- `scene_support.py`: 1.1 と同じ文字高・数式本体の中心合わせ、文ごとの音声同期。
- `narration_content.py`: 字幕 `display` と音声 `speech` の正本。全122文。
- `narration_script.md`: 原文参照、全字幕・収録文、実測の beat 尺。
- `write_narration_script.py`: 正本と音声manifestから台本を再生成。
- `make_voicevox_narration.py`: 文単位合成、シーン単位保存、再開、ハッシュ照合。
- `check_narration_readings.py` / `reading_check.md` / `reading_check.json`: 全文の修正前後のAPI読み。
- `assets/voicevox/`: 8 WAV と manifest。新構成で全8本を置換。
- `verify_numerics.py`: 統計量、評価点の除外、表示範囲、音声整合の5検証。

[動画](media/videos/prml_1_3_model_selection/480p15/PRML13ModelSelection.mp4)

## 再生成

このディレクトリで実行します。既存venv、Noto Sans CJK JP、LaTeX、dvisvgm、ffmpegを使用します。
Engineは `http://127.0.0.1:50021`。接続できなければ音声生成を中止してください。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python write_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_3_model_selection.py PRML13ModelSelection
```

中断後は `make_voicevox_narration.py --from-scene scene04` のように再開できます。
話速1.08、抑揚0.95。各beat末尾の余白は約0.35〜0.42秒。
台本・WAVハッシュが一致する音声のみ採用し、不整合の場合は描画を停止します。
文ごとのPCM尺で字幕を切り替え、同じ時計で動作を実行します。
実測のシーン・beat・文・動作時刻は `media/prml13_timeline.json` に出力します（Git管理外）。
字幕内の `$...$` はMathTexで描き、数式本体の中心を日本語かなの中心へそろえます。
文字高と左右の間隔も1.1と同じ実測方式です。音声用の読み仮名は字幕に使いません。

## 実験の条件と解釈

PRMLの図や数値の複製ではありません。生成関数は sin(2πx)、独立ガウスノイズの標準偏差は0.25です。
訓練10点（seed 7）、検証24点（17）、最終テスト80点（37）。
交差検証は別の開発用24点（52）、分割用乱数seed 91を使用します。

- 多項式次数は `d` と表記します。整数の候補間だけを係数補間し、非整数の次数のモデルとは解釈しません。
- 予測曲線・残差・誤差は実計算値です。曲線や誤差を表示範囲へ切り詰めません。
- 誤差は平均二乗誤差です。最良の検証次数は3、検証誤差0.067846、選択済みモデルのテスト誤差0.082608。
- 最終テストでは選択済みの三次式のみ採点します。交差検証でもテスト点は使いません。
- 4分割は各6点。各回18点で再学習し、評価6点の誤差を算出します。
  0.163962、0.048375、0.105348、0.021510を平均し、0.084799です。
- 等しい大きさの組なので、平均した誤差は全点のout-of-fold二乗誤差の平均と一致します。
- 計算量の80・240回は全候補比較の学習回数です。選択後の全開発データによる再学習は別です。
- Eq.(1.73) は `ln p(D|w_ML) - M` の最大化。Mはパラメータ数で、多項式次数と区別します。
  既知の分散を固定したこの実験ではM=d+1。分散も推定する設定とは数え方が異なります。
- 連続値の観測の尤度は密度に基づきます。表示する対数尤度はガウス密度の対数和です。
  原文本文の記述と区別し、pそのものは尤度、ln pが対数尤度です。
- 通常のAIC=-2 ln p+2Mは最小化します。符号と定数倍が異なる同じ比較で、この例では次数3を選びます。
- 最後の曲線群は係数の不確かさの模式図です。事後分布を計算した標本ではありません。
  BICとベイズ的モデル比較は原文の予告範囲に留めています。

## 演出の参照

[3b1b/manim](https://github.com/3b1b/manim) と [3b1b/videos](https://github.com/3b1b/videos) を参照。
ManimGLコードは持ち込まず、Manim CEのValueTracker、updater、変形と色の対応で再構成しました。

- `videos/_2023/clt/main.py` の `MeanAndStandardDeviation`: 一つの量から図形・数値を同時更新。残差から評価値への対応。
- `videos/_2019/bayes/part1.py` の `ProbabilityBar` / `HeartOfBayesTheorem`: 同じ対象の役割と色を保った変形、図形から式へ進む説明順。
- `manim/manimlib/mobject/value_tracker.py` / `mobject_update_utils.py`: 状態を一箇所に置き、関連する表示を連動させる考え方。

参照コミットと最終検証値は作業レポートに記録します。

## 音声クレジット

ナレーション：**VOICEVOX:WhiteCUL**（ノーマル、speaker 23、Engine 0.25.2）。
全文のAPI読みを確認し、読みの誤りをspeech側で修正しました。これは全編の通し聴取を意味しません。

## 最終検証

全122字幕、数値・音声の5検証、キャッシュ無効の全編レンダリングを確認しました。
最終40画像（全11記号字幕を含む）を目視し、3場面の音声と動作を照合しました。
映像546.400秒、音声546.432秒、差0.032秒。3秒以上の無音0件。平均−26.6 dB、ピーク−6.6 dB。
全編の通し聴取・全フレーム目視は未実施です。
詳しい時刻と修正履歴は[作業レポート](../../../reports/working/20260925-2310-prml-1-3-3b1b-remake.md)を参照してください。
