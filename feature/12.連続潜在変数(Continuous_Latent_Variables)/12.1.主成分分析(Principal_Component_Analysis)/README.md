# PRML 12.1 主成分分析

## 物語の設計

全編の問いは「傾いた13点を1本の線に圧縮するとき、横軸より斜めの線を選ぶ理由は何か」。11章では複雑な分布から標本を作った。12章では多くの観測値を少ない連続潜在変数で表す。説明用の2次元点群を一貫して使い、横軸へ落とす予想から始める。横軸への平均二乗再構成誤差は約1.349、主成分へ落とすと約0.046。角度を動かして「投影後の分散を最大にする」と「直交する残差を最小にする」が同じ線を選ぶことを発見し、最後に同じ13点と値へ戻る。青は元の点、黄は投影・操作、緑は保持、紫は比較、赤は誤差。

| シーン | 直前の疑問 → 実験 → 発見 |
|---|---|
| 1 | 多次元観測を1変数へ縮められる？ → 13点を横軸へ投影 → 誤差が大きく、線の向きが問いになる |
| 2 | 何を中心に線を置く？ → 平均を動かし中心を一致させる → 平均を通る線が自然 |
| 3 | 向きをどう選ぶ？ → 角度つまみと影を同じ散布図上で動かす → 影の分散が最大の線を候補にする |
| 4 | 分散の式は図の何を数える？ → 2.3の共分散を復習し、影の二乗を図へ対応 → 単位方向の二次形式になる |
| 5 | 最大の角度はどう見つける？ → 単位円上の方向と分散曲線を連動 → 最大固有値の固有ベクトルを得る |
| 6 | 長い影を残すと誤差はどうなる？ → 同じ角度つまみで直角の残差を測る → 全分散から保持分散を引く |
| 7 | 1本から何本へ増やす？ → 再構成を M=0,1,2 と切り替える → 捨てた固有値の和が平均二乗誤差 |
| 8 | 主成分は何に使える？ → 同じ点群を保存・復元し、白色化を短く見せる → 圧縮と前処理、線形の限界 |
| 9 | 冒頭の横軸でよかった？ → 同じ13点・横軸・斜め線・誤差を再表示 → 斜め線の理由を確認し、12.2の確率的な扱いへ渡す |

## 視覚補足と章をまたぐ復習

- 2.3の共分散・ガウス楕円を使う直前、scene04の最初に復習カードRを置く。2.3 `PRML23GaussianDistribution.geometry()` の楕円と主軸の配色・動きから必要な図形だけを再現し、過去節をimportしない。実測9.6秒（約10秒）。
- 単位方向への投影と分散の二次形式は、scene04の元の点・黄色い影・式を対応させる約9秒の補足を本編の動作へ組み込む。別の待機画面は置かない。
- 固有ベクトルは、単位円上の方向を動かし、分散の極値を同時に示す。微分とラグランジュ乗数の細かい代数操作は本編に詰め込まない。
- 復習と補足の合計は約19秒で、音声493秒の約3.9%。本編尺の15%以内。

## 原文との照合

Bishop (2006) §12.1、印刷pp.561–570（PDF pp.581–590）の全文と図12.2–12.8を確認した。中心は式(12.1)–(12.6)、(12.10)–(12.20)である。[公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)ではp.564の式(12.13)後の参照、pp.565–567の数字データの対象、図12.3の固有ベクトル数、p.567の式(12.22)の標準偏差への訂正を確認した。図12.3–12.5は数字の「3」に対象が限られ、図12.3は最初の4固有ベクトルである。本編では原典の手書き数字データの値を再現せず、説明用13点で圧縮を示す。CCA、Fisher判別との詳細比較、白色化の代数的証明、高次元データの双対固有値問題(12.26)–(12.30)は扱わない。

## 数値例と近似の条件

13点の説明用データは `pca_model.py` で固定する。横座標は等間隔の13個の値に1.15を足し、縦座標はその0.72倍に固定した小さな揺れと0.35を足す。標本共分散はPRMLの式(12.3)と同じ1/Nを使う。線は平均を通し、方向は単位ベクトル。誤差は点から線への直交距離の二乗の標本平均。平均は(1.15,0.35)、第一主成分の角度は約35.388°、固有値は約3.932411と0.045573。横軸の保持分散は約2.629、誤差は約1.349。第一主成分の誤差は約0.046。M=0,1,2で誤差は約3.978, 0.046, 0。個々の点と再構成値は同じデータから計算する。丸め表示は約と明示する。白色化は2つの正の固有値に対してのみ説明する。標準化による主成分方向の変更は一般的な可能性として述べ、このデータでの変更値は示さない。

## ファイルと再生成

[完成動画（480p15）](media/videos/prml_12_1_principal_component_analysis/480p15/PRML121PrincipalComponentAnalysis.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

- `pca_model.py` / `verify_numerics.py` / `numerical_validation.json`: 13点、固有値、分散と誤差、Mごとの再構成を再計算する。
- `narration_content.py` / `narration_script.md`: 9シーン・54拍・108文の字幕 `display` と読み上げ `speech`。
- `make_voicevox_narration.py` / `assets/voicevox/`: WhiteCUL ノーマル（23）、話速1.08、抑揚1.08の9 WAVとSHA-256・文時刻付きmanifest。
- `check_narration_readings.py` / `reading_check.json` / `.md`: 全108文の `audio_query` 読み。
- `scene_support.py` / `prml_12_1_principal_component_analysis.py`: PCM時刻と同期した字幕、点群、回転する投影線、誤差と分散の図。
- `verify_caption_layout.py` / `caption_validation.json`、`verify_video.py` / `video_validation.json`: 字幕と完成映像の検査。

対象ディレクトリでメイン作業ツリーの仮想環境を使う。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_12_1_principal_component_analysis.py PRML121PrincipalComponentAnalysis
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

VOICEVOX Engineが停止中なら、`.working/voicevox-engine-0.25.2/linux-cpu-x64/run --host 127.0.0.1 --port 50021`で起動し、自分で起動したものは音声生成後に停止する。レビュー用フレームはGit管理しない。

## 検証と限界

- `py_compile`、`verify_numerics.py`、`verify_caption_layout.py`を実行した。13点の固有分解、任意の角度の保持分散と直交誤差の和、M=0,1,2の誤差、白色化後の共分散I、点と曲線の表示範囲を検査した。全108字幕の最大幅は10.582、最大高さは0.619 Manim単位で、字幕の安全範囲内。
- VOICEVOX Engine 0.25.2 の `audio_query` を全108文に実行し、結果を `reading_check.json` / `.md` に保存した。「値」の「ね」読みと数値0.046、3.98の連数字の誤読を `speech` で修正して再生成した。辞書登録はせず、自分で起動したEngineは停止した。
- `--disable_caching --flush_cache -ql`で全編をレンダリングした。`verify_video.py`はmanifestのWAV SHA-256、9シーン・54拍・108字幕の時刻、MP4のH.264/AAC、854×480・15fps・493.133秒、音声493.163秒、文単位の音声相関（最小0.999518、最大時刻ずれ0.0214秒）、3秒以上の無音0件、平均音量−26.3dB・最大−7.2dB、全編デコードを確認した。
- 代表225フレームを各拍の冒頭・途中・直後・強調動作の直後と場面末から抽出し、15枚のシートを目視した。冒頭の予想中に結果を隠す最終変更について、変更12フレームを再確認し、他の213フレームが先に目視した版とSHA-256で一致したことを確認した。グラフ背面の装飾枠、軸・目盛り・数値・注記と字幕の重なり、曲線の範囲、移動中の明るさ、英語注記を確認した。レビューフレームはGit管理外。
- 480p版の技術検査と代表フレーム目視を実施した。全編の実時間での通し視聴・試聴、全フレームの目視、1080p版のレンダリングと視聴者による理解度評価は未実施。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL
