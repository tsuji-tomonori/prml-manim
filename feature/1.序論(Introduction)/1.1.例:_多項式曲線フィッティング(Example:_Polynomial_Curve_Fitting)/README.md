# 1.1 例：多項式曲線フィッティング

「10点すべてを通る曲線が、なぜ次の1点を外すのか」を追う、音声・字幕付きの Manim 動画です。
冒頭で失敗を見せ、つまみ・誤差・過学習を手がかりとして調べます。データ追加と正則化を試した後、冒頭と同じ未知の1点へ戻ります。
数値と点は自作の再現可能な実験です。ナレーションは **VOICEVOX:WhiteCUL**（ノーマル / 23）。
全編は約10分44秒（854×480 / 15 fps）、冒頭と結末のプレビューは約1分51秒（1280×720 / 30 fps）です。

## 表示と構成

- グラフと補足図の背面には、枠・台・影を配置しません。軸、目盛り、数値の周囲を空けます。
- 点・つまみの陰影と、等高線から立ち上がる誤差の谷で奥行きを表現します。正方形・棒の厚みは一定で、値は正面の面積・長さで読み取ります。
- 個数、番号、倍率は「10個」「0番」「2本」「2倍」のようにアラビア数字で表示します。二次式、三乗、最小二乗などの数学用語は漢字を使います。
- 各場面を前の疑問への実験としてつなぎます。次数を読み上げる列挙を減らし、予想、意外な結果、原因の発見へ視線を誘導します。
- 冒頭のオレンジの点と黄色い残差は最後にも登場します。調整前後の予測は同じ入力・観測で比べます。
- `$...$` 内の数式は MathTex、日本語は Noto Sans CJK JP。字幕と読み上げを独立に記述し、誤読の修正は `speech` に保持します。

## シーン

<!-- scene-table-start -->
| Scene | 題名 | 内容 |
|---:|---|---|
| 1 | 完璧な曲線の、意外な失敗 | 10点を通る九次式、未知の1点での大きな外れ、最初の謎 |
| 2 | 曲線を動かす4本のつまみ | 高さ、傾き、二乗、三乗を混ぜて曲線を作る |
| 3 | ずれ0、の落とし穴 | 打ち消し合う残差、正方形の面積、誤差の棒 |
| 4 | 答えを探すと、谷が現れる | 等高線を誤差の高さに立ち上げ、最良の直線へ |
| 5 | 点数は満点。予測は？ | 同じ図で M=0…9、訓練とテストの成績が分かれる |
| 6 | 曲線の裏側を、のぞいてみる | 対数目盛の係数、巨大な正負の項が打ち消し合う |
| 7 | 同じ九次式を、救える？ | 元の点を残し、N=10→15→40→100 |
| 8 | 10点のままで、曲線を救う | 正則化の強さを往復し、暴れと抑えすぎを観察 |
| 9 | 最初の問いに、もう一度 | 同じ未知の1点で改善を確認し、残る不確かさへ |
<!-- scene-table-end -->

正本は `narration_content.py`。全9シーン、69 beat、166文です。実測の尺と全文は [narration_script.md](narration_script.md) を参照してください。

## 再生成

この README のあるディレクトリから実行します。既存 venv、Cairo/Pango、Noto Sans CJK JP、LaTeX、dvisvgm、ffmpeg を使います。
VOICEVOX Engine 0.25.2 を `127.0.0.1:50021` で起動してから音声を生成します。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
```

全編の出力先：

```text
media/videos/prml_1_1_polynomial_curve_fitting/480p15/PRML11PolynomialCurveFitting.mp4
```

`-ql` は 854×480 / 15 fps。`-qm` は 1280×720 / 30 fps、`-qh` は高品質版です。
音声を変えた後はキャッシュを無効化して再レンダリングしてください。

### 冒頭と結末を確認する

同じ音声・字幕を使い、Scene 1 と Scene 9 を続けてレンダリングします。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --media_dir media/story-preview -qm prml_1_1_polynomial_curve_fitting.py PRML11StoryPreview
```

```text
media/story-preview/videos/prml_1_1_polynomial_curve_fitting/720p30/PRML11StoryPreview.mp4
```

誤差の谷だけを確認する場合は `PRML11DepthPreview` を使用します。
全編、物語のプレビュー、谷のプレビューのタイムラインはそれぞれ `prml11_timeline.json`、`prml11_story_timeline.json`、`prml11_depth_timeline.json` です。
並行レンダリングでは `--media_dir` を分け、TeX や一時動画の競合を避けます。

## 音声と同期

話速1.08、抑揚1.08、音量係数1.0。文ごとに合成し、PCM の実測尺に字幕と視覚操作を同期します。
各 beat の末尾だけ約0.35〜0.42秒の余白を加えます。旧台本の尺に引き延ばしません。
読みの API 結果は [reading_check.md](reading_check.md)、アクセントを含む詳細は `reading_check.json` に保存します。API の確認と実際の聴取は区別します。

`assets/voicevox/manifest.json` に台本・WAV のハッシュ、文と beat の時刻を保存します。
古い台本の音声や破損した WAV は採用しません。文キャッシュは Git 管理外の `.working/voicevox-lines/` です。

```bash
# 中断したシーンから再開
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py --from-scene scene05
# 変更したシーンだけを再生成
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py --only-scenes scene05 scene06
```

`--prepare-only` は Engine に接続せず、古い音声の無効化と manifest の準備を行います。
音声がない・ハッシュが合わないときは無音の構成確認も可能ですが、完成動画の検証では全シーンの音声の有効性を確認します。

## 数値実験と解釈

- 生成関数は sin(2πx)、ノイズ標準偏差0.25。seed と追加データの条件は `polynomial_model.py` に固定します。
- 訓練は等間隔10点。テストは別のノイズで作った100点。冒頭・結末の未知の1点は `XT[5], TT[5]` で、係数推定には使いません。Scene 5 の左図は代表10点、テスト RMS は100点全体から計算します。
- 曲線がはみ出す場合は縦軸の縮尺を連続的に広げ、目盛りも更新します。曲線の値は切り詰めません。
- M は整数、係数は M+1 個。整数の間は前後の最適係数を補間し「遷移中」と表示します。RMS の点は整数 M でだけ記録します。
- 誤差の谷の座標は `(w₀,w₁,E)`。二乗誤差そのものを高さに使います。谷底の誤差も約1.065で、床は `E=0`。外周は `E=E_min+4` です。谷底への道筋は説明用で、特定の最適化アルゴリズムの反復ではありません。
- データ追加は元の点を保持し、整数 N ごとに最小二乗を計算します。追加中の曲線は隣り合う解を補間しています。1点増えるたびに誤差が下がる保証ではありません。
- 係数の棒は `sgn(w) log₁₀(1+|w|)` の長さです。単位・材料が変われば係数の大きさも変わるため、大きさだけで過学習とは判定しません。
- 正則化は原文 Eq.(1.4) に合わせ、切片 w₀ を含む係数すべての二乗和を使います。拡大最小二乗を `numpy.linalg.lstsq` で解きます。
- 正則化シーンの λ=exp(ln λ)>0。左端の ln λ=−32 は λ=0 と同じではありません。最後の比較は冒頭の無正則化九次式から ln λ=−18 の解への補間です。
- 訓練／テスト RMS は、正則化項を含まない残差から計算します。次数と λ の実用上の選択には、テストとは別の検証データを使います。
- 最後の帯は生成関数の周りの既知ノイズ ±2σ です。学習した曲線の信頼区間を計算したものではありません。

## 原文と演出の参照

Bishop (2006), *Pattern Recognition and Machine Learning*, 印刷 pp.4–12、Fig.1.2–1.8、Eq.(1.1)–(1.4)、Table 1.1–1.2。
ローカル原文は `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf`。図と数値は原文の直接複製ではありません。

[3Blue1BrownJapan](https://www.youtube.com/@3Blue1BrownJapan) と [3Blue1Brown の勾配降下の教材](https://www.3blue1brown.com/lessons/gradient-descent)を構成の参考にしました。
自分で結果を予想し、図の変化から原因を見つける流れを、この実験に合わせて独自に組み立てています。
本文の転載や ManimGL コードのコピーは行わず、Manim Community で実装しています。

過去の検証は [奥行き演出のレポート](../../../reports/working/20261007-2313-prml-1-1-depth.md)、[視覚補足のレポート](../../../reports/working/20260930-0042-prml-1-1-visual-aid-recap.md)、[字幕サイズのレポート](../../../reports/working/20260925-2144-prml-1-1-caption-size.md) に残しています。

## 今回の検証（2026-10-08）

構文確認、既存6テスト、全9音声の台本・WAVハッシュ、166字幕の表示範囲・アラビア数字の表記を確認しました。
字幕の最大幅9.346094、最大高さ0.698750で安全領域内です。等高線が立ち上がる動作と対応文の開始差は約0.029秒です。

全編とプレビューをキャッシュ無効で再生成し、映像・音声を最後までデコードしました。
全編の映像643.600秒・音声643.626667秒、プレビューの映像110.733333秒・音声110.762667秒です。
どちらも3秒以上の無音0件。全編の平均音量−26.5 dB・最大−6.1 dB、プレビューの平均−26.6 dB・最大−7.0 dBです。

全編37枚、プレビュー11枚の代表フレームを目視しました。全フレームの目視、全編の通し聴取、高解像度の全編レンダリング、視聴者による楽しさの評価は未実施です。
詳細は [今回の作業レポート](../../../reports/working/20261008-0107-prml-1-1-story.md) を参照してください。
