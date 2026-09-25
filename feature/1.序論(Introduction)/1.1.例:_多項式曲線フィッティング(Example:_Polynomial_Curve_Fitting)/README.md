# 1.1 例：多項式曲線フィッティング

「この点を生んだ曲線は？」から始める、Manim Community 版の日本語教材です。
係数・残差・面積・誤差・曲線を連動させ、同じデータの上で予測を観察します。
低品質版は約 **10分（854×480 / 15 fps）**。**WhiteCUL の音声と数式記号の字幕付き** です。

## シーン

| Scene | 内容 | 目安 | 見るもの |
|---:|---|---:|---|
| 1 | この点を生んだ曲線は？ | 51.7秒 | 点だけの問い、隠れた sin、未知の観測 |
| 2 | 多項式はつまみの集まり | 67.5秒 | w₀…w₃ の縦スライダーと式の色分け |
| 3 | ずれを面積に変える | 79.7秒 | 符号付き和／二乗和、残差→正方形→E の棒 |
| 4 | 二つのつまみで谷底へ | 56.1秒 | 誤差の等高線、係数の点、直線が連動 |
| 5 | 点を通れば予測もよくなる？ | 105.8秒 | M=0…9、残差から訓練／テスト RMS を点打ち |
| 6 | 曲線の裏側で係数が膨らむ | 55.7秒 | 正負の係数を対数の棒で比較 |
| 7 | 点が増えると同じ九次式は？ | 50.1秒 | M=9、N=10→15→40→100 を同じ軸で比較 |
| 8 | 係数をゼロへ引くバネ | 83.4秒 | ln λ を往復、曲線・棒・RMS が連動 |
| 9 | 一本の線のその先へ | 50.1秒 | 同じ入力の繰り返し観測、既知ノイズの帯 |

台本の各文に字幕用 `display` と読み上げ用 `speech` を持たせ、各 beat に実測尺と原文参照を記録します。
正本の `narration_content.py` を変えた場合は `narration_script.md` も合わせて更新してください。

## ファイル

- `prml_1_1_polynomial_curve_fitting.py`: CE の描画。公開クラス名は従来と同じ `PRML11PolynomialCurveFitting`。
- `polynomial_model.py`: NumPy によるデータ生成、最小二乗、正則化、RMS。
- `narration_content.py` / `narration_script.md`: 9 シーンの日本語台本・字幕・beat 情報。
- `make_voicevox_narration.py`: WhiteCUL の音声生成。`SCENES` は共通台本から読み込み。
- `assets/voicevox/manifest.json`: 台本と WAV のハッシュ、生成状態、beat・文ごとの実測時刻。
- `check_narration_readings.py` / `reading_check.md` / `reading_check.json`: 全文の audio_query と修正前後の読み。
- `verify_numerics.py`: 数式と数値の整合、音声のハッシュ・余白・再開処理の確認。
- `media/prml11_timeline.json`: レンダリング実測のシーン・beat 境界（生成物、Git 管理外）。

動画：

```text
media/videos/prml_1_1_polynomial_curve_fitting/480p15/PRML11PolynomialCurveFitting.mp4
```

## レンダリング

この README のあるディレクトリで実行します。既存 venv を使用します。
Cairo/Pango、Noto Sans CJK JP、LaTeX、dvisvgm、ffmpeg が必要です。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
```

従来の再生コマンドも使用できます：

```bash
uv run manim -pql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
```

高品質版は `-ql` を `-qh` に変更します。

## 音声

2026-09-25 に Engine 0.25.2 と WhiteCUL ノーマル（speaker ID 23）の疎通を確認し、全9 WAV を生成しました。
話速 1.08、抑揚 0.95、音量係数 1.0。音声を文単位で合成し、字幕の切り替えと動作を PCM の長さに同期します。
各 beat の末尾だけ約0.35〜0.42秒の余白を加えます。旧無音版の最低尺には引き延ばしません。
字幕は `display`、音声は対応する `speech` を使います。字幕内の `$...$` を MathTex で描き、日本語の漢字は保持します。
例：字幕 `$w_0$`／音声「ダブリューゼロ」、字幕「値」／音声「あたい」。
160文すべての読みを audio_query で取得し、全文を確認しました。記録と再確認方法は [reading_check.md](reading_check.md) を参照してください。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
```

このコマンドは API の読みを再取得し、初回の修正前データを保持します。生成された一覧の目視確認も行ってください。
ユーザー辞書への登録は行わず、読みの修正は `speech` 内に保持します。

Engine の稼働を確認して再生成する場合：

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
```

生成中断時は `--from-scene scene05` のように再開できます。
各シーンを保存するたびに manifest を更新します。生成済みで整合する前半の WAV は保持します。
文音声のキャッシュは worktree の `.working/voicevox-lines/` に保存し、同じ設定・文は再利用します。
各 beat の動作開始・終了、文字幕の開始・終了は `media/prml11_timeline.json` に記録します。
単に WAV ファイル名が一致するだけでは、音声を採用しません。
`--prepare-only` は Engine に接続せず、古い音声の無効化と manifest の整合だけを行います。

WAV がない・台本ハッシュが合わない・WAV が破損した場合は、文字数から尺を推定して無音レンダリングできます。
ナレーション：**VOICEVOX:WhiteCUL**（ノーマル / style ID 23）。

## 数値実験と解釈

PRML の図の直接複製ではなく、自作データの再現可能な実験です。
生成関数は sin(2πx)、ノイズ標準偏差は 0.25。乱数 seed と追加データの条件は `polynomial_model.py` に固定します。
訓練は等間隔10点、テストは新しいノイズで生成した100点です。左図に出すテスト残差は代表10点で、RMS は100点全体から計算します。

- 曲線軸は通常 ±1.5、データ追加は最低 ±1.7。曲線が外へ出る場合だけ連続的に縦軸を広げ、目盛りも更新します。係数・RMS の軸は固定です。
- 係数は塗りつぶした棒で表示。式には明示的な絶対値・括弧を使い、正則化のバネは w₆ の横の2巻きだけにします。
- M は整数、係数は M+1 個。整数の間は前後の最適係数を補間して「遷移中」と表示します。RMS の点は整数 M でだけ記録します。
- 谷の等高線は E の二次形式から計算。谷底へ向かう経路は説明用で、特定の最適化アルゴリズムの反復ではありません。
- データ追加は元の点を保持し、整数 N ごとに再計算します。追加中の曲線は隣り合う N の解の補間です。
- 正則化は Eq.(1.4) に合わせて切片 w₀ も含みます。拡大最小二乗を `numpy.linalg.lstsq` で解き、正規方程式による条件数の悪化を避けます。
- 正則化シーンの λ は exp(ln λ) > 0。左端の ln λ=−32 を λ=0 と同一視しません。
- 訓練／テスト RMS は正則化項を含まない予測残差から計算します。M や λ の実用上の選択には、テストと別の検証データが必要です。
- 最後の帯は生成関数の周囲の既知ノイズ ±2σ。学習した曲線の信頼区間を計算したものではありません。

## 原文と演出の参照

Bishop (2006), *Pattern Recognition and Machine Learning*, 印刷 pp.1–12 を参照。
主に Fig.1.2–1.8、Eq.(1.1)–(1.4)、Table 1.1–1.2。Fig.1.1 の手書き数字の導入は省き、汎化と回帰を最初の問いへ組み込みました。
ローカル原文はリポジトリ本体の `.working/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` です。

[3b1b/manim](https://github.com/3b1b/manim) と [3b1b/videos](https://github.com/3b1b/videos) の演出と実装を参照しました。
ManimGL のコードはコピーせず、CE の `ValueTracker`、updater、`always_redraw`、`TransformMatchingTex` で実装しています。

- `videos/_2023/clt/main.py`: 量を動かすと図形・数値・式が一緒に変わる表示、偏差の二乗を面積に置き換える説明。
- `videos/_2017/nn/part2.py`: 観測と予測の違いを先に見せ、二乗誤差の項へ変換し、コストの低下を図へ戻す構成。
- `manim/manimlib/mobject/value_tracker.py` / `mobject_update_utils.py`: 一つの状態から複数の表示を更新する考え方。

参照コミット、数値・映像検証の結果、制約は worktree の `reports/working/` の作業レポートに記録します。

## 今回の検証

字幕数式を、かなの実測文字高に合わせて調整しました。x は67%、λ は約105%、M+1 は112%。
下付き・上付きを除いた本体の中心を日本語へそろえ、数式の左右にかな文字高の35%の間隔を設けます。
構文確認、既存5テスト、全160字幕の安全領域、キャッシュ無効の全編レンダリングを確認しました。
全記号字幕18枚を含む最終42フレームを目視しました。映像600.066344秒、音声600.106667秒、差0.040323秒。
3秒以上の無音0件、平均音量−26.6 dB、ピーク−5.7 dB。全フレーム目視・全編の通し聴取は未実施です。
場面別時刻と寸法は [今回の字幕サイズ調整レポート](../../../reports/working/20260925-2144-prml-1-1-caption-size.md) に記録しています。
全文の読み確認は [前回の字幕・読み上げレポート](../../../reports/working/20260925-2131-prml-1-1-caption-readings.md) を参照してください。
以前の検証は [音声同期レポート](../../../reports/working/20260925-1630-prml-1-1-voicevox-sync.md)、[初回制作レポート](../../../reports/working/20260925-1501-prml-1-1-3b1b-remake.md) に残します。
