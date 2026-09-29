# PRML 1.6 情報理論

「短く伝えるには？」という問いから、情報量・エントロピー・分布の学習へ進む、約10分9秒の日本語動画です。Manim Community、VOICEVOX:WhiteCUL（ノーマル、speaker 23）を使用します。

## 動画と構成

[音声付き480p15動画](media/videos/prml_1_6_information_theory/480p15/PRML16InformationTheory.mp4)

| シーン | 尺（秒） | 視覚的な実験 | PRML 印刷ページ・式・図 |
|---|---:|---|---|
| 1 珍しい知らせ | 59.600 | 8枚のカードを絞り、確率と情報量の点を動かす | pp.48–49、(1.92) |
| 2 短い符号 | 62.800 | 4状態の棒を変形し、確率×符号長を面積にする | pp.49–50、(1.93) |
| 3 並べ方と不確かさ | 59.667 | 6位置の20通り、30状態の山→一様→集中 | pp.50–52、(1.94)–(1.100)、図1.30 |
| 4 細かい目盛り | 61.533 | 1.2の密度と面積を復習、4→8→16分割、密度と区間確率、負の微分エントロピー | pp.52–53、(1.101)–(1.104) |
| 5 最大エントロピー | 57.400 | 同じ平均・分散で形を変え、σで幅を変える | pp.53–54、(1.105)–(1.110) |
| 6 条件付き情報 | 60.867 | 同時確率の面積から列を取り出し正規化する | pp.54–55、(1.111)–(1.112) |
| 7 想定のずれ | 58.133 | pとqを重ね、平均符号コストとKLを連動させる | pp.55–57、(1.113) |
| 8 なぜ非負か | 62.267 | −lnの曲線と弦、比の重み付き平均が1になる数値例 | pp.55–57、図1.31、(1.114)–(1.118) |
| 9 圧縮から学習へ | 59.800 | 自作20観測、θのつまみで対数尤度の谷へ | p.57、(1.119) |
| 10 共有する情報 | 67.267 | 周辺確率を保って依存を強め、残りと共有を積み上げる | pp.57–58、(1.120)–(1.121) |

## ファイル

- `prml_1_6_information_theory.py`：公開クラス `PRML16InformationTheory`、10シーンの実装。
- `information_model.py`：NumPyによる確率・エントロピー・KL・尤度の計算。
- `narration_content.py`：61 beat、148文の字幕 `display` と読み上げ `speech` の正本。
- `narration_script.md`：原文参照・各beatの視覚操作・全文の台本。
- `caption_layout.py`：1.1から引き継いだ日本語とMathTexの文字高・中心線・余白の調整。
- `make_voicevox_narration.py`：文ごとのPCM尺で合成、ハッシュと字幕の一致検査、シーン単位の再開。
- `check_narration_readings.py` / `reading_check.md` / `reading_check.json`：全148文のAPI読み、修正前後の記録。
- `assets/voicevox/`：10 WAVとmanifest。今回の変更分はscene04・scene08。ほかの8シーンは既存音声を保持。
- `verify_numerics.py`：符号の一意性、積分、KL、連鎖則、相互情報量、最尤解、音声整合性の検証。
- `review_video.py`：全beat・全記号字幕・3シーンの同期前後画像を抽出する補助ツール。

## 再生成

このディレクトリで実行します。既存venvを使い、VOICEVOX Engineをユーザー側で起動してください。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python narration_content.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_6_information_theory.py PRML16InformationTheory
/home/t-tsuji/project/prml-manim/.venv/bin/python review_video.py
```

`make_voicevox_narration.py --only-changed` は、ハッシュ・字幕・WAVの整合性が確認できるシーンを再利用し、変更シーンだけを生成します。

生成途中からは `make_voicevox_narration.py --from-scene scene06` のように再開できます。映像は音声と台本が一致しない場合に停止します。文ごとのPCM時刻と動作の時刻は `media/prml16_timeline.json` に出力されます。字幕の数式は読み仮名へ変換しません。

## 視覚補足と復習

- scene04冒頭（15.267秒）：1.2の緑の密度曲線・青い区間面積を再現。区間を広げる操作を振り返り、今回の一様分布へ接続します。既存導入7.667秒の置換なので純増は7.600秒です。
- scene08の比の平均の説明直後（7.067秒）：青い重み `p=(1/2,1/2)` と黄色の比 `q/p` から分母を消し、残る `q=(1/4,3/4)` を棒で集めて緑の合計1へ。両分布が同じ2状態で正という条件を表示します。
- 見直し計画の小窓を、分数を読める本文カードへ拡大しました。復習は導入を置き換え、対数・符号長・階乗の既存説明は重ねません。

## 原文との対応と条件

原文PDFの印刷pp.48–58（PDFの68–78ページ）をテキスト抽出して照合し、図1.30・1.31も画像で確認しました。本節に表はありません。原文の8状態の符号例を、独自の4状態 `p=(1/2,1/4,1/8,1/8)` に置き換えています。図を複製せず、図1.30と1.31の数学的構造を自作しています。

- 情報量と符号化はbit、自然対数の説明以降はnat。量子化の2/3/4 bitと最後の共有1 bitには単位を明示します。
- 符号長の一般式 `−ln q` は理想的な平均符号化コスト。1記号の整数長符号が常にエントロピーと等しくなるという主張ではありません。
- 微分エントロピーは画面で `h[p]` と記し、原文の `H[x]` に対応させます。負にもなり、測定単位に依存します。
- ガウス分布の最大性は正規化・平均・分散の固定のもとで述べます。変分法の計算過程は省略しています。
- 条件付きエントロピーと相互情報量は離散版で例示。観測による不確かさの減少は条件について平均した性質です。
- JensenによるKLの説明では、pとqが同じ範囲で正という条件を明示。p>0かつq=0でKLが無限大となる場合も説明します。
- 式(1.119)の標本和には式(1.35)に従って `1/N` を補い、標本平均として表示しています。目的関数の正の定数倍は最適解を変えません。未知の分布とのKLを直接計算したものではありません。

## 3Blue1Brownの参照

[3b1b/videos](https://github.com/3b1b/videos) と [3b1b/manim](https://github.com/3b1b/manim) を参照し、演出の考え方をManim CEで実装しました。ManimGLのコード・素材は取り込んでいません。

- `_2026/cross_entropy/entropy.py` の `InformationGraph`：確率を半分にする操作から対数グラフへ移す。
- `_2026/cross_entropy/distribution.py`：確率を面積・幅に対応させる。
- `_2019/bayes/part1.py` の `ProbabilityBar`：同じtrackerで割合と形を更新する。
- `_2023/clt/main.py` の `BuildUpGaussian`：σと分布・目印を連動させる。
- `manimlib/mobject/value_tracker.py`：連続変形の状態を一箇所に置く。

参照commit：videos `ae2b911326b8255dfae790b6e8764a8a400c5875`、manim `fafa083a4fb274bba9cabde0b6e2f50ba6da0622`。

## 読み確認と制約

API読みの全文確認で13文を修正しました。例：弦「ツル→ゲン」、負「マケ→フ」、値「ネ→アタイ」、黄色「オオショク→キイロ」、1.75「イッテンナナジュウゴ→イッテンナナゴ」、対数の底「ソコ→テイ」、分数の「フン／プン→ブン」。字幕は漢字・数式のままです。

初回制作時は全文のAPI読みを目視しました。今回も全148文のAPI読みを取得し、新規・変更5文を目視しました。追加文の新たな誤読は0件です。「一章二せつ」は speech だけに用い、字幕には「1.2」と表記します。全編の通し聴取、音素の強制アラインメント、全フレームの目視は実施していません。最終動画の計測と画像確認の記録は今回の作業レポートを参照してください。

ナレーション：**VOICEVOX:WhiteCUL**。

今回の検証記録：[視覚補足・復習レポート](../../../reports/working/20260930-0255-prml-1-6-visual-aid-recap.md)。映像609.333秒、増加2.466%、音声との差0.032322秒、3秒以上の無音0件、平均音量−26.6dB・最大−5.4dB。

初回制作記録：[作業レポート](../../../reports/working/20260927-1306-prml-1-6-3b1b-remake.md)、[計測JSON](validation_results.json)。
