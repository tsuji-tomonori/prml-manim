# PRML 1.5 決定理論

「病気の確率が8%なら、どう決める？」から始める、9分14.2秒・9シーンの日本語解説です。問い、連続変形、数式の順で、確率と損失から行動を選ぶ考え方を説明します。Manim Community Edition 0.20.1、VOICEVOX:WhiteCUL ノーマル（23）を使用します。

## 動画

[音声付き480p15 MP4](media/videos/prml_1_5_decision_theory/480p15/PRML15DecisionTheory.mp4)

## 構成

| シーン | 尺 | 視覚的アイデア | 原文 |
|---|---:|---|---|
| 確率が８％なら、どう決める？ | 50.200秒 | 同じ確率バーから二つの行動を問い、入力を動かして確率へ戻る | pp.38–39 / Eq.(1.77) |
| 間違いを、面積で数える | 64.200秒 | 二つの同時密度上で境界と誤分類面積を連動 | pp.39–40 / Eqs.(1.78)–(1.79) / Fig.1.24 |
| 見逃しの重さを変えたら？ | 74.800秒 | 損失のつまみがリスクの棒と最適境界を変える | pp.41–42 / Eqs.(1.80)–(1.81) / Fig.1.25 |
| 自信がない場所は、保留する | 52.133秒 | しきい値と棄却帯・棄却率を同じ事後確率の上で連動 | p.42 / §1.5.3 / Fig.1.26 |
| 確率は、どこまで求める？ | 68.000秒 | 同じ図を密度→事後確率→ラベルへ変形 | pp.42–44 / Eqs.(1.82)–(1.83) / Fig.1.27 |
| 集団が変わる、情報が増える | 77.333秒 | 事前のつまみで確率バーを縮め、二つの証拠を結合 | pp.45–46 / Eqs.(1.84)–(1.85) |
| 一本の予測線は、どこに置く？ | 73.067秒 | 条件付き分布と動く予測線、二乗損失の面積、消せない分散 | pp.46–47 / Eqs.(1.86)–(1.90) / Fig.1.28 |
| 損失を変えると、答えも変わる | 56.333秒 | 同じ非対称分布で平均→中央値、損失の形を連続変形 | pp.48–49 / Eq.(1.91) / Fig.1.29 |
| 確率と損失から、行動を選ぶ | 38.133秒 | 冒頭の８％へ戻り、損失だけを変えて結論を確かめる | pp.38–48 / §1.5 全体 |

## 数値と原文の扱い

- 分類は平均 −1 と 1、標準偏差1、事前確率各0.5の自作ガウス分布。密度の交点は0、最小誤分類率は0.158655。画面は x∈[−5,5]、確率計算は実数全体の裾を含む。
- 病気を C₁、健康を C₂ とする。損失行列は行＝真のクラス、列＝判断。正解0、健康を病気と判断すると1、見逃しの損失 c を1→20→1000へ変える。確率0.08・c=1000で期待損失は0.92対80。損失棒は共通の線形縦縮尺を自動調整する。
- 棄却条件は maxₖ p(Cₖ|x)≤θ。θ=0.6で棄却率0.098108、θ=0.9で0.521351。θ<1/2は棄却なし、θ=1は全件棄却。
- 事前確率補正はクラス条件付き分布が変わらないことを仮定する。学習事前0.5・事後0.8、新事前0.01で事後0.038835。
- 情報結合はクラスを条件とした独立を仮定する。事前0.2、画像事後0.5、血液事後0.6から、結合事後0.857143。事後の積を事前で一度割り、正規化する。
- 回帰の第1例は標準偏差0.6。二乗損失の下限は分散0.36。黄色の細い棒は二乗損失の密度を共通縮尺で描いた面積近似であり、軸の下側への表示は負の損失を意味しない。
- 第2例は0.65 N(−1,0.45²)+0.35 N(1.6,0.6²)。平均−0.09、中央値−0.668720、最頻値約−1.0。同じ分布上で予測を比較する。
- 原文は印刷pp.38–48（PDF pp.58–68）。Fig.1.29は印刷p.49。Fig.1.25は表形式の「図」で、独立した表番号ではない。
- Eq.(1.90)は[著者の正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)に従い、第2項を ∫Var[t|x]p(x)dx とする。Fig.1.27の等事前確率という条件も確認した。
- Eq.(1.91)は q=2（平均）とq=1（中央値）を扱う。原文のq→0と最頻値の記述は、連続分布一般の結論として用いない。最頻値は、別の損失である「狭い許容幅に入る確率」を最大化する考え方で補足する。
- 医療の確率・損失は説明用の仮定であり、診断性能の実測値ではない。PRMLの図は複製していない。

## 字幕・音声・同期

`narration_content.py` が正本です。112文を display（数式記号）と speech（読み上げ文）で保持します。MathTexの本体中心と日本語の実測文字高を合わせ、添字は自然な位置を保ちます。1.1で確認済みの字幕ヘルパーを節内へ移植しています。

音声は文単位で合成して9 WAVへ結合します。PCMの実測時間、台本とWAVのSHA-256、文ごとの字幕時刻をmanifestへ保存します。アニメーションは同じ文の時計を使い、15 fpsのフレーム境界へ合わせます。古い音声や欠落した音声ではレンダリングを停止します。

`reading_check.md` / `reading_check.json` に全112文の修正前後の読みを記録しています。「行→くだり」「値→ね」「絶対値→ぜったいね」「二分の一→にふんのいち」「零点九二→れいてんきゅうじゅうに」をspeech側だけで修正し、再取得した読みを確認しました。音声全編の通し聴取は実施していません。

## 再生成

このディレクトリで実行します。Engineが http://127.0.0.1:50021 で稼働している必要があります。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_5_decision_theory.py PRML15DecisionTheory
```

音声生成の再開例: `make_voicevox_narration.py --from-scene scene05`。最終動画以外のmedia生成物と`.working/`はGit管理しません。

## ファイル

| ファイル | 用途 |
|---|---|
| `prml_1_5_decision_theory.py` | 公開クラス `PRML15DecisionTheory` と9シーンの演出 |
| `decision_model.py` / `verify_numerics.py` | 数値モデルと独立した数値恒等式・音声整合の検証 |
| `visual_support.py` / `narrated_scene.py` | 日本語＋数式字幕、文単位の音声同期 |
| `narration_content.py` / `narration_script.md` | display/speechと原文対応付き台本 |
| `make_voicevox_narration.py` / `assets/voicevox/` | 音声生成、9 WAV、manifest |
| `check_narration_readings.py` / `reading_check.*` | 全文の読み取得、修正前後の記録 |
| `validation_results.json` | 最終MP4のストリーム・音響・同期時刻・画像抽出時刻 |

## 参考にした演出

[3b1b/videos の Bayes](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2019/bayes/part1.py)の確率バー、面積から式への対応、同じ図の状態変化と、[CLT](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/clt/main.py)の確率で重み付けした偏差の面積を参考にしました。[3b1b/manim](https://github.com/3b1b/manim)のValueTracker・updaterの考え方を、Manim CEで実装しています。ManimGLコードは移植していません。

## 検証

最終検証結果は `validation_results.json` と作業レポートに記録しています。映像554.20秒／音声554.24秒、3秒以上の無音0件、平均音量−26.7 dB／最大−6.4 dB。全112文の読み、数値検証、全56段階と数式字幕12場面を含む76枚の画像、4シーンの同期を確認しました。全フレームの人手確認、高解像度版のレンダリング、音声全編の通し聴取は未実施です。

## 音声クレジット

ナレーション: VOICEVOX:WhiteCUL（ノーマル、speaker=23）。Engine 0.25.2、話速1.08、抑揚0.95。
