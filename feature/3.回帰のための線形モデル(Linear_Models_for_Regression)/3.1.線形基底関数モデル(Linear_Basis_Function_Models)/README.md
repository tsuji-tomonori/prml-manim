# 3.1 線形基底関数モデル

曲がった線を「線形モデル」で作れるのでしょうか。部品となる基底関数を足す実験から始め、ガウスノイズ、最小二乗、直交射影、逐次学習、正則化、複数出力へ進む約10分15秒の日本語動画です。

Manim Community Edition 0.20.1、VOICEVOX:WhiteCUL ノーマル（speaker 23）を使用します。公開ファイル名とクラス名は従来どおりです。

## 構成

| シーン | 音声尺 | 視覚的な実験 | PRML 印刷ページ・式・図 |
|---|---:|---|---|
| 曲がった線を、線形モデルで作れる？ | 67.40秒 | 正負の重み・切片のつまみと合成曲線 | pp.138–139、(3.1)–(3.3) |
| 部品の形を変えると、どこが動く？ | 64.33秒 | 同一座標で多項式→ガウス→シグモイド、中心と幅を操作 | pp.139–140、(3.4)–(3.6)、Fig.3.1 |
| なぜ、ずれを二乗するの？ | 70.00秒 | ガウスの幅、密度の高さ、残差と面積 | pp.140–143、(3.7)–(3.12)、(3.21) |
| 全部の点を、一度に予測するには？ | 68.53秒 | 基底の値→行→行列積→正規方程式 | pp.141–142、(3.14)–(3.20) |
| 最小二乗は、どこへ下ろす垂線？ | 68.53秒 | 二観測の実例で距離が最小になる直交射影 | p.143、Fig.3.2 |
| 点が一つ届くたび、どう直す？ | 64.93秒 | 一点ずつ更新、学習率で飛び越しを観察 | pp.143–144、(3.22)–(3.23) |
| よく合う曲線の、重みはどうなっている？ | 69.13秒 | 正則化を往復し、曲線と係数棒を連動 | pp.144–145、(3.24)–(3.28) |
| なぜ、角があると重みがゼロになる？ | 73.53秒 | 同じ重み空間で円→ひし形、等高線の接点 | pp.145–146、(3.29)–(3.30)、Figs.3.3–3.4 |
| 一つの入力から、二つの出力へ | 69.00秒 | 基底を共有して二曲線を学習、片方の観測だけ変更 | pp.146–147、(3.31)–(3.35) |

原文は Bishop (2006), *Pattern Recognition and Machine Learning*, §3.1（§3.1.1–3.1.5 を含む）。印刷 pp.138–147 は手元の PDF の158–167ページに対応します。該当節に表はありません。図を転写せず、自作データ・数値例で構造を説明しています。

## 数値例と成立条件

- 入力12点、乱数 seed 31、生成関数 `sin(2πx)+0.25cos(5πx)`、ノイズ標準偏差0.18。ガウス中心8個と定数基底1個を使います。第二出力は別の自作データ（seed 32）です。
- 本節の `M` は重みの個数です。1.1 の多項式の最高次数とは異なります。
- 中心・幅の操作は基底の形を説明する実験です。学習では基底を固定します。ガウス基底を確率密度として正規化する必要はありません。
- ガウスノイズは独立・同一分散を仮定します。最尤ノイズ分散は残差二乗和を `N` で割り、不偏分散推定とは区別します。
- 逆行列による最小二乗解の表示には列の独立性が必要です。実装は `numpy.linalg.lstsq`、正則化は拡大最小二乗を使います。特異な場合も擬似逆行列で予測を扱えます。
- 射影の導入は二観測 `(1,3)` と定数基底 `(1,1)` の説明例です。最短の予測は `(2,2)`、残差は `(-1,1)` です。Fig.3.2 の高次元空間への入口にしています。
- 正則化は Eq.(3.27) に合わせて切片も含めます。`log10 λ` を −6 から2まで動かします。これは λ が正の範囲の比較で、λ=0を含みません。各係数の単調な縮小や、汎化誤差の単調改善は主張しません。
- Fig.3.4 に対応する独自例では誤差中心を `(1.8,0.5)` とし、半径1の円での解は約 `(0.964,0.268)`、ひし形での解は `(1,0)`。円からひし形へ変わる途中の指数でも、接点と等高線を毎回計算します。原図とはゼロになる軸が異なります。制約定数は学習率との混同を避けて `c` と表記します。
- Eq.(3.29) の添字表記は映像で `Σ_j` として扱い、実装上の対象係数を明示します。`q<1` の図形は形状の比較であり、凸最適化や一般の制約・ペナルティの同値性を主張するものではありません。制約との対応を説明する最後の場面では `q=1` に戻します。
- 複数出力は同じデザイン行列と出力別の重みを用います。片方を一律0.3上げる実験では、定数基底により対応する予測も0.3上がります。この性質を独立に再計算して検証します。

## 字幕・読み・同期

`narration_content.py` が63 beat・133文の正本です。字幕は `display`、読み上げは `speech` に保存します。字幕中の `$...$` は MathTex で描き、1.1 の実測文字高・添字を除いた本体中心・左右余白の仕組みを継承しています。

`check_narration_readings.py` が全 speech の `audio_query` を speaker 23 で取得し、`reading_check.md/json` に初回と修正後の読みを並べます。値・負・節・行・一行・黄色・絶対値・角の誤読を修正しています。全133文の API 読みを確認しました。アクセントの自然さを全編通しで聴取した検証ではありません。

文ごとのPCM長、台本・WAVのSHA-256、字幕の開始終了時刻を `manifest.json` に保存します。映像は同じ時計を使い、beat の境界を15 fpsへ揃えます。音声欠落・台本不一致の場合は描画を停止します。`media/prml31_timeline.json` は描画時に生成する一時的な同期記録です。

## ファイルと再生成

- `prml_3_1_linear_basis_function_models.py`：`PRML31LinearBasisFunctionModels` と9シーン。
- `basis_model.py` / `verify_numerics.py`：NumPy計算、数値恒等式・表示範囲・音声整合の検証。
- `visual_support.py` / `narrated_scene.py`：日本語＋MathTex字幕、PCM時計とアニメーションの同期。
- `narration_content.py` / `narration_script.md`：字幕・収録文・原文参照。
- `make_voicevox_narration.py`：音声生成。`--from-scene scene05` 等で再開。
- `check_narration_readings.py` / `reading_check.md/json`：全文の読み確認。
- `review_video.py` / `validate_video.py`：全beat・全数式字幕・同期比較の画像抽出と音響・時計の検証。
- `assets/voicevox/`：今回の9 WAVとmanifest。旧構成の音声は置換。
- `numerical_results.json`：数値検証の実測値。

対象ディレクトリで実行します。VOICEVOX Engine は事前に `http://127.0.0.1:50021` で起動してください。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
# reading_check.md の全文を確認し、必要なら speech を修正して再実行
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_3_1_linear_basis_function_models.py PRML31LinearBasisFunctionModels
/home/t-tsuji/project/prml-manim/.venv/bin/python review_video.py
/home/t-tsuji/project/prml-manim/.venv/bin/python validate_video.py
```

動画：`media/videos/prml_3_1_linear_basis_function_models/480p15/PRML31LinearBasisFunctionModels.mp4`。
音声クレジット：**VOICEVOX:WhiteCUL**。

## 演出の参照

[3b1b/manim](https://github.com/3b1b/manim) の ValueTracker / updater の考え方、[3b1b/videos](https://github.com/3b1b/videos) の `_2016/eola/chapter7.py`（射影）、`_2017/nn/part2.py`（差から誤差関数へ）、`_2023/clt/main.py`（分布と数値表示の連動）を参照しています。ManimGL のコードを取り込まず、Manim CE で再実装しています。

## 検証結果

最終MP4は854×480・15fps、映像615.399678秒・音声615.424000秒（差0.024322秒）です。`silencedetect=noise=-45dB:d=3` で長時間無音0件、平均音量−26.5dB・ピーク−5.8dBでした。全133文の字幕時刻をPCMと照合し、全63beat・全13数式字幕・3シーンの同期前後を含む82枚を画像で確認しました。

実測値と動画ハッシュは `validation_results.json`、数値検証7群の結果は `numerical_results.json` に保存しています。レビュー画像は再生成可能な一時ファイルです。

全編の通し聴取、全フレームの人手検査、高解像度版のレンダリングは未実施です。

詳しい原文対応・参照ファイル・修正記録は[作業レポート](../../../reports/working/20260927-1804-prml-3-1-3b1b-remake.md)を参照してください。
