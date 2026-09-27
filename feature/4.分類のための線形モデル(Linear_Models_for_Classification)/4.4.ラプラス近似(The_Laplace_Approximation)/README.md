# PRML 4.4 ラプラス近似

山の頂上の位置と曲がり具合から、分布と積分を近似する9分40秒の日本語動画です。Manim Community と VOICEVOX:WhiteCUL を使用します。

[動画（480p15）](media/videos/prml_4_4_laplace_approximation/480p15/PRML44LaplaceApproximation.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## シーンと原文

Bishop (2006), *Pattern Recognition and Machine Learning* §4.4–4.4.1、印刷pp.213–217（PDF pp.233–237）を `pdftotext -layout` で抽出・照合。図4.14はPDF画像でも確認しました。式(4.125)–(4.139)に対応し、本節に表はありません。

| シーン | 開始秒 | 尺（秒） | 視覚的な操作 | 原文 |
|---|---:|---:|---|---|
| scene01 山の頂上だけで、分布を描ける？ | 0.000 | 57.467 | 非対称の密度・積分面積・頂上へのガウス重ね合わせ | pp.213–214 / (4.125), Fig.4.14 (p.215) |
| scene02 頂上を探して、対数を取る | 57.467 | 61.867 | 点と接線を動かし、同じ軸上で関数から対数へ変形 | p.214 / (4.125)–(4.128) |
| scene03 曲率のつまみが、幅を決める | 119.333 | 63.667 | 曲率のつまみで放物線とガウスの幅を変える | p.214 / (4.127)–(4.130); Fig.4.14 |
| scene04 正規化すると、頂上の高さは？ | 183.000 | 63.800 | 個別に正規化し、頂上の高さ・平均とモードの違いを見る | pp.214–215 / (4.129)–(4.130), Fig.4.14 |
| scene05 二方向の幅を、一つの行列へ | 246.800 | 67.333 | 二方向の曲率を変えて楕円を縮め、回転する | pp.215–216 / (4.131)–(4.134) |
| scene06 積分は、山の高さ × 広がり | 314.133 | 66.667 | 短冊面積と高さ・幅・積分値を連動 | p.216 / (4.133), (4.135) |
| scene07 モデル比較では、幅も効く | 380.800 | 65.800 | 同じ軸上で尤度の幅と事前の範囲を変える | pp.216–217 / (4.136)–(4.138) |
| scene08 BIC は、どこを簡略化する？ | 446.600 | 62.933 | データ数のつまみと2個・5個のパラメータの補正量 | p.217 / (4.138)–(4.139) |
| scene09 一つの山を見る道具、その限界 | 509.533 | 70.467 | 第二の山・非対称性・対数変換・集中する形を比較 | p.216; §4.5への接続 p.217 |

## 数値例と説明上の区別

- 一次元の例は `f(z)=exp(-z²/(2×1.1²)) / (1+exp(-(4z+0.8)))`。原図のパラメータやデータを複製していません。モードはNewton法、積分は−12〜12の48,001点の台形則で計算し、別の密な格子でも照合します。
- モード0.401033、曲率2.042328、分散0.489637。元の平均は0.707493。積分1.563198、近似積分1.505225（相対差−3.7086%）。正規化後のピークは0.548985と0.570129です。
- 対数での局所的な高さ・傾き・曲率の一致と、正規化後の密度の高さを区別します。大域的な誤差最小化や平均への適合として説明しません。
- 多変数の等高線は回転行列と精度の固有値から計算。正定値が必要で、極大点でも二階微分がゼロになる場合を除きます。固有値0の例ではガウス楕円を描かず、平らな方向の等高線を表示します。
- 高さと幅の例の面積は3.759942→1.253314→2.982888。表示の短冊は近似ですが、数値表示は全実数での解析積分です。
- 証拠の例はガウス形の尤度と有限範囲の一様事前の積を数値積分。幅0.8→0.3、事前幅5→8で、証拠は0.400347→0.150398→0.093999です。
- BICは独立な大標本・固定次元・識別可能な正則モデルと広い事前を想定した粗い簡略化です。原文の大きいほど良いスコアと、通常の小さいほど良いBICを区別。通常の式の最尤点と原文のMAP点は、この大標本条件で漸近的に対応します。データ数の途中の小数はつまみの表示補間で、端点は整数です。
- 二峰の例のモードと曲率も数値計算します。モード間を移す途中の赤線は表示上の補間です。最後の集中実験は `p_n∝f^n` の自作例で、横軸を `sqrt(nA)(z-z₀)` に換算して形を比較します。個別の実データによる事後更新ではありません。
- 正の変数の対数変換ではヤコビアン `1/τ` も表示。ラプラス近似が多峰性・強い非対称性・大域的な裾を捉えにくい点を説明します。

## 3Blue1Brown の参照

ManimGL のコードは移植せず、演出の考え方を Manim CE で実装しています。

- [ExampleApproximation / ConstructQuadraticApproximation](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/eoc/chapter10.py): 同じ座標上で関数と近似を比較し、式の各項と高さ・傾き・曲率を対応させる。
- [BellCurveArea / AntiDerivative](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/gauss_int/integral.py): 積分を面積や短冊として見せ、動く端点と面積を連動させる。
- [UpdatersExample / CoordinateSystemExample](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/example_scenes.py): 一つの状態から点・曲線・ラベル・数値を更新する。

## ファイルと再生成

- `prml_4_4_laplace_approximation.py`: 公開クラス名 `PRML44LaplaceApproximation` を維持した本体。
- `laplace_model.py` / `verify_numerics.py`: 数値例と独立検証。
- `narration_content.py`: 字幕display／音声speechの正本。
- `caption_layout.py` / `scene_support.py`: 1.1の文字実測高と数式本体の中心線、PCMに同期するアニメーション。
- `make_voicevox_narration.py`: speaker=23、話速1.04。文キャッシュ、台本・WAVハッシュ、途中再開に対応。
- `check_narration_readings.py` / `reading_check.md` / `reading_check.json`: 全文の修正前後の読みとkana。
- `export_narration_script.py`: 原文参照・視覚操作・実測尺付き台本を書き出す。
- `verify_video.py`: ffprobe・音響ログ・全区切りと全記号字幕・同期比較画像を収集。画像の目視や聴取を自動検査で代用しません。

このディレクトリから実行します。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -W error -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_4_4_laplace_approximation.py PRML44LaplaceApproximation
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

`make_voicevox_narration.py --from-scene scene04` で再開できます。VOICEVOX Engine は http://127.0.0.1:50021、0.25.2。コンテナの起動停止は行いません。9 WAVを新台本で置き換え、manifestと整合を確認します。古い音声・欠落した音声では描画を開始しません。

字幕は日本語「あ」とTeXの「x」の実測高で合わせ、添字を除いた数式本体の中心線をそろえます。左右には文字高の35%の余白。72件の字幕は最大幅9.6453、高さ0.7569（上限12.9、0.9）です。

## 音声クレジット

VOICEVOX:WhiteCUL（ノーマル、speaker 23）。全文のAPI読みを確認し、「一変数」「正」「負」「値」「節」の誤読をspeech側で修正。修正後も全件を再取得しました。全編の通し聴取は未実施です。

## 最終検証

構文確認、数値・音声整合の7検証群、全編レンダリング（72 animations）が成功。映像579.998044秒、音声580.032000秒。3秒以上の無音0件、mean_volume: -26.9 dB, max_volume: -6.5 dB。最終動画の78枚（全72区切り、全11記号字幕を含む）を目視し、3シーンのPCM発声と動作を照合しました。

[作業レポート](../../../reports/working/20260928-0211-prml-4-4-3b1b-remake.md)に数値・修正履歴・検証範囲を記載しています。
