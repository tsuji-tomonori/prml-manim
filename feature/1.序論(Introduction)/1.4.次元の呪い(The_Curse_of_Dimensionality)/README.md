# PRML 1.4 次元の呪い

問いを出し、数値実験を動かし、式で確かめる約10分6秒の解説動画です。
Manim Community で制作し、ナレーションは VOICEVOX:WhiteCUL（ノーマル、speaker 23）です。

[動画（480p15）](media/videos/prml_1_4_curse_of_dimensionality/480p15/PRML14CurseOfDimensionality.mp4) ／ [全台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## 構成

時刻・尺は文単位で合成した WAV の実測値（秒）です。

| シーン | 問い | 開始 | 尺 | 視覚的アイデア | 原文参照 |
|---|---|---:|---:|---|---|
| scene01 | この点は、何色？ | 0.000 | 62.533 | 自作の点群を箱で囲み、多数派を数える | pp.33–35 / Figs.1.19–1.20 |
| scene02 | 箱は、どれだけ増える？ | 62.533 | 64.933 | 5区間→25マス→125区画、次元と対数グラフを連動 | p.35 / Fig.1.21 |
| scene03 | 多項式なら、つまみはいくつ？ | 127.467 | 75.000 | 交差項の係数で曲面を変形し、係数数を数える | p.36 / Eq.(1.74), Ex.1.16 |
| scene04 | 球の体積は、どこにある？ | 202.467 | 76.733 | 内側を縮め、体積比の定数を約分する | p.36 / Eqs.(1.75)–(1.76) |
| scene05 | 薄い表面が、ほとんど全部？ | 279.200 | 69.600 | 厚さと次元を動かし、同じ座標上で殻の割合を比較 | pp.36–37 / Fig.1.22 |
| scene06 | 中心が一番高いのに、中心にはいない？ | 348.800 | 99.067 | 600標本を半径のヒストグラムへ移し、環と面積を連動 | pp.36–37 / Fig.1.23, Ex.1.20 |
| scene07 | 次元を上げると、確率の山は？ | 447.867 | 74.533 | 同じ半径軸で山を移し、規格化した半径で相対集中を示す | pp.36–37 / Fig.1.23, Ex.1.20 |
| scene08 | 画像の数字は多い。でも自由度は？ | 522.400 | 83.533 | 16×16画素を位置・角度の3つのつまみで連動させる | pp.37–38 / 実効次元・滑らかさ |

## 視覚補足と復習（第1〜5章の見直し）

- **348.800–360.400秒、11.600秒：1.2 ガウス分布の復習。** 既存の導入beatの中で節番号を字幕・音声に示し、1.2 `likelihood()` の赤いベル、黄色の平均線、標準偏差による幅の変化を再現します。そのまま各座標が独立・平均0・標準偏差1という本編の条件へつなぎます。独立した復習beatは追加していません。
- **413.333–427.733秒、14.400秒：半径密度の補足 V02。** `scene06-06-02` の直後で、本文図を三段グラフへ置換。D=2、σ=1とし、青 `#58C4DD` の `exp(-r²/2)`、黄 `#FFFF00` の `r`、緑 `#83C167` の積を別軸で示します。同じ半径の縦ガイドを0.3→1→2へ動かし、最大になる半径1で一度保持します。
- 上二段は定数を省略した因子で、D=2・σ=1ではその積が正規化済みの半径密度になります。縦軸の尺度は段ごとに独立です。補足後は元のヒストグラムへ戻します。

[見直し計画](../../../reports/review/ch1-5-visual-aid-recap-plan.md)の補足1件を実装しました。480pで三段の数値と曲線を読めるよう、右半分だけでなく本文領域全体を使っています。ガウスの説明には短い1.2参照と元図の操作を組み込み、同じ説明の二重化を避けました。箱の指数増加、球の体積比、組合せ公式の証明、中心極限定理の説明は追加していません。

## 原文と数値実験

Bishop, *Pattern Recognition and Machine Learning* (2006), §1.4、印刷 pp.33–38（PDF pp.53–58）を参照。式 (1.74)–(1.76)、図 1.19–1.23 の構造を扱います。この節に表はありません。

- 分類用の90点は seed=1401 の独自データです。原文の油流測定データを再現したものではありません。例の箱には A=22、B=12、C=0 点が入ります。
- 5分割格子は `5**D`、空の箱の期待割合は、一様・独立に1000点を置く仮定で `(1-1/5**D)**1000`。D=10では 99.9897605%。任意のデータ分布についての値ではありません。
- 三次以下の多項式の独立係数数は `comb(D+3,3)`。掛ける順番だけが異なる項をまとめます。D=2,10,100で 10,286,176851。次数固定のべき乗則と、格子の指数増加を区別します。
- 球の外殻の体積比は `1-(1-epsilon)**D`。ε=0.1、D=2,3,20,50で19%、27.1%、87.8423%、99.4846%。D=50で半分の体積を含む厚さは約1.37673%。円は半径の模式図で、高次元の体積比は計算値と曲線で示します。
- ガウス分布は各座標が独立、平均0、標準偏差1。半径密度は `p(r)=r**(D-1)*exp(-r*r/2)/(2**(D/2-1)*Gamma(D/2))`。原文図1.23の描画値を転写せず、この条件で再計算しています。
- ガウス標本は seed=1406 の600点。ヒストグラムは個数を標本数とビン幅で割った密度で、理論曲線と単位をそろえています。
- `u=r/sqrt(D)` への変換後は `q(u)=sqrt(D)*p(sqrt(D)*u)`。面積を保存し、相対的な幅の縮小を示します。半径の絶対的な幅が0に縮むとは説明しません。
- 次元は整数です。球とガウスのつまみの途中は、整数の状態をつなぐために公式を正の実数へ拡張した図形変形です。係数数・箱数は整数次元ごとに更新します。
- 画像は一定の照明・大きさの非対称物体を仮定した、自作の滑らかな画素モデルです。256画素、位置2＋角度1の自由度、向きの予測に関わる自由度1を分けます。輪郭を滑らかにして画素境界の不連続を避けています。

## 実装

- `narration_content.py`: 全119文の display / speech と59 beat、原文参照の正本。
- `caption_layout.py`: 1.1と同じ日本語・MathTex混在字幕。かなの実測文字高、添字を除いた本体中心、数式左右の余白で配置します。
- `check_narration_readings.py`: 全 speech を audio_query に送り、初回と修正後の読みを `reading_check.md` / JSON に記録。
- `make_voicevox_narration.py`: 文単位PCMを連結し、短い息継ぎを加え、文の字幕時刻・台本とWAVのハッシュをmanifestに保存。`--from-scene scene05` で再開できます。台本・字幕・WAVのハッシュが一致するシーンは再合成せず保持します。
- `dimension_model.py`: NumPy による数値計算と自作データ。
- `prml_1_4_curse_of_dimensionality.py`: 元のクラス名 `PRML14CurseOfDimensionality` を維持。ValueTracker / updater / 数式の色分けを使用。
- `verify_numerics.py`: 格子・単項式列挙、球の体積比とMonte Carlo、半径密度の積分・モーメント、画素、音声整合性の検証。

各シーンの音声開始と映像時計を一致させ、文の実測時間から字幕・動作を切り替えます。境界は累積時間を15fpsに丸め、音声の長さを静止待ちで埋める方式を避けています。古い台本の音声はハッシュ検査で排除し、音声が未生成・不一致なら描画を停止します。

## 再生成

このディレクトリで実行します。既存の仮想環境を使用します。

```bash
curl --fail http://127.0.0.1:50021/version
/home/t-tsuji/project/prml-manim/.venv/bin/python narration_content.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
# reading_check.md の全文を確認し、誤読は speech を修正して再確認
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_4_curse_of_dimensionality.py PRML14CurseOfDimensionality
```

Engine が応答しない場合は音声生成を停止してください。生成スクリプトはコンテナを操作しません。

## 演出の参照

[3b1b/manim](https://github.com/3b1b/manim) と [3b1b/videos](https://github.com/3b1b/videos) を参照しました。ManimGL のコードは取り込んでいません。

- `_2023/clt/main.py` の `BuildUpGaussian` / `get_variable_display`: つまみ・数式・曲線・塗り面積を同じ状態に連動させる考え方。
- `_2023/convolutions2/gauss_example_supplements.py` の `RotationalSymmetryAnnotations`: 座標の二乗和から半径への対応を色で保持し、密度と面積を結ぶ説明。
- `manimlib/mobject/value_tracker.py` と `mobject_update_utils.py`: 状態を一つに持ち、図を更新する構成。実装は CE の ValueTracker / always_redraw を使用。

## 検証と制約

最終検証数値は `validation_summary.json` と作業レポートに保存しています。

最終版の実測値（今回実施分）:

- `py_compile`、既存の `verify_numerics.py` の5群が成功。最終コードでキャッシュ無効のフルレンダリングを完了（8シーン、95アニメーション）。
- ffprobe: H.264 / AAC、854×480、15fps。映像 591.864067秒 → 605.930733秒（+2.376672%）。音声 605.973333秒、映像との差 0.042600秒。
- `silencedetect=noise=-45dB:d=3`: 長い無音 0件。`volumedetect`: 平均 -26.8dB、最大 -5.7dB。
- 119文の `audio_query` を取得し、新規・変更5文の読みを確認。既存114文のPCM完全一致、変更しない7シーンのWAV・manifest一致。
- 全119字幕を組版し、最大幅 9.389844、高さ 0.699531。記号の読み仮名検索0件。字幕とPCM開始の差は最大 5.7e-11秒。
- 最終29フレームを目視。全8シーンと、復習・補足の各段階および前後を確認。2か所・3操作で語句の時刻と動作を照合し、差は最大 0.033109秒。
API の読みの全文照合は、実音声の全編通し聴取を意味しません。全編の通し聴取、高品質レンダリング、全フレームの目視は実施していません。

音声クレジット: **VOICEVOX:WhiteCUL**
