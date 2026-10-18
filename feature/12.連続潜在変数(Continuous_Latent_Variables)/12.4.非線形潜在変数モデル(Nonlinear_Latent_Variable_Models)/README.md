# PRML 12.4 非線形潜在変数モデル

## 物語の設計

全編の問いは「曲がった帯状の13点を、1本の潜在軸で表して、同じ形へ戻せるか」。12.1 と 12.2 の PCA／確率的 PCA は直線状の潜在空間を使い、12.3 は曲がった得点の等高線を見せた。今回はその得点から観測点を再構成する方向へ進む。冒頭では13点を PCA の第1軸へ落とすと、点から再構成線まで大きな隙間が残る。最後に**同じ13点**を、固定した説明用の曲線写像で再構成し、差を再計算する。

| 場面 | 直前の疑問 → 実験 → 発見 | 原典 |
|---|---|---|
| 1 | この曲がった点群を1本の軸で再現できる？ → PCA の直線に落とす → 端で大きく外れる | §12.4導入、§12.4.3 |
| 2 | 直線が拾えない情報は曲がり方だけ？ → 2音源の線形混合を眺める → ガウス分布の回転対称性も障害になる | §12.4.1、(12.89) |
| 3 | 無相関なら独立？ → 依存する無相関の点群と独立な音源を比較 → 非ガウス分布を使う独立成分分析 | §12.4.1、(12.90)訂正 |
| 4 | 入力をそのまま出力にすればよい？ → 幅の狭い自己連想ニューラルネットワークへ通す → 単純な構成の最良解は PCA の部分空間 | §12.4.2、図12.18、(12.91) |
| 5 | 間に曲げる層を足すと？ → 符号化 $F_1$ と復号 $F_2$ を動かす → 非線形な多様体へ写せる | §12.4.2、図12.19–12.20 |
| 6 | 曲線そのものをどう決める？ → 点を最近接位置へ投影し、近くの点の平均で曲線を更新 → 主曲線の考え方 | §12.4.3、(12.92) |
| 7 | 曲線の上で確率を扱える？ → 局所の直線を継ぐ混合モデルと潜在格子を比較 → 確率的 PCA の混合と生成位相写像 | §12.4.3、図12.21 |
| 8 | 元の13点をどう説明する？ → 同じ点の直線再構成と曲線再構成を切替 → 潜在軸は1本でも形を曲げられる | §12.4全体 |

## 視覚補足と章をまたぐ復習

- 場面4に「復習: 5.3 誤差逆伝播」を短く置き、再構成誤差を小さくする学習の入口を示す。過去節の import はしない。`display` では `5.3`、`speech` では「五章三節」と読む。
- 場面3では無相関と独立の違いを少数の点の同時変化で、場面6では条件付き平均を「同じ投影先に集まる点の平均」で短く補う。どちらも直後の図を読むためだけに使う。
- 同じ章の 12.1–12.3 は冒頭と遷移中の一言でつなぎ、独立の復習カードを重ねない。補足・復習の合計は全編の15%以内にする。
- 青は観測、黄は注目する操作、緑は再構成、紫は比較、赤はずれ。軸・目盛り・数値の背面に装飾枠や台を置かない。

## 原文との照合

Bishop (2006) §12.4、印刷 pp.591–599（PDF pp.611–619）を通読し、図12.18–12.21を PDF 画像で確認した。[公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf) では p.592 の式(12.90)の分子を `2` に訂正する。場面中の数式はこの訂正に従う。p.593 の「activations」→「activation」は表記訂正で、映像の数学には影響しない。

中心に置くのは独立成分分析、自己連想ニューラルネットワーク、主曲線、生成位相写像。独立因子分析の EM、LLE、Isomap、潜在特性モデル、密度ネットワーク、SOM、GTM の拡張は、全編の問いから外れるため導出しない。GTM の画面は1次元の説明用格子9点へ一様重みを置いた模式図で、GTM を学習した結果ではない。原典の図12.21は石油データの PCA と GTM の比較。本編の13点は自作例であり、その結果を再現したものではない。

## 数値例と近似の条件

説明用データは $t$ を $-1.8$ から $1.8$ まで $0.3$ 刻みで取った13点。基準曲線は $f(t)=(t,0.48(t^2-1.08))$。各点に指定の小さな法線方向のずれを足す。PCA は13点の中心化共分散を $N$ で割って計算する。非線形側の $F_1(x)=x_1$、$F_2(t)=f(t)$ は**学習結果ではなく固定した説明用写像**である。再構成誤差は各点の二乗距離の平均で比べる。`verify_numerics.py` が全数値を再計算する。13点の中心は $(-0.004249,0.0864)$、共分散の固有値は $0.289465,1.262087$。直線の平均二乗再構成距離は $0.289465$、固定曲線は $0.0103842$。比は約 $27.88$。両者とも同じ13点に適用し、曲線の写像は訓練で推定していない。

## シーンと実測尺

| シーン | 開始秒 | 音声尺（秒） | 主な画面 |
|---|---:|---:|---|
| 1 谷の13点 | 0.000 | 66.933 | PCA の直線と赤い残差 |
| 2 2本の声 | 66.933 | 62.067 | 2波形の混合と逆変換 |
| 3 無相関と独立 | 129.000 | 62.267 | 無相関でも曲がった点の関係 |
| 4 狭い通り道 | 191.267 | 60.067 | 自己連想の層と5.3の復習 |
| 5 曲がった復元 | 251.333 | 67.267 | 固定写像の潜在軸と曲線 |
| 6 主曲線 | 318.600 | 65.200 | 投影と平均による曲線の更新 |
| 7 GTM | 383.800 | 62.133 | 格子点、非線形写像、密度 |
| 8 同じ13点 | 445.933 | 63.867 | 直線と曲線の誤差を切替 |

全編の音声尺合計は509.800秒。映像の実測尺は完成後に記録する。

## ファイルと再生成

[全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

- `nonlinear_model.py` / `verify_numerics.py` / `numerical_validation.json`: 13点、PCA、固定曲線、無相関だが依存する点、既知の係数による音声混合を再計算する。音声例は ICA の係数を学習した結果ではない。潜在音源の波形は正弦波と疎なパルスで、独立性の統計的検証には使わない。
- `narration_content.py` / `narration_script.md`: 8シーン・48拍・96文の字幕 `display` と読み `speech`。
- `make_voicevox_narration.py` / `assets/voicevox/`: WhiteCUL ノーマル（23）、話速1.08、抑揚1.08の8 WAVと文時刻・ハッシュ付き manifest。
- `check_narration_readings.py` / `reading_check.json` / `reading_check.md`: 全96文の audio_query と修正前後の読み。
- `scene_support.py` / `prml_12_4_nonlinear_latent_variables.py`: PCM 時計に同期する字幕、13点の再構成、音源波形、主曲線と潜在格子の図。
- `verify_caption_layout.py` / `caption_validation.json`、`verify_video.py` / `video_validation.json`: 全字幕の安全範囲と完成 MP4 の技術検査。

対象ディレクトリでメイン作業ツリーの仮想環境を使用する。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_12_4_nonlinear_latent_variables.py PRML124NonlinearLatentVariables
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

VOICEVOX Engine が停止中なら、メイン作業ツリーの `.working/voicevox-engine-0.25.2/linux-cpu-x64/run --host 127.0.0.1 --port 50021` で起動し、自分で起動した Engine は作業後に停止する。抽出フレームは Git 管理外。

## 検証と限界

制作後、実施した MP4 検証、代表フレーム目視、未実施事項を分けて記録する。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL
