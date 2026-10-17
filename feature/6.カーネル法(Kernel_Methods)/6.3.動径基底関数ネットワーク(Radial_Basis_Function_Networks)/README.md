# PRML 6.3 動径基底関数ネットワーク

## 物語の設計

[完成動画（480p15）](media/videos/prml_6_3_radial_basis_function_networks/480p15/PRML63RadialBasisFunctionNetworks.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

全編の問いは「3点をぴったり通る局所的な予測が、少し外へ行くとどうして消えるのか」。6.2と同じ入力 $x=(-1,0,1)$、目標 $t=(1,0,1)$ を使う。右端の未観測点 $x_*=2.5$ の目標を、最後の点 $t=1$ に近いと予想させる。幅 $\sigma=0.7$ の3つのガウス基底を各点に置き、全点を補間する重みを求めると、3点では誤差0なのに $f(2.5)=0.131365$。なぜ外で消えるのかを、山の高さ・重み・正規化という順で調べる。最後は同じ3点、同じ $x_*=2.5$、同じ幅で正規化した Nadaraya–Watson 回帰を計算し、$y(2.5)=0.983401$ に戻る。外挿値はモデルの選択に依存し、遠方の予測精度を保証するものではない。

| シーン | 開始秒 | 尺 | 疑問→実験→発見 | 原典 |
|---|---:|---:|---|---|
| 1 消える予測 | 0.000 | 65.200秒 | 6.2の距離の山を3点へ置く→外で小さい | §6.3 pp.299–300, (6.38) |
| 2 山を並べる | 65.200 | 57.333秒 | 中心からの距離を動かす→3つの山の線形和 | (6.38) |
| 3 ぴったり通る代償 | 122.533 | 64.267秒 | 最小二乗で重みを求める→訓練3点は一致、外は不確実 | p.299 |
| 4 山の合計を1に | 186.800 | 55.533秒 | 同じ軸で生の山と正規化した山を切り替える | (6.40)–(6.41), 図6.2 |
| 5 近い点の投票 | 242.333 | 60.600秒 | 同じ3点・同じ幅で分子と分母を動かす | (6.45)–(6.46) |
| 6 確率からも同じ比へ | 302.933 | 63.067秒 | 入力と目標の密度を重ね、条件付き平均へ | (6.42)–(6.47), 図6.3 |
| 7 中心を減らすと？ | 366.000 | 61.800秒 | 中心を3本から2本へ減らし、最小二乗で係数を学習 | pp.300–301 |
| 8 同じ点へ戻る | 427.800 | 63.467秒 | 入力2.5の0.131365と0.983401を再比較 | (6.38), (6.41), (6.45) |

## 視覚補足と章をまたぐ復習

- シーン2の直前に「3.1 入力を特徴へ」の復習カード R（約12秒）。青い入力から緑の基底3本へ写す図を再現する。過去節を import しない。
- シーン3の前に1.1の「点を全部通っても未知点を保証しない」を短く復習（約12秒）。過学習という語はノイズがあるデータでの危険として使い、今回の3点が実際にノイズを含むとは断定しない。
- シーン4の分数の前に、3つの高さを足して各々を合計で割る本文カード（約15秒）。分母が正の位置に限る。シーン6の条件付き平均の前に、密度の高さから面積を集める補足（約15秒）。合計は全編の15%以内。
- 補足は入力=青 `#58C4DD`、操作=黄 `#FFFF00`、結果=緑 `#83C167`。装飾枠を置かない。数式は画面、意味は発話へ分ける。

## 原文との照合

Bishop (2006), *Pattern Recognition and Machine Learning*, 印刷pp.299–303（PDF pp.319–323）の全文を `pdftotext -layout` で読み、図6.2・6.3を画像で確認した。[公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf) のp.300の訂正に従い、(6.39)直後の関数は $y(x)$、(6.40)の左辺も $y(x)$ とする。原典の「各点に1中心」「正規化」「少数中心」「Nadaraya–Watson」という道筋を扱う。Green関数と変分法による一般的導出、直交最小二乗・K-meansの詳細、混合ガウスの一般化は物語に必要な範囲を超えるため省く。

## 数値例と近似の条件

- 自作の1次元3点 $x=(-1,0,1)$、$t=(1,0,1)$。6.2の点と目標を再利用。未観測入力 $x_*=2.5$、共通のガウス幅 $\sigma=0.7$。
- 生の基底は $h_n(x)=\exp[-(x-x_n)^2/(2\sigma^2)]$。補間の重みは3×3の行列 $H_{in}=h_n(x_i)$ を `numpy.linalg.solve(H,t)` で求める。今回は $H$ が正則である。この例の完全一致を一般のデータ・基底へ無条件に広げない。
- 正規化は $q_n(x)=h_n(x)/\sum_m h_m(x)$。今回のガウスは有限の $x$ で分母が正。Nadaraya–Watson の係数は $t_n$、補間で学習した $w_n$ を流用しない。確率的導出では各成分密度の目標方向の平均を0とする。入力方向の山は正のガウス形で、共通の正規化係数は比で相殺される。
- 図6.3に対応する2次元の楕円は位置関係の模式図であり、幅や面積を原典のデータから推定したものではない。正規化係数 $q_n(x)$ は $x$ について正規化した重みで、一般に対称な半正定値カーネルとは限らない。
- $x_*=2.5$ は訓練範囲の外。正規化回帰の値は右端の目標1へ近づくが、外挿の真値が1であるとは主張しない。

## ファイルと再生成

- `rbf_model.py`、`verify_numerics.py`、`numerical_validation.json`: 補間、正規化、2中心の最小二乗を自作データから再計算。
- `narration_content.py`、`narration_script.md`: 8シーン・56 beat・112文の `display` / `speech`。
- `make_voicevox_narration.py`、`assets/voicevox/`: WhiteCUL 8 WAVとハッシュ・字幕時刻付き manifest。
- `check_narration_readings.py`、`reading_check.json` / `.md`: 全112文の `audio_query` 読みと修正履歴。辞書登録なし。
- `scene_support.py`、`prml_6_3_radial_basis_function_networks.py`: PCM同期の字幕・描画と時刻記録。
- `verify_caption_layout.py`、`verify_video.py`: 字幕寸法、完成MP4・WAV・タイムラインの整合。

対象ディレクトリで、メインの `.venv` を使って実行する。プレビューと抽出フレームはGit管理外。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_6_3_radial_basis_function_networks.py PRML63RadialBasisFunctionNetworks
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

## 検証と限界

- `py_compile`、数値再計算、全112字幕の寸法検証を通過。字幕の最大幅10.100、高さ0.330（Manim座標）で安全領域内。字幕と画面テキストの個数・番号・倍率の漢数字、英語の説明注記を検索して該当なし。
- 台本、`display` / `speech`、8 WAVのSHA-256、manifestの音声尺、112件の読み記録が一致。`audio_query` の全文を目視し、「負→マケ」「通っても→カヨッテモ」などを `speech` で修正。辞書登録なし。
- 指定のキャッシュ無効・消去コマンドで全56拍をレンダリング。MP4は854×480・15fps、491.265689秒（8分11秒）。AAC音声は491.306667秒で尺差0.040978秒。`ffprobe` と全編デコードを通過。
- `silencedetect=noise=-45dB:d=3` で3秒以上の無音は0件。`volumedetect` は平均-26.6dB、最大-6.4dB。場面時刻と音声尺の最大差は約4e-12秒。
- 初回完成版から全56拍の序盤・中盤・終盤と場面末の176フレームを抽出し、8場面の一覧を目視。シーン2で残った曲線と数式の重なりを修正。修正版を全編再レンダリングし、再抽出した176フレームのうちシーン2の22フレームと、冒頭・結末・補足・復習・つまみ・密度図の拡大代表8フレームを目視。軸ラベル、数値ラベル、強調中の図、字幕と式を確認。抽出画像はGit管理外。
- 数値: 補間重みは $(1.320943,-0.952262,1.320943)$、訓練3点の予測は $(1,0,1)$。$x_*=2.5$ で生の補間は0.131364689、正規化回帰は0.983400921。2中心の説明例の訓練二乗誤差は0.401649651。いずれも `verify_numerics.py` が元データから再計算。

全編の実時間通し聴取、全フレーム目視、720p以上のレンダリングは未実施。外挿の真値は設定していないため、どちらの予測が実際に正しいかは検証していない。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL（ノーマル、speaker 23、話速1.08、抑揚1.08）。動画内には表示しない。
