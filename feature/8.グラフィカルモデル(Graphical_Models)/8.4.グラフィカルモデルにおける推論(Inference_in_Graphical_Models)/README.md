# PRML 8.4 グラフィカルモデルにおける推論

Manim と VOICEVOX:WhiteCUL（ノーマル、speaker 23、話速 1.08、抑揚 1.08）による日本語動画。

## 物語の設計

全編の問いは「観測した端のランプを手がかりに、真ん中のランプと4個全部の最良の並びをどう求めるか」。4個の二値ランプを鎖でつなぎ、右端 $x_4=1$ だけを観測する。冒頭に、$x_2$ 単独では1が約65.8%なのに、最も重い全体の並び $(1,0,0,1)$ では $x_2=0$ という意外な結果を置く。「同じ図からなぜ答えが食い違うのか」を予想させる。

8.3 の1画素ずつの ICM は全体の最良解を保証しなかった。今回はループのない小さな鎖で、全候補を明示し、和で求める周辺確率と最大値で求める全体の最良解を分ける。端から2値のメッセージを渡し、最後は冒頭と同じ因子表・同じ観測で2つの答えを再計算する。木なら正確に解ける一方、ループがある場合は計算量と近似の問題が残る。

## シーン

| シーン | 開始 / 尺 | 疑問 → 実験 → 発見 | 原典 |
|---|---:|---|---|
| 1 食い違う2つの答え | 0:00 / 1:01 | 4個のランプで単独の答えと全体の答えが違うのはなぜか → 同じ鎖の2つの問いを提示 | pp.393–394、8.4 全体 |
| 2 観測で重みを変える | 1:01 / 0:58 | 端の観測をどう使うか → 1.2 の正規化を復習し、観測に合う候補だけ残す | pp.394–395、(8.47)–(8.49) |
| 3 局所の表 | 1:59 / 1:03 | 全16候補を保存する必要があるか → 3枚の2×2因子表を鎖へ置く | pp.395–396、(8.49)–(8.53) |
| 4 先に足す | 3:02 / 0:54 | どの計算を先に済ませるか → 端から和を取り、2値の札へ圧縮する | pp.395–397、(8.51)–(8.56) |
| 5 両側のメッセージ | 3:56 / 1:01 | 真ん中の確率はどう決まるか → 左右の札を掛け正規化 → $p(x_2=1\mid x_4=1)=144/219$ | pp.396–398、図8.38、(8.54)–(8.57) |
| 6 因子グラフと木 | 4:57 / 0:55 | 枝が増えたらどうするか → 変数の丸と因子の四角、葉からの和積伝播 | pp.398–410、図8.39–8.52、(8.59)、(8.66)、(8.69) |
| 7 和を最大へ | 5:52 / 0:53 | 1つの全体配置はどう探すか → 同じ鎖で和を最大へ切り替え、最大和の札を前へ送る | pp.411–414、(8.89)–(8.100) |
| 8 道を逆にたどる | 6:46 / 1:02 | 各点で最大の色を選んでよいか → 前の状態を記録し、最後から逆追跡 → $(1,0,0,1)$ | pp.413–415、図8.53、(8.101)–(8.102) |
| 9 最初の鎖へ | 7:48 / 1:03 | なぜ2つの答えが異なったか → 同じ8候補で和144対75、単一最大48を照合 | pp.393–418、§8.4.6–8.4.8 |

## 視覚補足と章をまたぐ復習

- scene02 の最初の2拍で、1.2 の正の重みを合計で割る操作を節番号付き復習カード R として約16秒で再現する。`display` は `1.2`、`speech` は「一章二節」。元動画から import せず、青い候補札と緑の正規化だけを本節に再現する。
- scene04 では表の列と Σ の2値を足す図解（V01a）、scene05 では左右の札を掛けて正規化する図解（V02）、scene07 では和と最大の操作の違いを本文図の中に短く入れる。復習16.87秒と3つの図解25.07秒の合計は41.93秒、音声実測530.80秒の7.90%。
- 青は入力・観測、黄は現在の演算、緑は結果、紫は比較、赤は不一致。グラフの背面に装飾枠・台・影を置かない。復習カード R の細い輪郭のみ共通仕様に従う。

## 原文との照合

Bishop (2006), *Pattern Recognition and Machine Learning* §8.4、印刷 pp.393–418（PDF pp.413–438）を `pdftotext -layout` で確認。図8.37、8.38、8.40、8.45、8.48、8.49、8.51、8.53 を PDF 画像で確認した。[公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)では本節の (8.57) の添字と因子の引数順、図8.41の説明、(8.66)、(8.69)、(8.79)、(8.86)、(8.93)、(8.101)、および §8.4.6 の名称と文に訂正がある。採用する式・ラベルは訂正後に合わせる。

物語の中心は鎖での正確な周辺化と最良配置。因子グラフと木への拡張は図と局所規則に絞る。一般のジャンクションツリーの構成法、ループ付き信念伝播のスケジュール、グラフ構造の学習は限界と次の問いとしてのみ触れ、詳細な導出を扱わない。

## 数値例と近似の条件

- 4変数 $x_i\in\{0,1\}$。3因子の表は行が左の変数0,1、列が右の変数0,1の順で、$\psi_{12}=\left(\begin{smallmatrix}1&4\\4&4\end{smallmatrix}\right)$、$\psi_{23}=\left(\begin{smallmatrix}4&1\\3&3\end{smallmatrix}\right)$、$\psi_{34}=\left(\begin{smallmatrix}3&3\\1&3\end{smallmatrix}\right)$。すべて正で、本動画用に固定した説明用の例。
- $x_4=1$ を確実な観測として固定する。残る8配置の重みの合計は219。$x_2=1$ の4配置はいずれも36で合計144、$x_2=0$ の合計は75。したがって $p(x_2=1\mid x_4=1)=144/219\approx0.6575$。最重の単一配置は $(1,0,0,1)$ で重み48、条件付き確率 $48/219\approx0.2192$。
- この4変数例は原典の図やデータそのものではない。メッセージを正規化せずに途中計算し、最後に合計219で割る。観測なしの全候補は16、観測後は8。木に対する和積・最大和と逆追跡は厳密計算であり、近似ではない。

## ファイルと再生成

| 種類 | ファイル |
|---|---|
| 完成映像 | [480p15 MP4](media/videos/prml_8_4_graphical_inference/480p15/PRML84GraphicalInference.mp4) |
| 台本 | [実測時刻付き台本](narration_script.md)、[字幕と発話の定義](narration_content.py) |
| 数値 | [固定した因子表と計算](chain_model.py)、[独立した列挙検証](verify_numerics.py)、[結果](numerical_validation.json) |
| 音声 | [VOICEVOX 生成](make_voicevox_narration.py)、[scene01–09 WAV と manifest](assets/voicevox/)、[全文の読み確認](reading_check.md) |
| 映像 | [Manim 本体](prml_8_4_graphical_inference.py)、[字幕と音声同期](scene_support.py) |
| 検証 | [字幕寸法](verify_caption_layout.py)、[完成映像](verify_video.py)、[結果](video_validation.json) |

VOICEVOX Engine 0.25.2 が `localhost:50021` で起動した状態で、このディレクトリから実行する。worktree に `.venv` がない場合は、メインの `/home/t-tsuji/project/prml-manim/.venv` を使う。

```bash
python -W error -m py_compile *.py
python verify_numerics.py
python check_narration_readings.py
python make_voicevox_narration.py
python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_8_4_graphical_inference.py PRML84GraphicalInference
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

音声を変更したら `audio_query` の全文確認と WAV 生成をやり直す。scene 単位で再開する場合は `make_voicevox_narration.py --from-scene scene05` のように指定できる。台本と WAV のハッシュが一致しなければ Manim は描画を停止する。レビュー画像は `media/review/` に生成し Git 管理しない。

## 検証と限界

`python -W error -m py_compile *.py`、独立列挙の `verify_numerics.py`、126件の字幕寸法検査を通した。字幕の最大幅は9.345、最大高さは0.619（Manim座標）で、安全領域内に収まる。VOICEVOX の `audio_query` で126文の読みを確認し、6文の `speech` を修正した。`display` の数式表記と WAV のハッシュ・尺は manifest と照合した。

完成 MP4 は8分50.8秒、854×480・15 fps の H.264/AAC。`verify_video.py` で ffprobe、`silencedetect`（3秒以上0件）、`volumedetect`（平均 -26.5 dB、最大 -7.7 dB）、全編デコードを通した。126文の WAV と圧縮後音声の照合では、最小相関0.9994、最大時刻差0.0214秒。各拍の入り・途中・終わりと場面末尾から261フレームを抽出し、117フレームを18枚の一覧と拡大した要所のフレームで目視した。因子表の数値、ランプの状態、逆追跡の経路、ループと観測ラベル、字幕と式の可読性を確認した。数値結果と検査値は [video_validation.json](video_validation.json) に保存した。

480p15 が納品版。高解像度版と、完成音声の全編通し聴取は実施していない。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL
