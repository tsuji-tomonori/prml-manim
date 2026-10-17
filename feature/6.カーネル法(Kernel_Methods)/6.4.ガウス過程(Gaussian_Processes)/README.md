# PRML 6.4 ガウス過程

## 物語の設計

[完成動画（480p15）](media/videos/prml_6_4_gaussian_processes/480p15/PRML64GaussianProcesses.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

全編の問いは「学習点から離れた $x=2.5$ で出た予測を、どのくらい信じられるか」。6.3 と同じ入力 $(-1,0,1)$、目標 $(1,0,1)$、未観測入力 $x_*=2.5$ を使う。6.3 の正規化回帰はここで $0.983401$ と返したが、予測の幅を示さなかった。ガウス過程では、長さ尺度 $\ell=0.7$、信号分散 $1$、独立な観測ノイズの標準偏差 $0.2$ を**説明用に固定**する。冒頭で「右端の点が1なら、2.5でも1に近いか」と予想させ、平均が約 $0.123$、観測値の標準偏差が約 $1.014$ という結果を先に見せる。なぜ平均が0へ戻り、幅が広がるのかを、関数の候補、点どうしの共分散、観測ノイズ、条件付き分布の順に追う。最後に同じ3点、同じ入力、同じハイパーパラメータへ戻り、数値と帯を再確認する。これは真の外挿値を知ったという話ではない。

| シーン | 開始秒 | 尺 | 疑問→実験→発見 | 原典 |
|---|---:|---:|---|---|
| 1 予測だけで足りる？ | 0.000 | 62.933秒 | 6.3の点を外へ動かす→平均と幅の食い違いを予想する | §6.4 pp.303–304, §6.4.2 |
| 2 曲線をくじにする | 62.933 | 54.400秒 | 重みの候補から曲線を数本描く→関数を確率で考える | (6.49)–(6.55), 図6.4 |
| 3 近さが連動を決める | 117.333 | 62.400秒 | 2点を動かし共分散を色・線で示す→カーネルが連動を決める | (6.55)–(6.56), 図6.4–6.5 |
| 4 観測は揺れている | 179.733 | 56.800秒 | 潜在曲線の点にノイズを足す→対角成分だけ観測の分散が増える | (6.57)–(6.63), 図6.6 |
| 5 点を見た後の分布 | 236.533 | 65.333秒 | 2変数の楕円を条件付きで切る→平均は移動し幅は縮む | (6.64)–(6.67), 図6.7 |
| 6 遠くで帯が広がる | 301.867 | 60.733秒 | 同じ軸で未観測点を右へ動かす→共有する情報が減る | (6.66)–(6.68), 図6.8 |
| 7 幅をどう決める？ | 362.600 | 55.667秒 | 長さ尺度とノイズのつまみを動かす→周辺尤度で選ぶ考えを示す | §6.4.3, (6.69) |
| 8 最初の点へ戻る | 418.267 | 66.533秒 | $x_*=2.5$ の同じ条件に戻る→平均と予測帯を読み直す | (6.66)–(6.67), 図6.8 |

## 視覚補足と章をまたぐ復習

- シーン2に「3.3 重みの事前分布から関数へ」の復習カード R（9.133秒）。青い重みから緑の曲線が変わる図を自作再現する。
- シーン5に「2.3 2変数ガウスの条件付き分布」の復習カード R（8.867秒）。楕円を縦に切る図を自作再現する。
- シーン3の共分散には、2点が同じ向きに動くほど値が大きいという視覚補足（8.533秒）。シーン5の行列式には、既知点と新点を色分けした2×2の視覚補足（8.400秒）。4件の合計34.933秒は本編484.800秒の7.2%。
- 入力は青 `#58C4DD`、操作は黄 `#FFFF00`、結果は緑 `#83C167`、事前は紫 `#9A72AC`、観測は赤 `#FC6255`。復習カードは左上に節番号を記し、囲みは文字・式・目盛りから離す。本文のグラフ背面に装飾枠を置かない。

## 原文との照合

Bishop (2006), *Pattern Recognition and Machine Learning*, §6.4 印刷 pp.303–320（PDF pp.323–340）の全文を `pdftotext -layout` で読み、図6.4–6.12を画像で確認した。動画に使う図は自作の数値例で再構成した。[公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf) の p.310 の削除指示を確認した。物語の中心は回帰の (6.49)–(6.69) とする。指数カーネルとの標本比較、定数・線形項を足したカーネル、$O(N^3)$ の計算量の詳細、周辺尤度の勾配 (6.70)、入力依存の観測ノイズ、自動関連度決定、分類、ラプラス近似、ニューラルネットとの極限の詳細は、本編の問いから外れるため省いた。後半の拡張はシーン8で次の問いとして触れる。

## 数値例と近似の条件

- 自作の3点 $x=(-1,0,1)$、$t=(1,0,1)$。6.3 と同じ観測を使用。未観測入力は $x_*=2.5$。
- $k(x,z)=\exp[-(x-z)^2/(2\ell^2)]$、$\ell=0.7$、$\beta^{-1}=0.04$。事前平均は0。潜在関数と観測値の予測分散は区別し、本編の帯は観測値の $m\pm2\sigma_t$ とする。正規分布でこの幅を「約95%」と読む際は、固定したハイパーパラメータの下でのモデル上の確率である。
- この3点だけで長さ尺度とノイズを学習したとは主張しない。シーン7のつまみと周辺尤度は考え方の例であり、結末の主例は固定値に戻す。シーン7の周辺尤度3件はノイズ標準偏差を0.2に固定し、長さ尺度だけを変えた値。
- 事前曲線は seed=6402、観測ノイズの模式図は seed=6430 で再現する。模式図の赤い点は冒頭の3点とは別に、固定オフセット $(0.22,-0.15,0.18)$ を足した例である。

## ファイルと再生成

- `gp_model.py`、`verify_numerics.py`、`numerical_validation.json`: 3点のカーネル行列、予測分布、周辺尤度を再計算。
- `narration_content.py`、`narration_script.md`: 8シーン・56拍・112文の `display` / `speech`。
- `make_voicevox_narration.py`、`assets/voicevox/`: WhiteCUL 8 WAV、字幕時刻、台本・音声ハッシュの manifest。
- `check_narration_readings.py`、`reading_check.json` / `.md`: 全112文の `audio_query` と修正前後の読み。辞書登録なし。
- `scene_support.py`、`prml_6_4_gaussian_processes.py`: PCMに同期した字幕と描画。
- `verify_caption_layout.py`、`verify_video.py`: 全字幕の寸法、完成動画と音声・時刻の整合。

このディレクトリで、メインの `.venv` を使って再生成する。VOICEVOX Engine は音声・読みの再生成時のみ `127.0.0.1:50021` で起動する。フレームとプレビューはGit管理外。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -W error -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_6_4_gaussian_processes.py PRML64GaussianProcesses
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

## 検証と限界

全編のレンダリングと映像検証が完了したら結果を記録する。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL（ノーマル、speaker 23、話速1.08、抑揚1.08）。動画内には表示しない。
