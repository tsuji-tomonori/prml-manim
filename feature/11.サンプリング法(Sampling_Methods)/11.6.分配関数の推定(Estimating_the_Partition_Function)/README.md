# PRML 11.6 分配関数の推定

## 物語の設計

全編の問いは「右の山にほとんど届かない標本から、2つのモデルの重みの比をどう測るか」。11.5までの標本生成では、密度を定数倍した形だけで十分だった。モデル比較ではその定数、分配関数の比が必要になる。説明用の2つのガウス形エネルギーを使う。基準の山は中心0、対象の山は中心4。基準から80点を引くと、重点サンプリングによる比は約0.241だが、厳密な比は3。同じ例に8個の近い橋を架けると約2.963となる。冒頭で結果を予想させ、最後に同じ乱数列と値へ戻る。青は基準・標本、黄は重みと操作、緑は推定結果、紫は比較、赤は大きなずれ。

| シーン | 直前の疑問 → 実験 → 発見 |
|---|---|
| 1 | 標本を引けるのにモデルの重みが見えない？ → 2つの山と80点、0.241対3を見せる → 比の測り方が問いになる |
| 2 | なぜ前節のサンプラーは定数を知らなくてよかった？ → 同じ曲線を縦に伸縮して標本の位置を保つ → 分配関数は面積を1にする量 |
| 3 | モデル比較で何を比べる？ → 2つの面積と同じ事前確率のモデルを対応させる → 必要なのは分配関数の比 |
| 4 | 面積を全部足さず比を出せる？ → 2.1の期待値を復習して重みを標本へ掛ける → 式(11.72)の重点サンプリング、1/Lを含む平均 |
| 5 | なぜ0.241と外れた？ → 同じ軸で基準と対象の山、少数の大きな重みを点灯 → 重なりの乏しさで分散が大きい |
| 6 | 近い山なら測れる？ → αのつまみを0から1へ動かして中間分布を作る → 隣どうしの比を測る連鎖法 |
| 7 | 小さい比をどう戻す？ → 8個の比を掛け、途中の分配関数を消す → 約2.963、連鎖の平衡維持が条件 |
| 8 | 冒頭の失敗をどう説明する？ → 同じ80点と2つの山を再表示 → 直接法は右の山を見逃し、橋を架けると比3に近づく |

## 視覚補足と章をまたぐ復習

- 重みの平均を使う直前に2.1「期待値」の復習カードRを入れる。複数の値に確率重みを付ける図をこの節の標本重みへ結び、過去節をimportしない。
- 式(11.72)の比→期待値では、基準分布から引いた点に黄色い重みを載せて平均を示す短い視覚補足を入れる。分配関数の面積は帯の面積で示し、高さを確率そのものと呼ばない。
- 復習カードは実測11.400秒で、音声481.067秒の約2.4%。別の視覚補足は式(11.72)の平均を本編の点と重みの動きに統合し、独立した待機時間は増やしていない。合計は本編の15%以内。字幕の数式は記号で、読みは音声側に置く。

## 原文との照合

Bishop (2006) §11.6、印刷pp.554–555（PDF pp.574–575）、式(11.71)–(11.75)の本文と紙面を確認した。公式正誤表では式(11.72)最終行と(11.73)右辺にそれぞれ`1/L`が欠ける。映像は修正後の平均を使う。[公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)。

物語の中心は、正規化定数とモデル比較、重点サンプリングの比、分布の重なり、連鎖法、中間エネルギー、積の望遠鏡的な消去である。式(11.73)のマルコフ遷移の混合分布は位置付けだけを述べ、具体的な遷移核の構成・証明は扱わない。複雑な実モデルの周辺尤度計算も扱わない。

## 数値例と近似の条件

説明用の連続1変数モデルで、基準`G(z)=z²/2`、対象`E(z)=(z−4)²/2−ln 3`。従って`Z_G=√(2π)`、`Z_E=3√(2π)`、厳密な比は3。`E_α=(1−α)G+αE`、`α=k/8`（`k=0,…,8`）。その正規化分布は平均`4α`、分散1のガウスであり、中間の分配関数は`Z_α=√(2π) exp[-8α(1−α)+α ln 3]`。NumPyの`default_rng(5)`で、直接法は基準から80個を引き、各橋では対応する中間分布から独立に80個ずつ引く。直接法は約0.241117、橋の8比の積は約2.963202。両手法の標本数は同じではなく、優劣の一般的保証ではない。中間分布を解析的に標本化できる簡単な教材例であり、実際の難しい分布ではマルコフ連鎖を使い、各段階で平衡に近い状態を保つ必要がある。この2つのエネルギーは面積比を見せる教材で、実データの尤度と事前には分解していない。モデル比較では対応する未正規化の積がエビデンスになる。音声・映像の数値は検証スクリプトで再計算する。

## ファイルと再生成

[完成動画（480p15）](media/videos/prml_11_6_partition_function/480p15/PRML116PartitionFunction.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

- `partition_model.py` / `verify_numerics.py` / `numerical_validation.json`: 直接法と8段階の比、解析的な正解を再計算する。
- `narration_content.py` / `narration_script.md`: 8シーン・48拍・106文の字幕 `display` と読み上げ `speech`。
- `make_voicevox_narration.py` / `assets/voicevox/`: WhiteCUL ノーマル（23）、話速1.08、抑揚1.08の8 WAVとSHA-256・文時刻付きmanifest。
- `check_narration_readings.py` / `reading_check.json` / `.md`: 全106文の `audio_query` 読み。
- `scene_support.py` / `prml_11_6_partition_function.py`: PCM時刻と同期した字幕、2つの山、αのつまみ、標本と比の図。
- `verify_caption_layout.py` / `caption_validation.json`、`verify_video.py` / `video_validation.json`: 全字幕と完成映像の検査。

対象ディレクトリでメイン作業ツリーの仮想環境を使う。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_11_6_partition_function.py PRML116PartitionFunction
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

VOICEVOX Engineが停止中なら、`.working/voicevox-engine-0.25.2/linux-cpu-x64/run --host 127.0.0.1 --port 50021`で起動し、自分で起動したものは音声生成後に停止する。辞書登録はしない。レビュー用フレームはGit管理外。

## 検証と限界

- `py_compile`、`verify_numerics.py`、`verify_caption_layout.py`でコード・解析解・標本実験・全106字幕の寸法を検査した。字幕の最大幅は9.047、最大高さは0.453 Manim単位。`reading_check.json`はVOICEVOXの全106文の`audio_query`を記録し、誤読箇所を`speech`で修正した。
- 全編を`--disable_caching --flush_cache -ql`で再レンダリングした。`verify_video.py`は、音声manifestのハッシュ、8シーン・48拍の字幕時刻、MP4のH.264/AAC、854×480・15fps・481.067秒、音声相関、3秒以上の無音0件、平均音量−26.3dB・最大−5.8dB、全編デコードを確認する。代表248フレームを発話・強調の前後と途中から抽出し、16枚のシートにまとめる。
- 代表フレームを目視し、枠、軸・目盛り・数値・注記の重なり、曲線の範囲、字幕と数式、英語注記、強調と曲線の同期を確認した。レビュー画像は`media/review/`に残すが、Git管理しない。全編の実時間での視聴・試聴と1080pでの確認は実施していない。
- 比較は固定乱数の教材実験であり、直接法と連鎖法の標本数は異なる。実際の複雑なモデルでのマルコフ連鎖の混合速度や推定分散は検証対象外。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL
