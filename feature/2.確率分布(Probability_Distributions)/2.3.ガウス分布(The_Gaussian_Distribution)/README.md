# PRML 2.3 ガウス分布

中心と広がりのつまみから、楕円の幾何、観測による更新、柔軟な分布へ進む日本語動画です。
全11シーン、約10分55秒。Manim Community、VOICEVOX:WhiteCUL ノーマル（speaker 23）、数式記号の字幕を使用します。

[動画（480p15）](media/videos/prml_2_3_gaussian_distribution/480p15/PRML23GaussianDistribution.mp4) ／ [全台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## 構成

| シーン | 秒 | 視覚的な問い・操作 | 原文（印刷ページ） |
|---|---:|---|---|
| 1 山の形は、何で決まる？ | 56.267 | 平均・標準偏差を動かし、区間を面積に変える | pp.78–79、式2.42 |
| 2 偶然を平均すると、なぜ山になる？ | 57.533 | 2万回の平均のヒストグラム、標準化 | pp.78–79、図2.6 |
| 3 丸い点群を、楕円へ伸ばす | 59.467 | 点群・等密度線・固有軸を同時に伸縮・回転 | pp.78,80–83、式2.43–2.64、図2.7 |
| 4 形を簡単にすると、何を失う？ | 62.067 | 一般→対角→等方を同じ点群の上で比較 | pp.83–85、図2.8 |
| 5 一方を知ることと、忘れること | 58.933 | 水平切断と条件付き密度、走査による周辺化 | pp.85–90、式2.81–2.98、図2.9 |
| 6 ノイズのある観測を、逆にたどる | 60.067 | 傾きとノイズ、観測の切断と事後の山が連動 | pp.90–93、式2.99–2.117 |
| 7 点から中心と広がりを学ぶ | 62.067 | 残差、最尤分散の偏り、逐次的な平均の移動 | pp.93–97、式2.118–2.126 |
| 8 中心そのものも、確かではない | 61.667 | 平均の事後を縮め、精度のガンマ事後へ移る | pp.97–102、式2.137–2.157、図2.12–2.14 |
| 9 外れた点に、引っ張られすぎない？ | 58.200 | 外れ値移動、ガウスと固定自由度tの最尤フィット | pp.102–105、式2.158–2.165、図2.15–2.16 |
| 10 角度の平均は、反対向き？ | 59.600 | 矢印の平均、円上の密度、集中度と方向のつまみ | pp.105–110、式2.167–2.186、図2.17–2.20 |
| 11 二つの集団を、一つの山で表せる？ | 59.400 | 成分の加算、混合比、動く一点と負担率の棒 | pp.110–113、式2.188–2.193、図2.21–2.23 |

## 原文と数値例

Bishop (2006), *Pattern Recognition and Machine Learning*, §2.3、印刷 pp.78–113（ローカルPDF pp.98–133）を読み、式と図の構造を照合しました。この節に表はありません。PDFや原図は収録せず、自作のデータと図を使います。

- 中心極限定理は独立・同分布・有限分散の例です。平均そのものは分散が縮むため、標準化した量の極限を示します。seed=2302、2万試行、N=1,2,10,32。棒の遷移中は表示の補間であり、非整数個の観測を意味しません。
- 楕円は共分散の固有分解から計算します。Δ=1 の半軸は固有値の平方根、密度比は exp(−1/2)。Δ=2 の外輪も表示します。白色化で元の丸い点群に戻ります。平面の縦横の単位長をそろえています。
- 条件付きの例は平均0、共分散 `[[1.4,0.85],[0.85,1]]`。x_b=1 で平均0.85、分散0.6775。周辺分散は1.4。走査中の曲線は −∞ から走査高さまで積分した同時密度で、最後に全実数の積分へ切り替えます。条件付けによる分散の減少は一般に半正定値の意味であり、独立なら減りません。
- 線形ガウスは x∼N(0,1)、y=ax+ε、ε∼N(0,s²)。一般形の Λ と L はそれぞれ事前と観測ノイズの精度行列、A は線形写像、b は切片です。S と m は事後共分散と平均です。
- 最尤推定は独立なガウス観測を仮定します。seed=2307 の16点で平均0.325439、分散0.966002。分散の偏りは seed=2317、N=4 の6000標本で検証し、最尤分散の平均0.747605、不偏補正後0.996806。累積平均は20回目から表示します。逐次更新は同じデータの4→5点で実計算しています。
- 平均のベイズ推定は事前N(0,1)、既知観測分散0.36、seed=2308 の10観測。最終事後平均0.614127、分散0.034749。途中の曲線は実測更新の端点間の視覚的補間です。精度の推定へ切り替えた後は平均0.8を既知とし、事前Gam(2,1)を更新します。ガンマの第2引数はrateです。
- t分布の比較は自由度3固定で位置・尺度を数値最適化します。seed=2309 の24点に、0→7へ動かす1点を追加。外れた点を動かす間は71個の最尤解を補間します。ガウスの平均移動は0.280000、tの位置移動は0.021682。これは特定の数値例です。λは逆尺度二乗で、tの分散の逆数とは一般に異なります。Γは正規化に現れるガンマ関数です。
- 周期変数は5°と355°の自作例。atan2で象限も扱い、平均ベクトルがゼロなら方向は未定義です。フォン・ミーゼスの図は円からの外向き距離を密度に比例させた表示で、囲まれた面積を確率とは解釈しません。I₀は正規化に用いる修正ベッセル関数です。
- 混合分布はseed=2311,2312、45点と55点。単一ガウスは全点の最尤解、混合は生成パラメータ（平均−1.7,1.5、標準偏差0.55,0.7）を表示します。混合モデルのEM学習を実行した図ではありません。混合比を変えた後も観測点は固定しています。

全9小節の主要な役割をつなぐ概説です。平方完成・ブロック行列の導出、Robbins–Monroの一般収束論、ガウス・ガンマ／ウィシャートのパラメータ更新の導出、フォン・ミーゼスの集中度の最尤解、EMの反復手順は省略しています。式2.129の符号や式2.187の展開を動画では使用せず、逐次平均の式2.126と円平均のatan2を直接検算しています。

## 3Blue1Brownから参考にした手法

ローカルの参照cloneを読み、Manim CEで演出を再実装しました。ManimGLのコードは取り込んでいません。

- [videos/_2023/clt/main.py](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/clt/main.py): 平均・標準偏差の線、同じ軸での分布の変化、trackerとスライダー。
- [videos/_2023/gauss_int/herschel.py](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/gauss_int/herschel.py): 点群・同密度の輪・半径を結ぶ見方。
- [videos/_2016/eola/chapter3.py](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2016/eola/chapter3.py): 一つの線形写像で点群全体を動かす。
- [videos/_2019/bayes/part1.py](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2019/bayes/part1.py): 観測に合う部分に絞り、正規化して条件付きへ移る。
- [manim/value_tracker.py](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py) と [mobject_update_utils.py](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/mobject_update_utils.py): 単一の状態から関連図形を更新する設計。

## 再生成

対象ディレクトリで、既存venvと稼働中のEngineを使います。

```bash
curl --fail http://127.0.0.1:50021/version
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python write_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_assets.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_2_3_gaussian_distribution.py PRML23GaussianDistribution
/home/t-tsuji/project/prml-manim/.venv/bin/python validate_video.py
```

`narration_content.py` が正本です。displayは字幕、speechは収録文。1.1と同じ日本語の実測文字高・数式本体の中心線・余白を使い、WAVの文境界に字幕と動作を同期します。台本とWAVのハッシュが合わなければ本編レンダリングを停止します。
`--from-scene scene05` などで音声生成を再開できます。`reading_check.md` は全speechのaudio_query結果であり、聴取済みを意味しません。

`validate_video.py` はffprobe・無音・音量の記録と、各beat・全記号字幕・3場面の同期比較の画像を `media/review/` へ生成します。画像を人が確認する工程は別途必要です。`media/` の生成物はignoreされ、最終480p15 MP4だけを強制追加します。

## 検証した配布動画

2026-09-27の最終480p15 MP4は15,189,917 bytes。映像655.261133秒、音声655.296000秒（差0.034867秒）。3秒以上の無音は0件（−45 dB）、平均音量−26.4 dB、最大−6.3 dB。
全132文の読み確認、数値・字幕/音声整合性検査、全11シーンと全記号字幕を含む82画像の目視、3場面の発声区間と動作の照合を実施しました。全編の通し聴取は未実施です。

SHA-256: `d8d6f9efcf35f8c03632443ac77953a4ebec45bc685db4c7f87eb178126cfd4a`

## 音声クレジット

VOICEVOX:WhiteCUL（ノーマル、speaker 23）。Engine 0.25.2、話速1.08、抑揚0.95。
