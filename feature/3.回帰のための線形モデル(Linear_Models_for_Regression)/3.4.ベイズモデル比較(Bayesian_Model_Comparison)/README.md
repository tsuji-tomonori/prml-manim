# 3.4 ベイズモデル比較

約10分40秒、9シーンの日本語動画です。「一番よく合う係数」と「モデル全体を平均した説明力」の違いを、面積・分布の幅・確率の更新として動かして示します。

[動画（480p15）](media/videos/prml_3_4_bayesian_model_comparison/480p15/PRML34BayesianModelComparison.mp4) / [全台本](narration_script.md) / [全文の読み確認](reading_check.md)

## シーン構成

| シーン | 尺（秒） | 視覚的な操作 | 原文の印刷ページ |
|---|---:|---|---|
| 1 点に合う曲線なら、どれでもよい？ | 55.867 | 同じ点群の上で直線→二次→七次、係数の別の可能性を重ねる | pp.161–162、式3.66,3.68 |
| 2 係数を動かすと、積分が見えてくる | 80.533 | 尤度×事前の曲線、面積を掃く、長方形の和、事後への正規化 | p.162、式3.68,3.69 |
| 3 モデルの確率は、どう更新される？ | 63.000 | 同じ棒を事前→積→事後へ変形、事前のつまみ | pp.161–162、式3.66、ベイズ因子 |
| 4 よく合う範囲は、全体のどれくらい？ | 78.733 | 幅を二倍にすると高さ半分、面積比から体積比へ | pp.162–163、図3.12、式3.70–3.72 |
| 5 予測を広げると、どこが薄くなる？ | 78.400 | 正規分布を同じ軸上で広げ、観測点での高さを追う | pp.163–164、図3.13 |
| 6 回帰でも、積分した数で比べよう | 71.267 | 次数と回帰曲線・実計算したエビデンスを連動 | pp.161–164、式3.68、図3.13の回帰例 |
| 7 モデルを平均すると、山は一つになる？ | 66.400 | 二峰の予測分布、混合の重みのつまみ、平均の位置 | p.162、式3.67 |
| 8 正しいモデルは、必ず勝つ？ | 70.200 | コイン六回の対数比を確率で重み付けし、正の平均へ | p.164、式3.73 |
| 9 事前も含めて、モデルを確かめる | 75.133 | 事前幅とエビデンス、独立テスト、三段階のまとめ | pp.164–165、§3.5への接続 |

## 原文・数値例

Bishop (2006) の印刷pp.161–165（PDF pp.181–185）を `pdftotext -layout` で抽出し照合しました。重要な式は(3.66)–(3.73)、図は3.12–3.13。この節に表はありません。原図を転写せず、以下の自作例を計算しています。

- 一係数の積分: 観測1、ノイズ標準偏差0.35、事前は中心0・幅4/8/40の一様分布。エビデンスは約0.249466/0.125/0.025。連続観測については確率密度であり、面積が確率です。
- オッカム因子: 一様事前・鋭い山の近似。長方形の有効幅は `sqrt(2π)×0.35`（半値全幅ではない）。多変量の単純例は各方向の幅比1/2。同じ幅比と相関に関する条件を明示しています。
- 図3.13の構造: 一観測 `D~N(0,s²)`、標準偏差0.35/1.3/3、観測1.4。これは `w~N(0,s²−0.2²)` と `D|w~N(w,0.2²)` の周辺化でも得られます。等しいモデル事前の事後確率は約0.001312/0.589538/0.409151。原文の一般的なデータ集合空間を、一観測の例として具体化しました。
- 回帰: seed=34、`x=linspace(-1,1,12)`、生成関数 `0.25+0.25x+0.9x²`、ノイズ標準偏差0.18。Legendre基底、全係数に `N(0,I)` の事前、次数0〜7。基底の尺度と事前は一体です。入力xを固定した観測tの密度を比較します。観測共分散 `C=0.18²I+ΦΦᵀ` の正規密度で積分を計算し、最大値を恣意的に正規化していません。二次の対数エビデンスは−3.139577、七次は−9.745197。最大対数尤度は5.003498→8.372967に増加します。次数間の連続変形と点を結ぶ線は表示用の補間で、非整数次数の確率モデルではありません。
- モデル平均: 平均−1.35/+1.35、標準偏差0.32の二つの正規密度を混合。重み0.5から0.85へ動かします。予測の平均一点と、二峰の密度を区別しています。
- 期待対数ベイズ因子: 表の確率0.65と0.35の候補、六回の表の数の二項分布。真が0.65のとき、対数比の平均は1.114271、誤った候補が有利になる確率は0.117424です。

式(3.72)直後の原文の増減の説明は、表示されている式の符号と整合しません。動画は式に従って、当てはまりの改善で対数尤度が増加し、幅比が1未満なら `M ln r` は減少すると説明します。式(3.73)はベイズ因子そのものではなく、その**対数の期待値**です。真の分布が候補内にある条件も明示しました。

## 3Blue1Brownから参考にした手法

ManimGLの実装を移植せず、Manim CEで再実装しています。

- [ProbabilityBar（Bayes）](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2019/bayes/part1.py): 同じ確率の棒を連続的に更新し、分布と数値を連動。
- [BuildUpGaussian](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/clt/main.py): 同じ座標上の幅・高さ・面積の変化、数式と図の色対応。
- [BellCurveArea](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/gauss_int/integral.py): 面積を細い長方形に分け、積分記号へ結ぶ。
- [ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py) / [updater helpers](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/mobject_update_utils.py): 一つの状態からつまみ・曲線・面積・数値を再描画。

## 再生成

このディレクトリで実行します。既存のvenvを使い、VOICEVOX Engineのコンテナ操作は行いません。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python narration_content.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_3_4_bayesian_model_comparison.py PRML34BayesianModelComparison
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

音声は `--from-scene scene04` などで再開可能。文単位の合成キャッシュは節内 `.working/voicevox-lines/`、PCM尺・両台本・ハッシュは `assets/voicevox/manifest.json` に保存します。字幕は1.1と同じ日本語の実測かな高・数式本体の中心線・左右余白を使います。古い音声はハッシュ不一致で拒否します。

`verify_video.py` はffprobe、3秒以上の無音、音量、57区切り・全記号字幕・同期画像を収集します。画像を人が見たことや、通し聴取を実施したことを自動検査の成功で代用しません。生成キャッシュや中間動画はGit管理しません。

## 音声

VOICEVOX:WhiteCUL（ノーマル、speaker=23）、Engine 0.25.2、話速1.05。displayとspeechは分離し、114文のAPI読みを全文確認しています。確認できた誤読を修正し、修正前後を `reading_check.md` とJSONへ保存しました。全編の通し聴取は未実施です。

## 最終検証

構文確認、数値・音声整合の6検証群、キャッシュ無効の全編レンダリングが成功。映像639.533333秒、AAC音声639.573333秒（差0.040秒）。3秒以上の無音0件、平均−26.6 dB、最大−7.0 dB。最終MP4から70枚（各シーン6枚以上、全7記号字幕を含む）を確認し、3シーンでPCM発声と動作を照合しました。

[作業レポート](../../../reports/working/20260927-1948-prml-3-4-3b1b-remake.md)に検証条件・修正記録・残る制約を記載しています。
