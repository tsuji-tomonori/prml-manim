# PRML 3.5 エビデンス近似

「どの曲線を信じますか？」から始める、8分56.9秒・10シーンの日本語解説動画です。Manim Community と VOICEVOX:WhiteCUL（ノーマル、speaker 23）を使用します。

[動画（480p15）](media/videos/prml_3_5_evidence_approximation/480p15/PRML35EvidenceApproximation.mp4) ／ [全台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## シーン

| シーン | 開始秒 | 尺（秒） | 視覚的アイデア | 原文参照 |
|---|---:|---:|---|---|
| scene01 どの曲線を信じますか？ | 0.000 | 50.600 | 正則化のつまみが同じ回帰曲線を変える | pp.165–166,171–172 / (3.74)–(3.76), Fig.3.16 |
| scene02 何を積分し、何を選ぶ？ | 50.600 | 52.800 | 事後曲線の集まりから精度の集中へ | pp.165–166 / (3.74)–(3.77) |
| scene03 山の高さより、重なる面積 | 103.400 | 54.533 | 正規化した事前の幅と尤度との積分面積 | p.166 / (3.77)–(3.78) |
| scene04 積分を、山の高さと幅へ | 157.933 | 55.133 | 曲率で山の幅を変え、対数エビデンスへ | p.167 / (3.79)–(3.86) |
| scene05 次数を増やせば、有利？ | 213.067 | 53.800 | 整数の次数に到達してから評価点を打つ | pp.167–168 / (3.86), Fig.3.14 |
| scene06 データが決める方向は？ | 266.867 | 52.133 | 尤度の楕円と二方向の縮小率 | pp.168–170 / (3.87)–(3.91), Fig.3.15 |
| scene07 十個の重みを、何個使った？ | 319.000 | 55.467 | 固有方向の棒から係数の軌跡へ | pp.169–172 / (3.90)–(3.92), Fig.3.17 |
| scene08 二本の交点が、山の頂点 | 374.467 | 51.333 | 二本の交点とエビデンスの頂点を連動 | pp.169,171–172 / (3.89)–(3.92), Fig.3.16 |
| scene09 ノイズも、一緒に推定する | 425.800 | 55.600 | 残差・自由度・両精度の実反復を連動 | pp.169–172 / (3.92)–(3.99) |
| scene10 選んだ後も、不確かさは残る | 481.400 | 55.467 | 事後標本から予測帯、動く入力断面へ | pp.165–166,171–172 / (3.74)–(3.75),(3.98)–(3.99); predictive variance: (3.59) |

## 原文と数値例

Bishop (2006), *Pattern Recognition and Machine Learning*, §3.5、印刷 pp.165–172（PDF pp.185–192）を抽出して参照しました。式 (3.74)–(3.99)、図3.14–3.17に対応します。この節に表はありません。予測分散には前節の (3.59) を補助参照します。

- 本文の M は重みの個数です。図3.14で次数を表す M は、動画では d と書き分けます。
- 訓練18点、seed=35、生成ノイズ標準偏差0.25。九つのガウス基底（中心0〜1、幅0.13）と定数で M=10。独立なテスト200点は seed=351 で、答え合わせだけに使用します。
- 多項式比較は中心化した入力 2x−1 の単項式、α=0.005、β=16 を固定。最大は d=3、対数エビデンス −17.823351。原文の図のデータ・尺度・数値を複製していません。
- β=16固定のエビデンス最大は ln α=1.175819、α=3.240795、γ=5.735996、対数エビデンス −12.469110。
- 両精度の再推定は α=20、β=2 から30回。最終 α=3.678368、β=10.581769、γ=5.288678、対数エビデンス −11.925703。両更新は同じ直前の事後平均と γ を使います。反復間は精度の対数を補間する演出です。
- 1次元の積分例は t=1.2、観測標準偏差0.35。尤度と事前の実際の積を描き、数値積分を解析値と照合します。描画外の裾も数値計算・表示値に含みます。
- 二方向の楕円は λ=(0.7,12)、最尤位置=(2,1.5) の説明用の例です。精度の集中は模式図と明示します。
- 固有値が0の方向も許す半正定値の場合を含め、0≤γ≤M。大標本の簡略式は全方向が決まり、N≫M の条件で表示します。
- 固定した精度の下で重みのガウス積分は厳密です。精度を代表点に固定することが近似です。精度の事前が比較的平らで、事後が集中する条件を説明します。
- 再推定式は停留条件です。大域的最大への収束、テスト誤差の最小点との一致、一般的な不偏性を保証する説明にはしていません。
- 予測帯は平均±2予測標準偏差。観測ノイズと重みの不確かさを含み、精度自体の不確かさは含みません。

## ファイル

- `prml_3_5_evidence_approximation.py`: 公開クラス `PRML35EvidenceApproximation` を維持した動画本体。
- `evidence_model.py` / `verify_numerics.py`: 数値実験と独立な数式・データ整合検証。
- `narration_content.py`: 全121文の字幕 display と読み上げ speech の正本。
- `caption_layout.py` / `scene_support.py`: 1.1の実測文字高・数式本体中心・文PCM同期を継承。
- `make_voicevox_narration.py`: 音声生成、再開、台本・WAVハッシュ確認。
- `check_narration_readings.py` / `reading_check.md` / `reading_check.json`: 修正前後の全文読み。
- `export_narration_script.py` / `narration_script.md`: 原文・視覚操作・実測尺を含む全台本。
- `assets/voicevox/`: 新10 WAVとmanifest。旧音声はすべて置換。

## 再生成

このディレクトリから実行します。VOICEVOX Engine は http://127.0.0.1:50021 を使用します。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -W error -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_3_5_evidence_approximation.py PRML35EvidenceApproximation
```

`make_voicevox_narration.py --from-scene scene04` で途中再開できます。文音声はworktree内 `.working/voicevox-lines/` にキャッシュされます。欠落・古い音声は描画開始時にエラーにし、無音版の公開を防ぎます。

## 演出の参照

- [3b1b/videos: BellCurveArea](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/gauss_int/integral.py): 面積を積分と対応させ、幅を動かす。
- [3b1b/videos: BayesDiagram](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2019/bayes/part1.py): 積の面積とベイズの数式を色で結ぶ。
- [3b1b/videos: CLT の get_variable_display](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/clt/main.py): tracker から数値・つまみ・図形を更新する。
- [3b1b/manim: ValueTracker](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py) と `mobject_update_utils.py`: 共有状態から連動させる設計。

ManimGL の実装は持ち込まず、Manim CE で演出を実装しました。

## 検証

最終版は86アニメーション。映像536.866667秒、音声536.896000秒（差0.029333秒）、長い無音0件、平均音量−26.3 dB、ピーク−6.4 dB。最終46画像（全10記号字幕を含む）と3シーンの同期を確認しました。全121字幕の最大幅9.374219、最大高0.699531で安全領域内です。数値検証・全Pythonの構文検証に成功しています。詳細と抽出時刻は `validation_results.json` と[作業レポート](../../../reports/working/20260927-2017-prml-3-5-3b1b-remake.md)に記録しています。全編の通し聴取は未実施です。APIから取得した全文の読みを目視確認する検査と、耳による聴取は区別します。

音声クレジット: VOICEVOX:WhiteCUL。
