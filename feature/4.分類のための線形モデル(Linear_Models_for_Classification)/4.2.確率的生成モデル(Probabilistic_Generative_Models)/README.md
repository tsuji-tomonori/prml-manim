# 4.2 確率的生成モデル

「この観測はどちらのクラスから来たか？」から始め、分布を動かして、ベイズ則・直線境界・最尤推定・ナイーブベイズの式へ進む動画です。9シーン、12分14.4秒。Manim Community、ナレーションは VOICEVOX:WhiteCUL（ノーマル、speaker 23）です。

動画：[`PRML42ProbabilisticGenerativeModels.mp4`](media/videos/prml_4_2_probabilistic_generative_models/480p15/PRML42ProbabilisticGenerativeModels.mp4)

## シーン

| シーン | 開始 / 尺（秒） | 視覚的な実験 | 原文 |
|---|---:|---|---|
| 1 この観測は、どちらのクラスから？ | 0.000 / 82.200 | 観測位置→二つの密度の高さ→正規化した棒 | pp.196–198、(4.57),(4.62) |
| 2 比を取ると、S字が現れる | 82.200 / 80.800 | 対数オッズを動かし、S字と事後確率を連動 | pp.197–198、(4.57)–(4.61)、Fig.4.9 |
| 3 丸い分布から、なぜ直線が出る？ | 163.000 / 90.200 | 同じ形の楕円、二次項の相殺、境界上と横断方向の動き | pp.198–199、(4.64)–(4.67)、Fig.4.10 |
| 4 観測は同じ。クラスの割合が変わると？ | 253.200 / 70.733 | 事前0.5→0.8→0.2→0.5、境界の平行移動 | p.199、(4.65)–(4.67) |
| 5 三つのクラスと、曲がる境界 | 323.933 / 78.067 | 三色の事後確率と、第三クラスの共分散変形 | pp.198–200、(4.62)–(4.63),(4.68)–(4.70)、Fig.4.11 |
| 6 分布の部品を、データから作る | 402.000 / 99.267 | 50点→クラス平均→残差の集合→共有共分散→分類 | pp.200–202、(4.71)–(4.80) |
| 7 一つの外れ値で、何が動く？ | 501.267 / 57.200 | 一点を移動し、平均・共分散・境界を再推定 | p.202、§4.2.2末尾 |
| 8 ゼロと一でも、足し算で分類できる？ | 558.467 / 88.667 | 組合せ→条件付き独立→ビットの追加と対数オッズの加算 | p.202、(4.81)–(4.82) |
| 9 共通部分を消すと、線形が残る | 647.133 / 87.267 | 指数型分布族の共通因子を消し、分類平面へ戻る | pp.202–203、(4.83)–(4.86)、§4.3冒頭 |

Bishop (2006), *Pattern Recognition and Machine Learning*, §4.2、印刷 pp.196–203（手元PDF pp.216–223）を抽出して参照しました。図4.9–4.11の構造を自作データで再構成しています。図4.9のprobit比較は省略。この節に表はありません。

## 数値例と成立条件

- 密度比較は平均−1.1と1.1、標準偏差0.9の一次元ガウス。曲線の高さは確率密度であり、事前を掛けた後は `q_k=p(x|C_k)p(C_k)` と表記します。
- 二次元の平均は `(-1.15,-0.45)`, `(1.15,-0.45)`, `(0,1.25)`。共有共分散は `[[0.52,0.16],[0.16,0.42]]`。第三クラスの共分散は `τ=0→1` で `[[1.1,-0.27],[-0.27,0.30]]` へ連続補間します。全状態で正定値です。
- 色は密度ではなく事後確率の混合比です。境界は NumPy で求めた対数スコアの等値線を、最大の二クラスが接する部分に限定します。原図のクラスの色順は使わず、本作では緑だけの共分散を変えます。
- 判定はクラス間の誤分類損失が同じ場合です。事前だけを変える実験では平均と共分散を固定します。
- 最尤推定用データは seed 4202、赤30点・青20点。事前は0.6。共分散は各クラスの平均からの残差外積を全50点で平均します。各クラスの共分散の分母も `N_k` であり、不偏推定の `N_k−1` ではありません。
- 外れ値実験は赤の最初の一点を `(4.3,3.2)` まで移し、毎フレーム再推定。平均の移動距離は0.214834、共有共分散のトレースは0.869110→1.696542です。
- 離散特徴は3単語の有無、出現確率は赤 `(0.8,0.65,0.25)`、青 `(0.2,0.35,0.7)`、事前は各0.5。`000→100→101` で赤の事後確率は `0.251799→0.843373→0.434783`。入力は0か1だけを取り、仮定は「クラスを条件とする独立性」です。
- (4.63)の対数同時密度そのものには入力の二次項が含まれます。(4.68)の線形スコアは全クラス共通の項を取り除いた、softmaxで同じ確率を与える表現です。画面の `≡` はこの同値性を示します。
- (4.71)は観測・ラベルの同時尤度として `L` と表記。指数型分布族は `u(x)=x`、共通スケールの条件を明示し、(4.85)–(4.86)を表示する際は `s=1` とします。一般の `u(x)` が非線形の場合まで、元の入力で線形とする説明はしていません。

## 字幕・読み・同期

`narration_content.py` に61 beat・122文を保存しています。`display` は記号で表した字幕、`speech` は読み上げ用の文です。字幕は1.1と同じ方法で、日本語かなの実測文字高を基準に数式を拡大し、添字を除く本体の中心と左右の間隔を合わせます。

`check_narration_readings.py` は全 speech の `audio_query` を取得し、初回と修正後の読みを `reading_check.md/json` に保存します。分数、同じ項、誤分類、値、節などの読みを修正しました。API読みの全文確認を実施しています。全編の通し聴取は実施していません。

各文のPCM長、字幕開始・終了時刻、台本とWAVのSHA-256をmanifestに保存します。映像と字幕は同じ時計で動き、15fpsの累積境界に合わせます。WAVの欠落・台本不一致を検出した場合は描画を停止します。

## 再生成

このディレクトリで実行します。Engine は `http://127.0.0.1:50021` を使用します。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
# reading_check.md の全行を確認し、speech を直した場合は再実行
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -W error -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_4_2_probabilistic_generative_models.py PRML42ProbabilisticGenerativeModels
/home/t-tsuji/project/prml-manim/.venv/bin/python review_video.py
/home/t-tsuji/project/prml-manim/.venv/bin/python validate_video.py
```

`make_voicevox_narration.py --from-scene scene04` のように途中再開できます。文WAVのキャッシュはworktree内 `.working/voicevox-lines/` に置きます。最終MP4以外のmedia、レビュー画像、キャッシュはGit管理しません。公開クラス名・動画本体のファイル名は維持しています。

## 演出の参照

- [3b1b/videos: BayesDiagram / HeartOfBayesTheorem](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2019/bayes/part1.py)：観測に対応する支持を取り出して正規化し、事前を変えた影響を追う。
- [3b1b/videos: IntroduceSigmoid](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/nn/part1.py)：入力範囲から確率の高さへの対応を連続的に見せる。
- [3b1b/videos: VariableC](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/gauss_int/herschel.py)：一つのtrackerに数値・つまみ・曲線を結び付ける。
- [3b1b/manim: example_scenes.py](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/example_scenes.py)：式の対応する項を保って変形する構成。

演出の考え方をManim CEで再実装しています。ManimGLのコードは取り込んでいません。字幕と音声同期はリポジトリ内1.1と3.5、連動する分布・等高線は3.3も参考にしました。

## 検証

数値検証の実測値は `numerical_results.json`、最終動画・音声・同期の実測値は `validation_results.json` に保存します。代表画像の目視範囲と未検証事項は、作業レポートに記録します。


最終検証：映像734.400秒、音声734.443秒、差0.042667秒、3秒以上の無音0件、平均−26.8 dB・ピーク−5.5 dB。76枚（全9記号字幕を含む）を目視し、3シーンの同期を照合。

[作業レポート](../../../reports/working/20260928-0057-prml-4-2-3b1b-remake.md)

## 音声クレジット

ナレーション：VOICEVOX:WhiteCUL
