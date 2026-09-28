# PRML 5.7 ベイズニューラルネットワーク

約9分20秒・9シーンの日本語動画です。重みの候補が予測の幅に変わる過程を、連続変形と実際の数値計算で追います。Manim Community と VOICEVOX:WhiteCUL（ノーマル、speaker 23）を使用します。

[動画（480p15）](media/videos/prml_5_7_bayesian_neural_networks/480p15/PRML57BayesianNeuralNetworks.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## シーン

| シーン | 開始秒 | 尺（秒） | 視覚的アイデア | 原文 |
|---|---:|---:|---|---|
| 1 点のない場所で、どこまで信じる？ | 0.000 | 58.467 | 結びつきの重み・曲線・候補の束を連動 | pp.277–279 / §5.7, (5.161), (5.168) |
| 2 データは、重みの候補をどう変える？ | 58.467 | 59.267 | 同じ重み軸上で事前×尤度→事後 | pp.278–279 / (5.161)–(5.165) |
| 3 谷の曲がり方から、重みの幅へ | 117.733 | 60.867 | 谷底の断面方向と局所ガウスの幅を連動 | pp.278–279 / (5.165)–(5.167) |
| 4 重みの雲を、予測の軸へ写す | 178.600 | 62.733 | 小さな重み変化を元の関数と線形化した関数に通す | p.279 / (5.168)–(5.172) |
| 5 予測の幅には、二つの原因がある | 241.333 | 65.333 | 入力つまみ・予測帯・二色の分散棒 | pp.279–280 / (5.171)–(5.173) |
| 6 よく当たる一点か、よく当たる広がりか | 306.667 | 56.933 | 事前の幅と積の面積、ラプラスの行列式項 | pp.280–281 / (5.174)–(5.176) |
| 7 どの方向が、データで決まった？ | 363.600 | 71.000 | 有効パラメータ数・実際の再推定・ユニット交換 | pp.280–281 / (5.177)–(5.180)、対称性 |
| 8 分類でも、重みの広がりを残す | 434.600 | 62.600 | 同じ分類図で弱い正則化→エビデンス選択 | pp.281–283 / (5.181)–(5.185), Fig.5.22 |
| 9 境界は同じでも、自信の強さは変わる | 497.200 | 62.600 | 活性の分散→確率平均→等確率線の広がり | pp.283–284 / (5.186)–(5.190), Fig.5.23 |

## 原文との照合

Bishop (2006), *Pattern Recognition and Machine Learning* §5.7–5.7.3、印刷 pp.277–284（PDF pp.297–304）を `pdftotext -layout` で抽出して参照しました。図5.22・5.23はPDF画像でも確認。本節に表はありません。

[公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf) p.15 に従い、(5.181) の総和記号、(5.183) の不要な定数項、(5.190) の引数を修正しています。(5.190) は `σ(κ(σ_a²) a_MAP)` です。非線形ネットワークでは `bᵀw_MAP` は一般に `a_MAP` と一致しません。

## 数値例と近似の条件

- 回帰: 自作12点、入力1・隠れ tanh 2・出力1、バイアスを含む7パラメータ。全パラメータを学習。事前精度 α=0.8、観測精度 β=64。SciPy の最小二乗最適化と NumPy による解析的ヤコビ行列を使用します。最終モデルは `model_data.npz` に保存。
- ラプラスの精度行列は **A=αI+βH**。H は二乗誤差のヘッセ行列で、残差に由来する二階微分も含みます。勾配を中心差分して求め、独立な目的関数の二階差分で照合します。共分散は A⁻¹。A の最小固有値は1.698267です。
- シーン2: 最小固有値の方向を座標軸に取り、他の直交座標をMAPの値に固定した条件付き密度です。事前の中心は0。尤度の導入中は `p(w) p(D|w)^r` の r を0から1へ動かします。表示外を含む区間[-12,12]で数値正規化。全重みの周辺事後とは区別します。
- シーン3: MAPを通る断面を固有方向の間で回転。右のガウスはその断面方向の条件付き局所近似で、分散は `1/(dᵀAd)` です。縦の範囲を固定して急な方向と緩い方向を比較します。
- シーン4: ガウスの二つの固有方向内で、標準偏差の0.16倍の小さな変位を一周させます。ガウス化と出力の局所線形化は別の近似です。重み図の楕円はこの二方向の相対的な幅を表します。
- シーン5: 帯は `y_MAP±2√(β⁻¹+gᵀA⁻¹g)`。重みだけの分散は x=-2.4, 0, 2.4 で0.051688, 0.006632, 0.052080。ノイズ分散は0.015625です。共分散の縮小は思考実験で、追加データによる再学習を装ったものではありません。幅は全モデルで距離に単調とは限りません。
- シーン6: 面積の説明例は `p(t=1.2|w)=N(1.2|w,0.35²)` と `p(w)=N(w|0,s²)`。全実数上のエビデンスは `N(1.2|0,s²+0.35²)`。これはネットワーク全体の証拠ではありません。
- シーン7: 四方向の棒は λ=(0.1,1,10,100) の説明例。実ネットワークの再推定では α が0.8→1.133457、β が64→37.585917、γ が5.976827→5.527019（4回更新）。非線形モデルの再推定式は固有値のハイパーパラメータ依存を無視した近似です。正定値の事後ヘッセ行列と、尤度ヘッセ行列の半正定値性は別の条件です。
- 原文の tanh 二層ネットワークでは、隠れユニットの交換・符号反転から M!2^M 個の同等な山が生じます。モデル比較時の局所証拠の補正として紹介。縮退した同等解を無条件に別の山として数える主張ではありません。
- 分類: 自作二次元50点（seed=571）、隠れ tanh 4、バイアスを含む17パラメータ。ベルヌーイ尤度を使用し、回帰の β は使いません。A は `αI + ∇∇[-log p(D|w)]`。α候補0.015, 0.06, 0.2, 0.6, 1.5の各6初期値から学習した局所解を比較し、局所ラプラス証拠が最大の0.6を採用。αについての大域最適化ではありません。
- Fig.5.22 の比較を自作データで再構成。元図の厳密な最尤解の代わりに、数値発散を避ける弱い正則化 α=0.015を起点にします。原図の8隠れユニットやデータを転写していません。
- シーン9の活性平均2・分散9では、80点 Gauss–Hermite 積分の平均確率は0.717424、κ近似は0.718946（差0.001522）。代表値だけなら0.880797です。境界の不変性は、同一MAP・ガウス近似・活性の線形化・κ近似の下での性質。任意の厳密なBNNの保証ではありません。
- 分類の等高線はこの節内の NumPy marching squares 実装で描きます。確率0.5は活性0の等高線を使い、格子上の補間誤差による見かけの境界移動も避けます。

## 3Blue1Brown の参照

数式は説明の冒頭で完成形を表示し、軸上の数値を変えない色の強調を使います。演出の考え方を Manim CE で実装し、ManimGL のコードは持ち込んでいません。

- [videos/_2017/nn/part2.py](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/nn/part2.py), `need_a_cost_function` / `break_down_cost_function`: 観測との違いを先に見せ、数式の各項と対応させる。
- [videos/_2017/nn/part3.py](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/nn/part3.py), `InterpretGradientComponents`: 結びつき・重みの操作・関数の変化を連動させる。`network.py` の forward / backprop も確認。
- [videos/_2017/eoc/chapter10.py](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2017/eoc/chapter10.py), `ConstructQuadraticApproximation`: 同じ座標で元の関数と局所二次近似を重ねる。
- [videos/_2019/bayes/part1.py](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2019/bayes/part1.py), `BayesDiagram`: 事前と尤度の積を、確率の配分と面積で見せる。
- [videos/_2023/gauss_int/integral.py](https://github.com/3b1b/videos/blob/ae2b911326b8255dfae790b6e8764a8a400c5875/_2023/gauss_int/integral.py), `BellCurveArea`: 積分の区間を掃き、面積を量として見せる。
- [manimlib/mobject/value_tracker.py](https://github.com/3b1b/manim/blob/fafa083a4fb274bba9cabde0b6e2f50ba6da0622/manimlib/mobject/value_tracker.py) と `mobject_update_utils.py`: 一つの状態をつまみ・曲線・数値が共有する考え方。CE の ValueTracker / updater を使用。

## ファイルと再生成

- `narration_content.py`: 54 beat・109文の display / speech と原文参照。
- `scene_support.py`: 1.1を経由した4.5の実測かな高・数式本体中心・余白・文PCM同期の仕組み。字幕のない古い音声ではレンダリングを停止します。
- `make_voicevox_narration.py`: WhiteCUL音声、台本とWAVのハッシュ照合、途中再開。`assets/voicevox/` は9 WAVとmanifestが一致します。
- `check_narration_readings.py`, `reading_check.md/json`: 全109文の修正前後の読み。見つかった「負→マケ」「別の山→別のサン」は、speechだけを「ふ」「やま」に修正して再照会しました。
- `bayesian_model.py`, `model_data.npz`, `contour_lines.py`: 学習と数値描画。
- `verify_numerics.py`, `verify_caption_layout.py`, `verify_video.py`: 数値、字幕安全領域、メディア・時刻の検証。検査結果は各 `*_validation.json`。

以下は、この節のディレクトリで実行します。既存venvを使用し、追加依存のインストールは不要です。

```bash
curl --fail http://127.0.0.1:50021/version
/home/t-tsuji/project/prml-manim/.venv/bin/python bayesian_model.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_5_7_bayesian_neural_networks.py PRML57BayesianNeuralNetworks
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

音声生成の再開は `make_voicevox_narration.py --from-scene scene05`。APIが停止している場合は起動・停止操作を行わず、接続エラーを報告します。生成キャッシュと抽出画像はGit管理外。最終動画は480p15のMP4だけを管理します。

## 検証と限界

数値検証では、ヤコビ行列の独立差分との最大差4.40×10⁻¹¹、ヘッセ行列の方向二階差分との差1.57×10⁻⁶、射影分散と12万標本の相対差0.002873、エビデンスの数値積分との差2.66×10⁻¹⁴、隠れユニット交換による出力差1.11×10⁻¹⁶を確認しています。

字幕109文は幅12.9・高さ0.9の安全領域内（最大幅9.391、高さ0.701）。記号の読み仮名カタカナはdisplayデータにありません。APIの全文読み確認は、全編の通し聴取を意味しません。通し聴取、全フレームの目視、高解像度版のレンダリングは未実施です。

最終MP4は854×480・15fps、映像559.799344秒、音声559.829333秒（差0.029989秒）。`silencedetect=noise=-45dB:d=3` の長い無音0件、平均音量−26.4dB、最大−6.4dB。54 beat・記号字幕15件・同期9時点の合計78枚を抽出して目視し、scene01・05・09のPCM開始時刻と動作区間を照合しました。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL
