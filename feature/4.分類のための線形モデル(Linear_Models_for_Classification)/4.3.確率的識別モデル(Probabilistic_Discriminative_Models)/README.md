# PRML 4.3 確率的識別モデル

「境界の近くでは、何パーセント？」から始まる **8分24.4秒・9シーン** の日本語動画です。問いを出し、曲線や点を連続的に動かし、数式で確かめます。

[動画（480p15）](media/videos/prml_4_3_probabilistic_discriminative_models/480p15/PRML43ProbabilisticDiscriminativeModels.mp4) ／ [収録台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

## シーンと原文

Bishop (2006), §4.3.1–4.3.6、印刷 pp.203–213（手元 PDF pp.223–233）を抽出して確認しました。本節に表はありません。

| シーン | 秒 | 視覚的な実験 | 原文参照 |
|---|---:|---|---|
| 境界の近くでは、何パーセント？ | 57.467 | 入力・確率・傾き・切片を連動 | pp.203–205、(4.87) |
| 曲がった境界を、直線で表せる？ | 60.333 | 二乗基底へ点群を移し、直線を円へ戻す | pp.204–205、Fig.4.12 |
| 間違った自信を、どう測る？ | 60.067 | 正解確率と対数損失、予測差の線 | pp.205–206、(4.88)–(4.91) |
| 全部当たると、学習は終わる？ | 50.800 | 完全分離で重みを増大、正則化で戻す | p.206 |
| 曲線の学習を、最小二乗に戻せる？ | 65.067 | 接線から有効目標、分散、実計算したIRLS | pp.207–208、(4.92)–(4.103) |
| 三つの確率を、合計一にするには？ | 49.933 | スコアと確率棒、共通シフトの不変性 | pp.209–210、(4.104)–(4.110) |
| しきい値が揺れると、確率になる？ | 45.867 | ガウス密度の左面積とCDF上の点 | pp.210–211、Fig.4.13、(4.111)–(4.116) |
| 遠くの誤ラベルに、どれだけ引かれる？ | 51.667 | 同じ軸の損失比較、反転率と確率上下限 | p.212、(4.117) |
| なぜ、同じ勾配が何度も現れる？ | 63.200 | 平均・自然パラメータの往復と微分の相殺 | pp.212–213、(4.118)–(4.124) |

## 数式と数値例

- 図を複製せず、内外それぞれ18点の同心円、12点の二値ラベルなどの自作データを使います。二乗基底は Fig.4.12 の構造を説明する独自例で、原図のガウス基底ではありません。
- 点数・確率・損失・更新値は `discriminative_model.py` で NumPy により計算します。IRLS は Newton 更新と重み付き最小二乗の一致を確認しています。反復間の連続移動は表示用の補間です。
- 完全分離では有限の最尤解が存在しない例を示します。L2 正則化の例では切片も含めて正則化します。訓練正解率から未知データでの確率精度を保証しません。
- 分散 R は曲率の重みです。R が小さくても有効目標 z は大きくなり得るので、確信の強い点を単純に無視するとは説明しません。
- softmax の最大スコアを引いて計算します。全スコアへの共通シフトは確率を変えず、無制約な全クラス重みの表現には冗長性があります。
- probit の活性化は標準正規CDF、統計でいう probit リンクはその逆です。比較する logistic は中央の形を近づけるためスコアを1.7倍します。図の密度は拡大せず、表示区間外の小さい裾だけを省略します。数値CDFは全区間の積分に対応します。
- ラベル反転モデルは `P(t_obs=1|a)=epsilon+(1-2epsilon)sigma(a)`。独立かつ一定の反転率を仮定します。
- [著者の正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)に従い、凸性の記述、(4.110)の符号、(4.115)–(4.116)のerf、(4.124)の勾配表記を訂正して解釈します。映像では尺度 `1/s` を保持します。

## 字幕・読み・同期

`narration_content.py` の50 beat・101文が正本です。`display` は漢字と数式記号、`speech` は音声用の読みです。MathTex の本体を日本語かなの高さ・中心線にそろえ、左右に余白を設けます。1.1 の修正済み実装をこの節内へ継承しました。

全 speech を speaker 23 の `audio_query` で取得し、「全問」「二色」「誤ラベル」の誤読を修正、再取得しました。修正前後は `reading_check.md/json` に保存します。音素列の確認と通し聴取は別で、全編の通し聴取は未実施です。

文ごとのPCM尺・字幕時刻・台本とWAVのハッシュを manifest に保存し、映像も同じ時計を使います。音声欠落・不整合時は停止します。旧8 WAVは新台本で置換し、scene09を追加しました。

## 再生成

対象ディレクトリで実行します。Manim Community、既存venv、Noto Sans CJK JPを使用します。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
# reading_check.md の全行を確認し、必要な speech を修正して再実行
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py --captions
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_4_3_probabilistic_discriminative_models.py PRML43ProbabilisticDiscriminativeModels
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py --output /tmp/prml43-review
```

VOICEVOX Engine は `http://127.0.0.1:50021`、WhiteCUL ノーマル（23）、話速1.08、抑揚0.95。`--from-scene scene05` などで再開できます。接続できないときは停止し、コンテナは操作しません。

音声クレジット：**VOICEVOX:WhiteCUL**。

## 3Blue1Brown の参照

[3b1b/videos](https://github.com/3b1b/videos) の `_2017/nn/part1.py`（スコアを確率へ）、`_2017/nn/part2.py`（予測と目標の差から損失へ）、`_2023/clt/main.py`（面積と確率の連動）を参照しました。[3b1b/manim](https://github.com/3b1b/manim) の ValueTracker / updater の考え方を Manim CE で実装し、ManimGL のコードは取り込んでいません。

検証の数値は `numerical_results.json` と `validation_results.json`、画像抽出と同期照合は `verify_video.py` に保存します。検証スクリプトだけでは目視済みとせず、画像を確認した範囲を別途記録します。

## 最終検証

映像は H.264・854×480・15fps・504.399344秒、音声は AAC・504.426667秒です。ストリームの差は0.027323秒、3秒以上の無音は0件でした。平均音量−26.6dB、最大−5.6dBです。

数値検証8群、字幕101文の安全領域、全50beat・全11記号字幕・3シーン同期の計67画像を確認しました。全編の通し聴取と全フレームの人手検査は未実施です。

[作業完了レポート](../../../reports/working/20260928-0134-prml-4-3-3b1b-remake.md)に原文対応・参照ファイル・実測値・修正記録をまとめています。
