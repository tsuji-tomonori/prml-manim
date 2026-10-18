# PRML 12.2 確率的PCA

## 物語の設計

全編の問いは「12.1の主成分の線上にある2点は、ともに再構成誤差0なのに、なぜ片方だけ珍しいと判断できるか」。12.1と同じ説明用13点の平均点Aと、平均から主軸方向へ6だけ離れた点Bを使う。冒頭で両者の再構成誤差が0と予想させる一方、点Bは訓練点群の外にあることを見せる。潜在変数の事前分布、線への写像、等方的なガウス雑音を順に足し、線の周りの密度を作る。最尤解で主軸と捨てた分散が現れることを確かめ、事後分布・EM・欠測とモデルの拡張をこの問いから必要な範囲で扱う。結末で同じA・Bの密度比約0.0103へ戻る。青は観測点、黄は操作・潜在変数、緑は密度・保持する構造、紫は比較、赤は雑音・誤差。

| シーン | 直前の疑問 → 実験 → 発見 |
|---|---|
| 1 | 線に近ければ普通の点か？ → 13点の主軸上にA・Bを置き再構成誤差を比較 → どちらも0で区別できない |
| 2 | 線上の位置に頻度を持たせるには？ → $z\sim\mathcal N(0,1)$を線へ写す → 中心付近が多い |
| 3 | 線の外の点はどう生む？ → 等方雑音を各線上位置に加える → 楕円状の密度が生まれる |
| 4 | この密度の幅は何で決まる？ → 2.3のガウス楕円を復習し、潜在方向と雑音の分散を足す → $C=WW^T+\sigma^2 I$ |
| 5 | 同じ13点にどう合わせる？ → 主軸・固有値から最尤パラメータを置く → $\mu=\bar x$、$W=u_1\sqrt{\lambda_1-\sigma^2}$、$\sigma^2=\lambda_2$ |
| 6 | 観測点から潜在位置をどう読む？ → 点の影と事後平均を比較 → 有限の雑音で平均側へ縮み、不確かさも残る |
| 7 | 大きなデータや欠測ではどう学ぶ？ → E段で潜在位置、M段で線を更新する図 → EMは共分散行列を作らず反復できる |
| 8 | 別の広がりをどう扱う？ → 等方雑音から軸別雑音へ切り替え、余分な潜在方向を弱める → 因子分析とベイズ的PCAへの接続 |
| 9 | 冒頭のA・Bはどう違う？ → 同じ13点・同じ最尤密度を再表示 → 再構成誤差0でもBの密度はAの約0.0103 |

## 視覚補足と章をまたぐ復習

- scene04冒頭に2.3の共分散・ガウス楕円を黄色い輪郭の復習カードRとして9.3秒再現する。既存節のコードはimportしない。続く主軸分散の棒グラフでは潜在変数・雑音・合計を同じ軸上で足す。
- scene06の条件付きガウスは、観測した点から線上へ戻す矢印と、事前・事後の相対密度を同時に示す。雑音分散のつまみを1.000から最尤値0.046へ動かし、事後平均と幅を切り替える。
- scene07のEMは、点の潜在位置を決めて線を動かす2段階と、実際に計算した更新角度を図で示す。同じ章の12.1は冒頭の一言でつなぎ、復習カードを作らない。復習9.3秒、事後分布の補足約17.8秒、EMの2段階の補足約17.8秒で合計約44.9秒、音声495.1秒の約9.1%。

## 原文との照合

Bishop (2006) §12.2、印刷pp.570–586（PDF pp.590–606）の全文と図12.9–12.15を確認した。中心は式(12.31)–(12.46)、(12.48)–(12.57)、(12.60)、(12.64)。[公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)では図12.9の記号、式(12.40)の$\sigma^{-2}I$、式(12.42)の事後共分散$\sigma^2 M^{-1}$、p.575の$\sqrt{\lambda_i-\sigma^2}$、式(12.53)・(12.58)、図12.12・12.14のキャプションの訂正を確認した。本編では大規模計算量の厳密比較、EMの全行列式導出、ベイズ的PCAのラプラス近似と超パラメータ更新、因子分析の全更新式、混合モデル・分類・Gibbsサンプリングは扱わない。

## 数値例と近似の条件

12.1の`pca_model.py`と同じ13点を対象節内に再現する。標本共分散は$1/N$で計算する。$D=2,M=1$とし、最尤解で$\lambda_1\simeq3.932411$、$\lambda_2\simeq0.045573$、$\mu=(1.15,0.35)$、$\sigma^2=\lambda_2$、$W=u_1\sqrt{\lambda_1-\lambda_2}$とする。主軸上のAを$\mu$、Bを$\mu+6u_1$とする。両者の直交再構成誤差は0。確率密度の比$p(B)/p(A)=\exp[-6^2/(2\lambda_1)]\simeq0.010282$で、密度そのものを確率とは呼ばない。Bは訓練点の主軸座標範囲（約−3.100〜3.030）より外にある。映像中の標本・等高線・曲線はこのモデルから計算し、生成例の乱数種は122に固定する。scene06の事後分布の比較だけは、同じ$W$と説明用の$\sigma^2=1.000$を一時的に使い、最後に最尤値へ戻す。このとき最尤モデルの事後分散は約0.011589。scene07のEMは$W$を横向き・長さ$\sqrt{\lambda_1-\lambda_2}$、$\sigma^2=0.3$から始める自作計算で、1回目の角度は約34.916度、8回目は約35.388度。欠測例は$x_1=2.5$のみ観測し、最尤モデルでの$x_2$の条件付き平均は約1.292である。scene08の因子分析・ベイズ的PCAの図は模式図で、13点をこれらのモデルに改めて当てはめた結果ではない。説明用の有限標本が厳密なガウス分布に従うと主張しない。

## ファイルと再生成

[完成動画（480p15）](media/videos/prml_12_2_probabilistic_pca/480p15/PRML122ProbabilisticPCA.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

- `pca_model.py` / `ppca_model.py` / `verify_numerics.py` / `numerical_validation.json`: 12.1と同じ13点、最尤モデル、密度比、事後分布、EM更新を再計算する。
- `narration_content.py` / `narration_script.md`: 9シーン・55拍・110文の字幕`display`と読み上げ`speech`。
- `make_voicevox_narration.py` / `assets/voicevox/`: WhiteCUL ノーマル（23）、話速1.08、抑揚1.08の9 WAVとSHA-256・文時刻付きmanifest。
- `check_narration_readings.py` / `reading_check.json` / `.md`: 全110文の`audio_query`読み。
- `scene_support.py` / `prml_12_2_probabilistic_pca.py`: PCM時刻と同期した字幕、生成モデル、楕円、分散、事後分布、EMの図。
- `verify_caption_layout.py` / `caption_validation.json`、`verify_video.py` / `video_validation.json`: 字幕と完成映像の検査。

対象ディレクトリでメイン作業ツリーの仮想環境を使う。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_12_2_probabilistic_pca.py PRML122ProbabilisticPCA
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

VOICEVOX Engineが停止中なら、`.working/voicevox-engine-0.25.2/linux-cpu-x64/run --host 127.0.0.1 --port 50021`で起動し、自分で起動したEngineは生成後に停止する。レビュー用フレームはGit管理しない。VOICEVOX:WhiteCULのクレジットは動画外の本READMEに置く。

## 検証と限界

- `py_compile`、`verify_numerics.py`、`verify_caption_layout.py`を実行した。13点の固有分解、$C=S$、A/Bの密度比、事後分散、EMの対数尤度の単調増加、欠測時の条件付き平均を検査した。110字幕の最大幅は9.341、高さは0.458 Manim単位で、安全範囲内。
- VOICEVOX Engine 0.25.2の`audio_query`を全110文に実行し、`reading_check.json` / `.md`へ保存した。「等方」「値」「0.046」「A」の誤読を`speech`で直し再生成した。辞書登録はせず、自分で起動したEngineは停止した。
- 指定の`--disable_caching --flush_cache -ql`で全編を再レンダリングした。`verify_video.py`はmanifestのWAV SHA-256、9シーン・55拍・110字幕の時刻、MP4のH.264/AAC、854×480・15fps・495.133秒、音声495.168秒、文単位の音声相関（最小0.999480、最大時刻ずれ0.0214秒）、3秒以上の無音0件、平均音量−26.5dB・最大−5.8dB、全編デコードを確認した。
- 代表284フレームを各拍の冒頭・途中・直後と動作中・直後から抽出し、18枚のシートを目視した。タイトル近くの潜在値ラベルと事後分布の雑音分散の表示を修正して再レンダリングし、変更53フレームを再確認した。残る231フレームは前回確認版とSHA-256が一致した。軸名・目盛り・字幕・式の重なり、曲線の範囲、発話中の明るさ、英語注記、枠の有無を確認した。レビュー画像はGit管理外。
- 480p版の技術検査と代表フレーム目視を実施した。全編の実時間での通し視聴・試聴、全フレーム目視、1080p版のレンダリング、視聴者による理解度評価は未実施。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL
