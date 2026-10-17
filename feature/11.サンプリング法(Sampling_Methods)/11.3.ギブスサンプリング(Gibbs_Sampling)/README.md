# PRML 11.3 ギブスサンプリング

## 物語の設計

全編の問いは「採用率が100%なら、細長い雲をすぐ横断できるか」。11.2末尾の相関した2変数の雲を引き継ぎ、説明用の標準2変量ガウス分布（相関 $\rho=0.98$）を使う。左下 $(-2,-2)$ から交互に座標を引き直すと、種43の20巡後も $(-2.223,-2.091)$。まず「棄却がなければ速い」と予想してもらい、この意外な結果を見せる。水平・垂直の切片に沿った条件付き分布、順次更新、分布が保たれる理由、細長い方向の進みの遅さへと原因をたどる。過緩和とブロック化で改善の方向を見せ、最後に同じ雲、出発点、20巡の結果へ戻る。

9シーン・54拍・108文、音声合計526.200秒（8分46.200秒）。数式は常に同じ楕円・座標更新・断面図と対応させる。背面の装飾枠は置かない。青は現在点、黄は更新中の座標、緑は新しい点、赤は目標密度、紫は比較・過緩和とする。

| シーン | 直前の疑問 → 実験 → 発見 |
|---|---|
| 1 | 11.2の細長い雲。全採用なら速い？ → 20巡を予想して経路を見る → 左下に残る |
| 2 | どこから次点を引く？ → 片方を固定し断面の確率を示す → 条件付き分布 |
| 3 | 新しい値はいつ使う？ → 水平・垂直を交互に更新 → 直前の値を即座に使う |
| 4 | なぜ目標分布を保てる？ → 周辺と条件付きの積を図に戻す → 各更新で不変、採用率1 |
| 5 | 全採用でもなぜ遅い？ → 楕円長軸と断面の幅、自己相関を重ねる → 短い歩幅の往復 |
| 6 | 反対側へ寄せたら？ → ガウス条件付きの過緩和を動かす → 分布を保って戻りを減らす選択肢 |
| 7 | 多変数では全変数を見る？ → 8.3のマルコフブランケットを復習し近傍を点灯 → 条件付きの計算を局所化 |
| 8 | 1変数ずつの限界は？ → 同じ楕円で2変数を一組として更新 → ブロック化には同時条件付きから引く手段が必要 |
| 9 | 20巡の結果へ戻る → 同じ軌跡と断面幅を重ねる → 採用率は正しさの一部で、探索速度は相関に依存 |

## 視覚補足と章をまたぐ復習

- 2.3「多変量ガウス分布」の復習カードR（実測11.667秒）: 楕円の向きが共分散で決まる図を再現し、今回の相関0.98へつなぐ。
- 条件付き確率の短い本文カード（実測8.800秒）: 縦の位置を固定し、その高さの横断面を正規化して水平の候補分布にする。条件付き密度の意味が後の更新の前提となるため採用する。
- 8.3「マルコフブランケット」の復習カードR（実測11.867秒）: 近傍ノードだけを着色し、ギブス更新の条件付き分布へ対応させる。
- 合計32.333秒で音声合計526.200秒の約6.1%。総尺の15%以内。過去節の動画コードはimportせず図形を再現する。

## 原文との照合

Bishop (2006) §11.3、印刷pp.542–546（PDF pp.562–566）、式(11.46)–(11.50)、図11.11–11.12。全文と図を確認した。公式正誤表ではp.544の余計な `{`、式(11.50)の末尾の $\alpha_i^2$ を $\alpha^2$ に訂正する。式と表示には訂正版を使う。[公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)。

中心は条件付きの順次更新、不変性、到達可能性、相関が強い場合の遅さ、過緩和、ブロック化。原典のGibbsの伝記、適応的棄却サンプリングの詳しい設計、指数型分布族の共役性、ICMとの詳細比較は本編から省き、関係は必要最小限の注記に留める。

## 数値例と近似の条件

説明用の同一例は平均 $(0,0)$、各成分の分散1、共分散 $0.98$ の正定値ガウス分布。条件付き分散は $1-\rho^2=0.0396$、標準偏差は約0.199。乱数はNumPy `default_rng(43)`。各巡で $z_1$、次に更新済みの $z_1$ を使って $z_2$ を引く。20巡後の数値は固定シード1本の例であり、収束速度の一般的な推定値ではない。長軸と条件付きの幅の比に関する $O((L/l)^2)$ は原典の定性的な尺度。定常時の $z_2$ の1巡遅れ相関はこの2変数モデルで厳密に $\rho^2=0.9604$。過緩和はガウス条件付きと $-1<\alpha<1$ に限定し、正誤表どおり $\sqrt{1-\alpha^2}$ を用いる。

## ファイルと再生成

[完成動画（480p15）](media/videos/prml_11_3_gibbs_sampling/480p15/PRML113GibbsSampling.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

- `gibbs_model.py` / `verify_numerics.py` / `numerical_validation.json`: 条件付き分布、固定乱数の経路、周辺・条件付きの積、過緩和の分散を検証。
- `narration_content.py` / `narration_script.md`: 9シーン・54拍・108文の字幕`display`と読み上げ`speech`。
- `make_voicevox_narration.py` / `assets/voicevox/`: WhiteCUL ノーマル（23）、話速1.08、抑揚1.08の9 WAVとハッシュ・文時刻付きmanifest。
- `check_narration_readings.py` / `reading_check.json` / `.md`: 全文の `audio_query` 読み。
- `scene_support.py` / `prml_11_3_gibbs_sampling.py`: PCM時刻と同期した字幕と映像。
- `verify_caption_layout.py` / `caption_validation.json`、`verify_video.py` / `video_validation.json`: 全字幕と完成映像の検査。

対象ディレクトリでメイン作業ツリーの仮想環境を使います。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_11_3_gibbs_sampling.py PRML113GibbsSampling
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

VOICEVOX Engineが止まっている場合は、`.working/voicevox-engine-0.25.2/linux-cpu-x64/run --host 127.0.0.1 --port 50021`で起動し、自分で起動したものは音声生成後に停止します。辞書登録はしません。レビュー用フレームはGit管理外です。

## 検証と限界

- `py_compile`、`verify_numerics.py`、`verify_caption_layout.py`、`verify_video.py` が通過。字幕108文のうち数式を含む12文を含め、全字幕の幅・高さを検査した。
- 全編をキャッシュ無効・flush付きのManimでレンダリング。`ffprobe`で854×480、15 fps、映像526.199678秒、音声526.229333秒を確認。9本のWAVとmanifestのハッシュ・文時刻を照合し、完成映像の108文の音声とWAVの相関は最小0.999606、ずれは最大0.0214秒。
- `silencedetect`（-45 dB、3秒以上）で長い無音0件、`volumedetect`で平均-26.7 dB・最大-5.7 dB。全編のデコードが成功した。
- 381枚の代表フレームを抽出。全24枚のコンタクトシートで初回の画面を確認し、見つけたラベルの重なり・つまみ値の崩れ・不要な補助線を修正して再レンダリングした。最終版は修正箇所の原寸フレームと該当コンタクトシートを再確認した。
- 未検証: 1080pのレンダリング、全編を通した実時間の視聴・聴取。目視は代表フレームに基づくため、その間の瞬間的な表示問題は完全には排除できない。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL
