# PRML 11.4 スライスサンプリング

## 物語の設計

全編の問いは「歩幅を0.5に決めたままでは届かない点へ、どうやって適切な歩幅を自動で選ぶか」。11.2のメトロポリス法で見た歩幅の難しさと、11.3の条件付き更新を受け継ぐ。説明用の未正規化密度は中心が鋭く裾が広い1変数の山 `f(z)=exp(-z²/(2·0.45²))+0.25 exp(-z²/(2·1.8²))`。現在点 `z=0`、高さ `u=0.15` では、最初の幅0.5の区間からは `z=1.4` に届かない。視聴者に予想してもらい、横線で切った範囲を見せる。区間の拡張、棄却後の縮小を通じて、`1.4` が許される理由を追い、結末で同じ値へ戻る。青は現在点、黄は高さとスライス、緑は採用点、赤は密度曲線、紫は候補または比較に使う。

| シーン | 直前の疑問 → 実験 → 発見 |
|---|---|
| 1 | 0.5の歩幅では1.4へ届かない？ → 同じ曲線上に予想と結果を置く → 高さ次第で許される距離が変わる |
| 2 | 高さは誰が決める？ → 現在点の曲線下から一様に高さを引く → 補助変数 `u` |
| 3 | 横線の下を選ぶと目標分布になる？ → 縦の細い面積を見せる → 周辺分布が `f(z)/Z` |
| 4 | 許される横幅をどう探す？ → 幅0.5から端を外側へ動かす → この高さの幅に適応 |
| 5 | 外側の候補を引いたら？ → `2.0`を棄却し区間を縮め、`1.4`を採用 → 現在点を残す縮小 |
| 6 | 高さが変わっても同じ幅でよい？ → 同じ曲線上で高さを切り替える → 高い位置は狭く、低い位置は広い |
| 7 | 複数の山でも自由に飛べる？ → 谷より高い横線を見せる → 離れた部分への移動は自動ではない |
| 8 | 多変数なら？ → 11.3の座標ごとの断面に同じ操作を重ねる → 条件付き密度へ適用 |
| 9 | 冒頭へ戻る → `z=0,u=0.15,w=0.5,2.0棄却,1.4採用` を再現 → 高さに合わせて幅を決める |

## 視覚補足と章をまたぐ復習

- 2.1「連続確率変数の密度」を使う直前に、復習カードRで曲線下の細い面積を再現する（10〜15秒）。過去節のコードはimportせず、必要な図形を再現する。
- 積分 `∫p(z,u)du` の直前に、同じ縦帯を上下に数える短い本文カードを置く（5〜12秒）。式の結果を、縦帯の高さ `f(z)` と対応させる。
- 実測は復習11.733秒、積分補足8.733秒。共同密度の補足10.333秒も含めると30.799秒で音声合計480.467秒の約6.4%。本編の15%以内。

## 原文との照合

Bishop (2006) §11.4、印刷pp.546–548（PDF pp.566–568）、式(11.51)–(11.52)、図11.13の全文・図像を確認。公式正誤表には図11.13の曲線名2か所の訂正がある。本編は未正規化密度を `f(z)` と統一し、表示上の混同を避ける。[公式正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)。

中心は補助変数、曲線下の一様分布、周辺化、区間の拡張と縮小、座標ごとの適用。複数の山での高度な区間構成や、詳細釣り合いの一般証明はこの節の尺では扱わない。図11.13の2山の形は注意点の説明でだけ使う。

## 数値例と近似の条件

説明用の `f(z)` は未正規化密度で、正規化定数は `Z=(0.45+0.25·1.8)√(2π)=0.9√(2π)`。値 `f(0)=1.25`、`f(1.4)≈0.19266`、`f(2.0)≈0.13490`。`u=0.15` では `1.4` はスライス内、`2.0` は外。初期区間 `[-0.2,0.3]`、幅 `w=0.5` を両方向へ端がスライス外になるまで伸ばすと `[-2.2,2.3]`。説明用に候補 `2.0` と `1.4` を固定し、前者を棄却して右端を `2.0` に縮めた後、後者を採用する。この2候補はアルゴリズムの経路を示す固定例で、確率実験の代表値ではない。スライスの正側境界は、高さ0.95で0.375958、高さ0.6で0.638673、高さ0.15で1.822635、高さ0.05で3.229421。検証スクリプトで再計算する。2山の模式図は別の未正規化関数 `exp(-0.5((z+1.5)/0.55)^2)+0.7 exp(-0.5((z-1.5)/0.65)^2)` を使い、高さ0.5と0.02で連結の違いを見せる。最初の数値例とは別で、統計的な主張には使わない。標本分布の実装確認では `default_rng(114)`、最初の2000更新を除く30000標本を使い、平均と分散を解析値0、1.72125と照合する。これは視覚内の固定候補2点とは別の検証である。scene08右の2変数の楕円は座標ごとの断面を示す模式図で、左の説明用 `f(z)` と同一の同時分布から計算した等高線ではない。

## ファイルと再生成

[完成動画（480p15）](media/videos/prml_11_4_slice_sampling/480p15/PRML114SliceSampling.mp4) ／ [全文台本](narration_script.md) ／ [全文の読み確認](reading_check.md)

- `slice_model.py` / `verify_numerics.py` / `numerical_validation.json`: 未正規化密度、スライス境界、区間の拡張・縮小、1変数の標本分布を検証。
- `narration_content.py` / `narration_script.md`: 9シーン・55拍・110文の字幕 `display` と読み上げ `speech`。
- `make_voicevox_narration.py` / `assets/voicevox/`: WhiteCUL ノーマル（23）、話速1.08、抑揚1.08の9 WAVとハッシュ・文時刻付きmanifest。
- `check_narration_readings.py` / `reading_check.json` / `.md`: 全文の `audio_query` 読み。
- `scene_support.py` / `prml_11_4_slice_sampling.py`: PCM時刻と同期した字幕と映像。
- `verify_caption_layout.py` / `caption_validation.json`、`verify_video.py` / `video_validation.json`: 全字幕と完成映像の検査。

対象ディレクトリでメイン作業ツリーの仮想環境を使います。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_11_4_slice_sampling.py PRML114SliceSampling
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

VOICEVOX Engineが止まっている場合は、`.working/voicevox-engine-0.25.2/linux-cpu-x64/run --host 127.0.0.1 --port 50021`で起動し、自分で起動したものは音声生成後に停止します。辞書登録はしません。レビュー用フレームはGit管理外です。

## 検証と限界

- `py_compile`、数値検証、110文のVOICEVOX `audio_query` 読み、WAVのSHA-256と台本・字幕・時刻の整合を確認した。標本検証は平均0.00530（解析値0）、分散1.72990（解析値1.72125）。
- 110字幕の寸法を検査した。最大幅10.526、最大高さ0.619（Manim座標）、数式入り8字幕を含む。
- `--disable_caching --flush_cache -ql` で全編をレンダリングした。完成MP4は480.467秒（8分00秒）、854×480、15fps。`ffprobe`、全文の音声照合、`silencedetect`（-45dB、3秒以上0件）、`volumedetect`（平均-26.7dB、最大-6.1dB）、全編デコードが通った。110文のWAVとMP4音声の照合は最小相関0.99953、最大時刻ずれ0.0214秒。
- 場面末尾だけでなく動作中・直後を含む308枚を抽出し、20枚のコンタクトシートと重要箇所の原寸フレームを目視した。軸、目盛り、式、候補点、字幕、つまみの位置と重なりを確認した。
- 480p15以外の解像度はレンダリングしていない。全編を等速で通し視聴・聴取した確認は行っていない。目視は抽出フレームを対象とする。2山と2変数の図は模式図であり、主数値例と同じ分布を表すものではない。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL
