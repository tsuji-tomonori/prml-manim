# PRML 9.2 ガウス混合

Manim と VOICEVOX:WhiteCUL（ノーマル、speaker 23、話速1.08、抑揚1.08）による日本語動画。全8シーン、9分14.8秒。

## 物語の設計

全編の問いは「2つの中心のうち青に近い $x=0.15$ を、なぜ赤の成分がより多く引き受けるのか」。9.1 の硬い割当てでこの点は青へ行く。ここでは同じ位置に赤と青のガウス成分を置き、混合係数と幅を加えると、初期設定で赤の負担率が 0.564、青が 0.436 になる意外な結果から始める。

密度の高さを混ぜる意味、見えない成分番号 $z$、観測後の負担率を順に明かす。16点へ広げ、Eステップでは各点の「赤いインクの割合」を計算し、Mステップではそのインクの重心・広がり・総量で平均、分散、混合係数を動かす。対数尤度の推移で交互更新を確かめ、1点へ幅が潰れる特異点を見せて最尤推定の限界を示す。結末では冒頭と同じ点 $x=0.15$、同じ初期設定の 0.564 / 0.436 に戻り、距離だけでは決まらない理由を図で説明する。次節 9.3 には「この交互更新を一般の潜在変数モデルでどう説明できるか」を残す。

## シーン

| シーン | 開始 / 尺 | 疑問 → 実験 → 発見 | 原典 |
|---|---|---|---|
| 1 青に近い点が赤に？ | 0:00.0 / 1:05.0 | 9.1 の硬い境界と確率の予想、赤 0.564 の意外な結果 | pp.428, 430–432 |
| 2 山を足す理由 | 1:05.0 / 1:09.7 | 2.3 を復習し、2成分の山の高さを混合係数で重み付け | pp.430–431、(9.7)–(9.12) |
| 3 隠れた番号と負担率 | 2:14.7 / 1:11.5 | $z$ を隠して同じ点を見直し、ベイズの分子と分母を棒で示す | pp.431–433、(9.13)、図9.4–9.5 |
| 4 16点なら何を上げる | 3:26.1 / 1:11.1 | 全点の密度と対数尤度、1点に幅が潰れる特異点 | pp.433–435、(9.14)–(9.15)、図9.7 |
| 5 Eで色の濃さを決める | 4:37.2 / 1:05.3 | 16点の負担率を計算して、境界付近に両色が残る | pp.435–437、(9.23) |
| 6 Mで山を動かす | 5:42.5 / 1:13.1 | 負担率で重み付けした平均・分散・混合係数に更新 | pp.435–437、(9.17)–(9.22)、(9.24)–(9.27) |
| 7 交互に何が起こる | 6:55.6 / 1:06.9 | E/M を繰り返して対数尤度と負担率の変化を同じ図で追う | pp.437–439、(9.28)、図9.8 |
| 8 $x=0.15$ へ戻る | 8:02.5 / 1:12.3 | 冒頭の負担率を同じ数値で再現し、距離だけでは分からない理由を確かめる | §9.2 全体 |

## 視覚補足と章をまたぐ復習

- 2.3 の「ガウス混合・負担率」を使う直前に節番号付きの復習カードを9.7秒置く。2つの山の高さを足す図形と配色を必要な部分だけ再現し、過去節は import しない。
- 条件付き確率の分子を 2 本の棒にし、合計を 1 にそろえる視覚補足を17.4秒置く。対数尤度の前には、2点の密度の積が対数で和になる8.3秒の補足を置く。合計35.3秒は全編の6.4%。
- 背景は既存節と同じ濃紺。入力=青 `#58C4DD`、注目操作=黄 `#FFFF00`、結果=緑 `#83C167`、比較=紫 `#9A72AC`、誤差=赤 `#FC6255` を基準にする。成分の赤・青は 9.1 の色に合わせる。グラフ背面に枠・台・影を置かない。

## 原文との照合

用語は[日本語版 PRML の公開和文索引](https://ibisforest.org/index.php?openfile=prml-jp-jp_index.pdf&plugin=attach&refer=PRML)に合わせ、responsibility を「負担率」とする。2.3 の動画ともそろえる。

Bishop (2006), *Pattern Recognition and Machine Learning* §9.2–9.2.2、印刷 pp.430–439（PDF pp.450–459）全文、および図9.4–9.8を確認した。[公式正誤表（2011-09-21）](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf)では p.434 の (9.15) の分母は一般の $D$ 次元で $\sigma_j^D$、p.435 の (9.16) は共分散の逆行列を掛け、余分な負号を除く訂正がある。本編は1次元の例を使い、(9.16) の誤植を表示しない。原典の成分番号の置換による識別不能性、行列の導出、画像例への拡張は、中心の問いに必要な範囲に絞る。図は原典を複製せず、自作の数値例で再描画する。図9.4–9.8はPDF画像で確認した。

## 数値例と近似の条件

- 説明用の1次元16点は `[-2.20, -1.80, -1.60, -1.30, -1.00, -0.80, -0.55, -0.25, 0.15, 0.40, 0.75, 1.05, 1.30, 1.60, 1.90, 2.30]`。標本は固定した説明用の数で、原典の Old Faithful データではない。対数尤度はこれらを独立な観測として計算する。
- 冒頭の2成分は $\mu=(-1.1,1.1)$、標準偏差 $\sigma=(0.9,1.2)$、混合係数 $\pi=(0.65,0.35)$。$x=0.15$ は青の平均まで0.95、赤の平均まで1.25。それでもガウス密度に混合係数を掛けて正規化すると、赤の負担率は約0.5636、青は約0.4364になる。
- 計算は1次元の正の標準偏差を持つ2成分ガウス混合の最尤 EM。分散更新では新しい平均を使う。21状態（初期値と20回の更新）の履歴と独立検証は `numerical_validation.json` に記録した。特異点は同じ16点への別のパラメータ操作で示す説明用の極限であり、通常の EM 実行履歴とは区別する。広い成分は平均0、標準偏差1.55、混合係数0.9に固定し、細い成分は平均0.15、混合係数0.1のまま標準偏差を0.03、0.005、0.001へ縮める。幅0.005では $x=0.15$ の密度は8.209。

## ファイルと再生成

| 種類 | ファイル |
|---|---|
| 数値 | [固定16点とEM](mixture_model.py)、[独立検証](verify_numerics.py)、[計算結果](numerical_validation.json) |
| 台本 | [表示と読みの定義](narration_content.py)、[実測時刻付き全文](narration_script.md) |
| 音声 | [VOICEVOX生成](make_voicevox_narration.py)、[8本のWAVとmanifest](assets/voicevox/)、[全文の読み確認](reading_check.md) |
| 映像 | [Manim本体](prml_9_2_gaussian_mixtures.py)、[字幕とPCM同期](scene_support.py)、[480p15動画](media/videos/prml_9_2_gaussian_mixtures/480p15/PRML92GaussianMixtures.mp4) |
| 検証 | [字幕寸法](verify_caption_layout.py)、[動画検証](verify_video.py)、[結果](video_validation.json) |

このディレクトリで実行する。worktree に仮想環境がなければメインの `/home/t-tsuji/project/prml-manim/.venv` を使う。音声生成時には VOICEVOX Engine 0.25.2 を `localhost:50021` で起動する。自分で起動した Engine は作業後に停止する。辞書登録はしない。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python export_narration_script.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_caption_layout.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_9_2_gaussian_mixtures.py PRML92GaussianMixtures
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_video.py
```

## 検証と限界

- `py_compile` は対象ディレクトリの全 Python ファイルで成功。`verify_numerics.py` は初期値から20回の更新まで21状態を独立に照合し、初期の赤い負担率0.563550と対数尤度の -27.737480 → -26.140535 を確認した。
- VOICEVOX Engine 0.25.2 の `audio_query` で全128文の読みを取得して確認した。誤読の修正前後は `reading_check.md` に残した。8本の WAV は manifest の台本・音声ハッシュと一致する。
- `verify_caption_layout.py` は全128字幕を確認し、最大幅9.395（安全域12.9）、最大高さ0.735（安全域0.9）だった。
- 全編を `manim --progress_bar none --disable_caching --flush_cache -ql` でレンダリングした。`ffprobe` では H.264 / AAC、854×480、15 fps、映像554.800秒、音声554.837秒。`verify_video.py` は全128文の音声とWAVの相関を確認し、最小相関0.999352、最大時刻差0.0214秒だった。`silencedetect` の3秒以上の無音は0件、`volumedetect` は平均 -26.3 dB、最大 -5.6 dB。全編デコードも成功した。
- 264枚の代表フレームを抽出し、93枚を目視した（64 beatの中央、7箇所の強調・切替の開始/途中/直後、8シーンの終端）。式、字幕、棒の値、軸・目盛り、曲線の範囲、注記の重なり、枠なしの構図、英語注記の有無を確認した。字幕128文と画面注記83件の文字列検索では、個数・番号の漢数字表記および英語の説明文は0件だった。レビュー画像はGit管理外の `media/review/` に置く。
- 内容は固定した1次元の説明用データに限る。実データへの当てはまりや全フレームの人手検査は対象外で、目視確認は上記の代表93枚による。

## VOICEVOX クレジット

VOICEVOX:WhiteCUL
