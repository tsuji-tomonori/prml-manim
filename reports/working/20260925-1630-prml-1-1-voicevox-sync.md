# PRML 1.1 表示修正・VOICEVOX 同期 作業完了レポート

保存先: `reports/working/20260925-1630-prml-1-1-voicevox-sync.md`

## 1. 受けた指示

同じ worktree でシーン6・8の表示を直し、曲線軸の余白を調整する。稼働中の VOICEVOX で WhiteCUL（23）の全音声を生成し、字幕と動作を読み上げに揃える。フルレンダリング、音響検査、各シーンの画像確認、3シーン以上の同期照合を行い、目的別にコミットする。push・PR 作成・コンテナの起動停止は行わない。

## 2. 要件と対応

| 要件 | 対応 | 根拠 |
|---|---|---|
| Engine の疎通を最初に確認 | 対応 | curl /version は 0.25.2、/speakers は WhiteCUL ノーマル 23 |
| シーン6の係数軸数式 | 対応 | 明示的な絶対値・括弧、完成した式を表示して強調 |
| シーン8の棒とバネ | 対応 | 塗りつぶした棒、w₆ の横の細い2巻きだけ |
| シーン8の凡例の余白 | 対応 | RMS 側へ移動、係数の横軸目盛りから分離 |
| 曲線の縦方向を有効利用 | 対応 | 通常 ±1.5、振動時は連続拡大、追加データは最低 ±1.7 |
| 全シーンの音声と字幕を同期 | 対応 | 9 WAV、67 beat、160 文。全文字列一致とハッシュを検証 |
| フルレンダリングと音響検査 | 対応 | 下記の実測値 |
| 各シーンの目視 | 対応 | 最終 MP4 から 35 枚を抽出し確認 |
| 3シーン以上の同期照合 | 対応 | シーン2・3・5・8、文の PCM 発声開始と動作・画像を照合 |
| 目的別コミット、報告 | 対応 | 表示 / 音声同期 / 動画 / 文書 |

## 3. 判断と実施内容

- Manim CE と公開クラス名・ファイル名を維持。NumPy のデータ・最小二乗・正則化・RMS 計算は変更していない。
- 係数ラベルは TeX の `\lvert` / `\rvert` と `\left` / `\right` で一つの式にした。途中の Write が欠けた数式に見えるため、完成形を表示して強調する。ほかの説明用数式も、読み上げ中に必要な項が欠けないよう調整した。
- 係数を細線から塗りつぶした長方形へ変更。代表のバネは w₆ の横へ置き、棒の高さを隠さない。ln λ = −32, −18, 0 で、この棒の高さは 5.7484, 2.7087, 0.0078。
- 正則化項と誤差項は、それぞれを説明する文に合わせて係数棒・残差と同時に強調する。
- 曲線の縦軸は通常 ±1.5。曲線が外へ出るときだけ縦軸を拡大し、点・生成関数・残差・目盛りを同時に更新する。曲線はクリップしない。係数・RMS の比較軸は固定。
- 正方形は共通の面積縮尺を使い、台本にも明記。符号付き和のパネルは「例」と明示した。
- 音声を文単位で合成し、PCM の長さを manifest に保存。旧無音版の最低尺を外し、beat 末尾のみ約0.35〜0.42秒を加える。
- 字幕は音声と同じ文を表示する。長い文は2行へ分け、行ごとの Text を VGroup で配置する。単一の Text の改行で生じた境界の異常は修正した。
- 中間画像の同期確認で、後続の強調が曲線の updater を止め、RMS 点を早く表示する問題を発見した。各段階を実行時に生成する方式へ直し、残差線には個別 updater、強調には静的コピーを使った。カウンターも一つの連続操作にした。短区間の試験描画後、フルレンダリングと35枚の確認をやり直した。
- M 操作、残差カウント、RMS 点打ちを別の時間区間にした。谷底や100点への到達は、その到達を告げる文の終わりまでに済ませ、続く説明では結果を強調する。
- 既存の原文参照と3b1b由来の連動表示を維持。原文・参照コードの調査記録は [前回レポート](20260925-1501-prml-1-1-3b1b-remake.md) を参照。

## 4. 音声とシーン構成

VOICEVOX Engine **0.25.2**、**WhiteCUL ノーマル（23）**。話速1.08、抑揚0.95、音量係数1.0。WAV は24 kHz・16 bit・mono。全9本を生成し、manifest の状態はすべて generated。旧音声の流用はない。

| Scene | 内容 | 開始 | 実測尺 | 原文参照 |
|---:|---|---:|---:|---|
| 1 | この点を生んだ曲線は？ | 0.000秒 | 51.067秒 | pp.2–5 / Fig.1.2 |
| 2 | 多項式は、つまみの集まり | 51.067秒 | 67.467秒 | p.5 / Eq.(1.1) |
| 3 | ずれを、面積に変える | 118.533秒 | 79.533秒 | pp.5–6 / Fig.1.3 / Eq.(1.2) |
| 4 | 二つのつまみで、谷底へ | 198.067秒 | 56.133秒 | p.6 / Eq.(1.2)（等高線は独自の補助図） |
| 5 | 点を通れば、予測もよくなる？ | 254.200秒 | 105.800秒 | pp.6–8 / Fig.1.4–1.5 / Eq.(1.3) |
| 6 | 曲線の裏側で、係数が膨らむ | 360.000秒 | 56.000秒 | pp.8–9 / Table 1.1 |
| 7 | 点が増えると、同じ九次式は？ | 416.000秒 | 50.133秒 | p.9 / Fig.1.6 |
| 8 | 係数を、ゼロへ引くバネ | 466.133秒 | 83.533秒 | pp.10–11 / Eq.(1.4) / Fig.1.7–1.8 / Table 1.2 |
| 9 | 一本の線の、その先へ | 549.667秒 | 49.667秒 | pp.5,11–12 / §1.2 冒頭 |

WAV の合計は **599.333秒**。

## 5. 実施した検証

| 検証 | 結果 |
|---|---|
| `python -m py_compile *.py` | 成功 |
| `python verify_numerics.py` | 5件成功。数値実験、PCM 余白、ハッシュ、字幕文一致、再開処理 |
| Manim `--progress_bar none --disable_caching --flush_cache -ql` | 全9シーン・67 beat のフルレンダリング成功（104アニメーション） |
| ffprobe 映像 | h264 / 854×480 / 15/1 fps / 599.332678秒 |
| ffprobe 音声 | aac / 599.360000秒 |
| ストリーム尺の差 | 0.027322秒 |
| `silencedetect=noise=-45dB:d=3` | silence_start なし。3秒以上の無音0件 |
| `volumedetect` | 平均 -26.6 dB、最大 -5.6 dB |
| 代表画像 | 35枚を目視。各シーン最低2枚、シーン6・8と縦軸変更箇所を追加確認 |
| `git diff --check` | 成功 |

最終化の途中で1回のレンダリングが終了コード143で中断した。ログに Python / Manim の例外はなかった。端末付きで最初から再実行し、完了した MP4 を採用した。字幕配置の初期エラーも修正後にフルレンダリングした。

検証コマンドは対象 feature ディレクトリで実施。既存 venv の Python / Manim を使用した。

```bash
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
ffprobe -v error -show_entries format=duration:stream=index,codec_type,codec_name,duration,width,height,r_frame_rate -of json <video.mp4>
ffmpeg -hide_banner -i <video.mp4> -map 0:a:0 -af silencedetect=noise=-45dB:d=3 -f null -
ffmpeg -hide_banner -i <video.mp4> -map 0:a:0 -af volumedetect -f null -
ffmpeg -y -ss <time> -i <video.mp4> -frames:v 1 <frame.png>
```

## 6. 音声と映像の同期照合

時刻は動画全体の秒数。文境界は合成した PCM の長さから取り、発声開始は WAV のサンプルが −45 dB を初めて超えた時刻で確認した。映像は同じ区間の前後・途中のフレームを目視した。画面更新の時間分解能は15 fps（約0.067秒）。

| Scene / Beat | 動作区間 | 発声開始（各文） | 照合内容 |
|---|---|---|---|
| scene02 / 1 | 53.701〜58.736 | 51.143, 53.794, 56.006 | 高さのつまみを説明する第2文から w₀ を操作 |
| scene03 / 4 | 142.067〜151.187 | 142.153, 147.576 | 縦の残差を正方形へ移す説明中に変形 |
| scene05 / 3 | 274.933〜285.216 | 275.014, 276.791, 281.349 | 第1文で M=1 へ移行、第2文でカウントと RMS 点打ち |
| scene08 / 4 | 493.600〜502.005 | 493.672, 496.163 | λ を強くする文で曲線・係数・RMS が連動 |
| scene08 / 6 | 511.867〜520.837 | 511.920, 513.927, 517.615 | 弱い λ へ戻す説明と逆方向の動作が一致 |

M=1 の内訳: M slider 274.933〜276.667秒 / count residuals 276.667〜279.667秒 / RMS stamp 279.667〜281.267秒 / explain RMS 281.267〜285.200秒 / breath 285.200〜285.600秒。

各シーンの映像尺と対応 WAV の長さ、全160文の連結文字列、全動作区間が対応 beat 内に収まることも検証した。

## 7. 成果物と保存場所

- [音声付き480p動画](<../../feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/media/videos/prml_1_1_polynomial_curve_fitting/480p15/PRML11PolynomialCurveFitting.mp4>)
- [README](<../../feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/README.md>)
- [台本](<../../feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/narration_script.md>)
- 音声9 WAV と manifest: `feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/assets/voicevox/`
- 再現用コード: `prml_1_1_polynomial_curve_fitting.py`, `make_voicevox_narration.py`, `narration_content.py`, `verify_numerics.py`。
- 検証画像・ffprobe・音響ログ・同期照合 JSON: `.working/review/prml11-voiced-final/`（Git 管理外）。
- シーン・beat・動作・字幕の実測境界: feature 内 `media/prml11_timeline.json`（Git 管理外）。
- MP4 SHA-256: `30893e94fb7f16c7efaab8514cb6a76150c8515a279a3d5ce7ae395e28def814`

## 8. 作業場所とコミット

Worktree: `/home/t-tsuji/project/prml-manim/.working/worktrees/prml-1-1-polynomial-3b1b-remake`

Branch: `codex/prml-1-1-polynomial-3b1b-remake`

```text
24bbd46 💄 style(prml-1.1): グラフの可読性を改善
67b2a46 🔊 feat(prml-1.1): WhiteCULの文音声へアニメーションを同期
783a52b 🐛 fix(prml-1.1): RMS点打ち前の曲線更新を維持
5de42f5 🔨 build(prml-1.1): 音声同期済みの480p動画を更新
```

このレポートと README は最後の文書コミットに含める。各コミットの前に `git diff --cached --name-only` を確認した。main は変更していない。push・PR 作成・Docker コンテナ操作は行っていない。partial_movie_files、Tex、images、`.working/` はコミットしていない。

## 9. 指示への適合度

| 評価軸 | 評価 | 理由 |
|---|---:|---|
| 指示網羅性 | 5/5 | 表示修正、全音声、同期、指定の検証を実施 |
| 制約遵守 | 5/5 | 指定 worktree・CE・venv・話者、push/PR/コンテナ操作なし |
| 成果物品質 | 4.5/5 | 480p の代表画像と音響・同期を確認。全編の聴感評価は未実施 |
| 説明責任 | 5/5 | 数値・再実行・確認方法と限界を記録 |
| 検収容易性 | 4.5/5 | 時刻・動画・台本・検証ログあり。画像はローカルの管理外領域 |

**総合fit: 4.8 / 5.0（約96%）**

主要な依頼を実施し、最終動画の映像・音声・代表フレームで確認した。全フレームの総当たり目視と全編の通し聴取を実施したとは扱わない。

## 10. 残課題・制約

- 依頼範囲で把握している未修正の表示・音声同期不具合はない。
- 全編の通し聴取による発音・抑揚の主観評価、全フレームの人手検査、高品質レンダリングは未実施。同期確認は文音声の時刻・PCM 発声開始・画像の照合による。
- 抽出画像・キャッシュ・詳細ログは Git 管理外。レポートに時刻と結果を残した。
