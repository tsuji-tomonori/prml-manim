---
name: manim-voicevox-education-video
description: Manim と VOICEVOX を使って日本語の数学・機械学習・PRML 解説動画を作成または改善するときに使う。ナレーション台本、VOICEVOX 音声生成、Manim アニメーション、LaTeX 数式、画面レイアウト、音声ギャップ、RMS/回帰などの説明アニメーション検証を含む動画制作タスクで使用する。
---

# Manim VOICEVOX Education Video

## 目的

Manim の映像と VOICEVOX の日本語ナレーションを同期させ、抽象的な数学・機械学習の説明を「画面で動かして理解できる」教育動画として作る。

このリポジトリでは、まず対象 feature ディレクトリの `*.py`、`make_voicevox_narration.py`、`narration_script.md`、`README.md` を確認してから作業する。

## 基本方針

- 数式は画面に出し、ナレーションは意味を説明する。
- 比較はできるだけ 1 つのグラフ上で切り替える。`M=0,1,9` や `lambda` の比較を単に並べるだけにしない。
- パラメータはスライダーやカウンタで動かし、曲線・点・誤差・係数が変わる様子を見せる。
- 「ズレを足すと打ち消しあう」「二乗すると打ち消されない」「訓練誤差とテスト誤差が分かれる」などは、説明より先にアニメーションだけが終わらないよう、ナレーション時間に合わせて動かす。
- 「罰する」など比喩が唐突な語は避け、正則化では「大きすぎる係数を抑える」「曲線の暴れを小さくする」のように説明する。
- 画像入力を説明するときは、例の前提を明示する。例: `28 x 28` は今回の白黒画像のピクセル数であり、カラーならチャンネル数が増える。

## ナレーション

- VOICEVOX の話者はユーザー指定を優先する。WhiteCUL のみ指定なら全編 WhiteCUL にする。
- 話速は聞き取りやすさを残しつつ、冗長に感じる場合は少し速める。既存の `make_voicevox_narration.py` の話速・ポーズ設定を調整する。
- `SCENES` の文言、音声ファイル、Manim 側の `add_sound` 対応を揃える。
- VOICEVOX 生成が中断された場合は、生成済み音声を壊さず `--from-scene <scene_id>` で再開する。
- 導入後に音声が消える問題を疑ったら、Manim キャッシュ、`add_sound` の対象ファイル、動画内音声ストリーム、長時間無音を順に確認する。
- ユーザーが最後のクレジット表示削除を求めた場合、動画内表示は消す。VOICEVOX のクレジット要件は概要欄や README など動画外で満たす。

## Manim 表現

- 日本語フォントは可読性優先で `Noto Sans CJK JP` 系を使う。画面内の日本語は `Text` ヘルパーでフォントを統一する。
- LaTeX 数式は `MathTex` / `Tex` を使い、タイトルや凡例と重ならない位置に置く。
- 凡例、タイトル、軸ラベル、式、注釈は全フレームで重なりを確認する。特に左上タイトル、グラフ右上凡例、下部字幕に注意する。
- シーン冒頭で前シーンの残りが邪魔になる場合は `self.clear()` や明示的な `self.remove(...)` を使う。
- M の説明は、1 つのグラフで `M=1` から `M=9` までスライダーを動かしながら曲線を切り替える。
- ズレの説明は、符号付き残差の和が打ち消しあうパネルと、二乗後にすべて正として足されるパネルを分けて見せる。
- 訓練誤差・テスト誤差は、左の曲線上で残差を数え、右の RMS グラフへ点を置く流れで説明する。RMS グラフの完成線は training と test を 1 本ずつにする。
- 正則化は `lambda` スライダーを複数回動かし、曲線の暴れと係数の大きさが抑えられる様子を見せる。

## 実行手順

1. 対象ディレクトリへ移動し、関連ファイルを読む。
2. 台本とアニメーションの対応を scene 単位で確認する。
3. `make_voicevox_narration.py` を更新して音声を生成する。
4. Manim ファイルを更新して、音声と映像の尺を合わせる。
5. キャッシュを無効化してレンダリングする。
6. `ffprobe` / `ffmpeg` / 代表フレームで検証する。
7. このリポジトリのルールに従い、作業完了レポートを `reports/working/` に残す。

代表コマンド:

```bash
python3 -m py_compile prml_1_1_polynomial_curve_fitting.py make_voicevox_narration.py
python3 make_voicevox_narration.py
python3 make_voicevox_narration.py --from-scene scene08
.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
```

`uv run manim` の通常キャッシュで音声ギャップが出た履歴があるため、音声や `add_sound` を変えた後は `.venv/bin/manim --disable_caching --flush_cache` を優先する。

## 検証

動画出力後、少なくとも次を確認する。

```bash
ffprobe -v error -show_entries format=duration:stream=index,codec_type,codec_name,duration -of json <video.mp4>
ffmpeg -hide_banner -i <video.mp4> -af silencedetect=noise=-45dB:d=3 -f null -
ffmpeg -hide_banner -i <video.mp4> -map 0:a:0 -af volumedetect -f null -
ffmpeg -y -ss <time> -i <video.mp4> -frames:v 1 /tmp/frame.png
```

確認観点:

- 長い `silence_start` がない。
- 動画時間と音声時間が大きくずれていない。
- 代表フレームで日本語、数式、凡例、軸ラベル、字幕が重なっていない。
- M スライダー、lambda スライダー、RMS 点打ち、残差カウントなど、ユーザーが求めたアニメーションが静止説明になっていない。
- RMS グラフは training/test が 1 本ずつで、重複した線が残っていない。
- ナレーションより先にアニメーションが終わって長く静止していない。

## 現在の PRML 1.1 パス

```text
feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/
```

主な成果物:

```text
prml_1_1_polynomial_curve_fitting.py
make_voicevox_narration.py
narration_script.md
media/videos/prml_1_1_polynomial_curve_fitting/480p15/PRML11PolynomialCurveFitting.mp4
```

## よくある失敗

- Manim キャッシュにより古い音声・古い演出が混ざる。
- `add_sound` の対象 wav が存在しない、または再生成されていない。
- 導入だけ音声があり、その後が無音になる。必ず `silencedetect` で確認する。
- 凡例やタイトルが、ズームや切り替え後に重なる。
- M や lambda の比較を並べただけにしてしまい、パラメータを動かす直感が出ない。
- 訓練誤差・テスト誤差を完成グラフだけで出し、残差から RMS を計算していることが伝わらない。
- 進捗表示やクレジットなど、ユーザーが消すよう求めた文字が最後に残る。
