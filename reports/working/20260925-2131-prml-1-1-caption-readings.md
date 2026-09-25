# 作業完了レポート：PRML 1.1 字幕記号と読み上げの修正

保存先: `reports/working/20260925-2131-prml-1-1-caption-readings.md`

## 1. 受けた指示と判断

- 指定 worktree・ブランチで継続し、Engine の疎通を最初に確認する。
- 文単位で字幕 `display` と音声 `speech` を分離する。字幕は漢字と数学記号を用い、安全領域の検査を維持する。
- 全文を audio_query で確認し、修正前後の読みを残す。変更した音声を再生成し、映像・字幕を実測尺へ同期する。
- フルレンダリング、構文・数値テスト、映像・音声ストリーム、無音、記号字幕の目視を検証する。
- 目的別に日本語 gitmoji コミットを作り、push・PR は行わない。

各文に明示的な display/speech の組を持たせた。字幕は Noto Sans CJK JP の Text と MathTex を混ぜ、数式の途中では改行しない。日本語の読点・下付き・累乗を自然な位置に置くため、非表示の参照文字から文字の下端を合わせる。辞書への登録はせず、誤読箇所を speech の読み仮名で固定した。

## 2. 対応状況

| 要件 | 結果 |
|---|---|
| Engine 疎通 | curl /version → 0.25.2。コンテナ操作なし |
| display/speech 分離 | 9シーン・67 beat・160文。manifest v4 に両文とID、timeline には display とIDを記録 |
| 数式字幕 | 18文を MathTex 化。x、t、w₀、w₁、x²、x³、M、M+1、E(w)、Σ、1/2、RMS、λなど |
| 全文読み確認 | 修正前160文、修正後160文の moras を目視。最終 display 更新後も全件再取得し、読みが前の確認結果と一致することを cmp で確認 |
| 音声再生成 | 変更10文を再合成し、9 WAV を再構成。6シーンの WAV が変更、3シーンは同じ文キャッシュで同一内容 |
| 同期 | 文 PCM 尺で字幕切替、各 beat の動作開始・終了を再計算。3シーンの文境界・発声開始・代表画面を照合 |
| 検証・提出 | 以下の実測結果、最終MP4、読み一覧、新規レポート、目的別コミット |

## 3. 誤読・読み方の修正

| 原文 | API の修正前の読み | 修正後の読み | 件数・扱い |
|---|---|---|---|
| 値 | ネ | アタイ | 6文。数値の「チ」は正しく、そのまま維持 |
| 負 | マケ | フ | 2文。負の係数・負の項 |
| 係数つまみ | カカリスウツマミ | ケイスウノツマミ | 1文。字幕も「係数のつまみ」に整える |
| 黄色 | オオショク | キイロ | 1文。読み方を文脈に合わせて統一 |
| 上向き | ウワムキ | ウエムキ | 方角の説明として統一。「負」と同じ文 |

変更対象は計10文。過学習（カガクシュウ）、正則化（セエソクカ）、残差（ザンサ）、二乗（ニジョオ）、訓練（クンレン）、汎化（ハンカ）、多項式（タコオシキ）、最小二乗（サイショオニジョオ）、係数、誤差関数、および数学記号・数・負号を全文で照合した。修正後の読み一覧に、追加で修正すべき語の読みは見つからなかった。

長母音の「ケエ」「セエ」等は Engine の moras 表記であり、そのまま記録した。修正前の原文、修正後の speech、両方の読み、display、アクセント付き kana は [reading_check.md](../../feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/reading_check.md) と同名 JSON に保存する。API の読み確認と、合成音声を耳で聴く確認は別である。

## 4. 表示とデータの流れ

| 字幕 display の例 | 音声 speech の例 |
|---|---|
| `$w_0$` は高さのつまみです。 | ダブリューゼロは高さのつまみです。 |
| 係数のつまみは、`$M+1$` 本あります。 | 係数のつまみは、エムプラス一本あります。 |
| 誤差関数 `$E(w)$` | 誤差関数イー |
| バネの強さが `$\lambda$` です。 | バネの強さがラムダです。 |
| 観測した値です。 | 観測したあたいです。 |

`narration_content.py` → speech で文 WAV 合成 → manifest の display/speech/時刻 → Manim の display 描画 → display のみの timeline、という流れに変更。台本ハッシュと WAV ハッシュ、各文のID・両文の一致を検査し、古い字幕や音声の採用を防ぐ。WAV がない場合も display と推定尺で描画できる。

最初のフルレンダリングの画像確認で、単独の読点が数式の中央の高さに出る問題を発見した。参照文字による配置へ修正し、短い静止プレビューを目視後、全編をもう一度レンダリングした。以下は再描画後の最終MP4の検査結果。

## 5. 実施した検証

| 検証 | 結果 |
|---|---|
| py_compile | 対象ディレクトリの全 Python ファイル、成功 |
| verify_numerics.py | 5テスト成功。最小二乗・正則化・追加データ・曲線範囲・音声管理。字幕両文の不一致検出も確認 |
| 字幕の読み方カタカナ検索 | display 抽出160文、描画コード、最終 timeline に rg。エックス／エム／ラムダ／ダブリュー／エヌ／イー／ティー／アールエムエス／サイン／シグマが0件 |
| 字幕安全領域 | 全160文成功。最大幅9.395（上限12.9）、最大高さ0.702（上限0.9）、単位は Manim 座標 |
| 全編レンダリング | Manim CE 0.20.1、`--progress_bar none --disable_caching --flush_cache -ql`、終了0、104アニメーション |
| ffprobe 映像 | H.264 / 854×480 / 15 fps / 600.066344秒 |
| ffprobe 音声 | AAC / 600.106667秒。映像との差 0.040323秒（1フレーム未満） |
| silencedetect | `noise=-45dB:d=3`、3秒以上の無音 0件 |
| volumedetect | 平均 -26.6 dB、ピーク -5.7 dB |
| 最終画像の目視 | 42枚。各シーン最低2枚、記号入り字幕18枚、同期比較6枚を含む。字幕の表記・句読点位置・下付き・上付き・重なり・はみ出しを確認 |
| manifest/timeline | 全160文のID・display・文時刻が対応。9 WAV のハッシュ整合。各シーンの WAV/映像進行時間差の最大 3.98e-11秒 |

音声は WhiteCUL ノーマル（speaker=23）、話速1.08、抑揚0.95、音量係数1.0。9シーンの WAV 合計 600.067秒。各 beat の末尾のみ約0.35〜0.42秒の余白を置く。

### 同期の照合

| シーン・動作 | 文に対応する動作範囲（動画内秒） | 発声開始と画像の照合 |
|---|---|---|
| 2 / w₀ スライダー | 54.301〜59.336 | 「w₀は…」の PCM 発声54.394秒。55.057秒では初期位置、58.329秒では線とつまみが上がる |
| 3 / 残差→正方形 | 142.867〜151.987 | 最初の発声142.953秒。144.235秒は移動途中、150.163秒は「面積」の文中で正方形が並ぶ |
| 8 / λ を強める | 493.933〜502.339 | 最初の発声494.005秒。495.194秒の ln λ≈−31.7 から500.658秒の約−18.6へ、曲線・係数・RMSが同時に変わる |

発声開始は WAV 内で振幅が −45 dB を初めて超えるサンプルから確認した。文内の音素を強制アラインメントした検証や、全編の通し聴取は実施していない。

再現コマンド（対象 feature ディレクトリで実行）:

```bash
curl --fail http://127.0.0.1:50021/version
/home/t-tsuji/project/prml-manim/.venv/bin/python check_narration_readings.py
/home/t-tsuji/project/prml-manim/.venv/bin/python make_voicevox_narration.py
/home/t-tsuji/project/prml-manim/.venv/bin/python -m py_compile *.py
/home/t-tsuji/project/prml-manim/.venv/bin/python verify_numerics.py
/home/t-tsuji/project/prml-manim/.venv/bin/manim --progress_bar none --disable_caching --flush_cache -ql prml_1_1_polynomial_curve_fitting.py PRML11PolynomialCurveFitting
```

ffprobe JSON、無音・音量ログ、抽出時刻一覧、同期照合、42 PNG は worktree 内 `.working/review/prml11-reading-final/` に保存（Git 管理外）。最終 timeline は対象の `media/prml11_timeline.json`（Git 管理外）。全文の読み一覧は Git 管理する。

## 6. 成果物・コミット

- [字幕・音声の台本](../../feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/narration_script.md)
- [全文の読み確認一覧](../../feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/reading_check.md)
- [最終動画](../../feature/1.序論(Introduction)/1.1.例:_多項式曲線フィッティング(Example:_Polynomial_Curve_Fitting)/media/videos/prml_1_1_polynomial_curve_fitting/480p15/PRML11PolynomialCurveFitting.mp4)
- manifest と9 WAV、字幕・音声の各スクリプト、README、今回のレポート。
- MP4 SHA-256: `7f170eb7dddada81d7c5fdc70d57086e200996f06a820dacafc05a83b48c813a`
- worktree: `/home/t-tsuji/project/prml-manim/.working/worktrees/prml-1-1-polynomial-3b1b-remake`
- branch: `codex/prml-1-1-polynomial-3b1b-remake`
- main は変更せず、push・PR は未実施。

本レポートの記録前のコミット:

```text
6c20db5 🔨 build(prml-1.1): 記号字幕付きの動画を更新
c03b4e4 🐛 fix(prml-1.1): WhiteCULの誤読を修正
5f1d986 🐛 fix(prml-1.1): 字幕を数式記号で表示
```

README と本レポートは別の文書コミットにする（自身のハッシュは最終回答に記載）。

## 7. 指示への fit と残課題

| 評価軸 | 評価 | 根拠 |
|---|---:|---|
| 指示網羅性 | 5/5 | 分離、全文の読み確認、修正、生成、同期、指定検証を実施 |
| 制約遵守 | 5/5 | 指定 worktree・Engine・話者・venvを使用。コンテナ操作、push、PRなし |
| 成果物品質 | 4/5 | 実画面の配置修正を反映。通し聴取・全フレームの目視は未実施 |
| 説明責任 | 5/5 | 読みの修正前後と検証数値を保存し、聴取との差を明記 |
| 検収容易性 | 5/5 | 文ID、台本、時刻、読み一覧、最終動画、目的別コミットを用意 |

**総合fit: 4.8 / 5.0（約96%）**

依頼された API 読みの全文目視、字幕の記号化、映像・音声検証は完了。現時点で確認済みの未修正誤読・表示不具合はない。イントネーションの自然さまで全編を耳で検証したものではなく、480p 代表画像以外の全フレーム目視も未実施。
