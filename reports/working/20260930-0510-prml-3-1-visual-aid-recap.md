# PRML 3.1 視覚補足・復習の作業完了レポート

保存先: `reports/working/20260930-0510-prml-3-1-visual-aid-recap.md`

## 1. 受けた指示と対応

3.1の動画を見直し、中学数学以上の操作の視覚補足と章をまたぐ復習を簡潔に追加する。計画・指定skill・既存ファイル・参照節・原文を読み、台本、読み、音声、アニメーション、480p動画を同期させる。節目ごとにcommit/pushし、PRを作成・マージ後に後片付けする。

| 要件 | 対応 |
|---|---|
| 固定のファイル名・クラス名 | 維持 |
| 計画3.1の補足2件・復習2件 | 全4件採用、下表 |
| 5–20秒の補足、10–30秒の復習、増加15%以内 | すべて適合、+7.366% |
| 字幕と音声に参照節番号 | 復習2件と既存の正則化説明2件に追加 |
| 共通配色・元図の色 | Rは原図の青・赤・黄、V08b/V04bは計画の色定数 |
| 新規・変更文の読みと同期 | audio_query・PCM・タイムライン・抽出画像で確認 |
| 検証 | 下表。未実施の聴取等は明記 |
| Git運用 | 指定worktree・branch、目的別commit/push。PR番号・マージ・後片付けの実績は最終回答で報告 |

## 2. 調査・判断

- `AGENTS.md`、`skills/manim-voicevox-education-video/SKILL.md`、`skills/post-task-fit-report/SKILL.md`、`skills/japanese-git-commit-gitmoji/SKILL.md`、教材レビューskillを適用。
- 見直し計画の判断基準、共通図法、概念索引、3.1全項目を確認。main開始点は `8289748`。
- 対象の描画・台本・音声生成・字幕ヘルパー・既存検証・読み記録を確認。既存9シーンの代表フレームを確認。
- 1.1 `squares()` / `regularization()`、2.3 `shape()` と対応台本・動画フレームから色・図を確認。2.5の `e1d11a5` / `9764b2e` のカードと発話同期の差分を参照。
- ローカルBishop原文の印刷pp.138–147（PDF158–167）を確認。基底拡張は(3.1)–(3.3)、ノイズは(3.7)–(3.12)、行列は(3.14)–(3.20)、一点の勾配は(3.22)–(3.23)、正則化は(3.24)–(3.30)と整合。

## 3. 入れた補足・復習

| 位置 | 内容 | 実測尺 | 共通の見せ方 |
|---|---|---:|---|
| scene01 / 16.667秒 | 1.1 最小二乗の復習 | 10.200秒 | R：青い点・赤い予測・黄色の残差平方→総和、材料をφへ |
| scene03 / 141.933秒 | 2.3 ガウス分布の復習 | 11.733秒 | R：赤いベル・黄色の中心、μ→y、σ²→β⁻¹ |
| scene04 / 250.733秒 | 行列積の補足 | 13.533秒 | V08b：青い行・紫の重み、黄色の符号付き積→緑の予測 |
| scene06 / 424.200秒 | 勾配の補足 | 8.200秒 | V04b：青い等高線、黄色の勾配、紫の逆向きの一歩 |

- 挿入アンカーは順に `scene01-03-01` の前、`scene03-01-01` の前、`scene04-03-02` の後、`scene06-06-02` の後。既存IDは維持。
- scene07/08では既存の重み・円・ひし形の説明を使い、「復習: 1.1 正則化」を表示。発話でも一章一節を示す。独立カードなし。

## 4. 計画からの調整と理由

- ガウスの復習と勾配の補足を本文カードに統一。480pで対応式と矢印を読みやすくし、本筋のつまみ・図との重複を避けた。
- 勾配の等高線を平行線にした。一点の線形予測の二乗誤差は階数1であり、円形の地形にすると説明対象が変わるため。有限差分で向きと減少量を検証。
- 行列積は2行目でも式と囲みを切り替え、「次の行も同じ重みで計算」と短く発話。移動だけで計算対象が不明になることを避けた。
- 復習はEngine実測が当初9秒程度だったため、元図の色・中心と幅の意味を明示して10秒以上へ調整。無音の水増しはしていない。
- 計画の+48秒に対し、実増加は+45.333秒。参照ラベル自体の追加は0秒だが、節番号を音声にも含める既存2文の変更で+1.667秒となった。
- 計画の採用箇所の省略なし。scene02の材料操作、scene05の射影、scene09の複数出力には、既存の図で追えるため独立補足を足していない。

## 5. 実施作業と成果物

対象: `feature/3.回帰のための線形モデル(Linear_Models_for_Regression)/3.1.線形基底関数モデル(Linear_Basis_Function_Models)`。

| 成果物 | 内容 |
|---|---|
| `narration_content.py` / `narration_script.md` | 67 beat、142文。新規9文・既存変更2文 |
| `reading_check.md/json` | speaker23のAPI読み、初回と修正後の記録 |
| `make_voicevox_narration.py` | 未変更文のPCMを既存WAVから抽出して保持、有効なシーンの再合成を省略 |
| `assets/voicevox/` | 変更6シーンのWAVとmanifest |
| `prml_3_1_linear_basis_function_models.py` | 4カード、参照表示、文・句に同期する演出 |
| `verify_numerics.py` / `review_video.py` / `validate_video.py` | 補足の数値、前後フレーム、尺・発話時刻の検証 |
| `numerical_results.json` / `validation_results.json` | 実測値 |
| `media/videos/prml_3_1_linear_basis_function_models/480p15/PRML31LinearBasisFunctionModels.mp4` | 480p配布動画 |
| `README.md` / 本レポート | 内容・条件・再生成・検証記録 |

読みで新たに発見した誤読は「一行」→イッコオ。speechを「いちぎょう」にし、イチギョウを確認した。追加の「次の行」はspeechで「ぎょう」と指定。章節・予測値・二乗和・逆数等も新規・変更11文の読みで確認した。字幕に読み仮名を入れていない。

## 6. 検証結果

| 検証 | 実施結果 |
|---|---|
| Python / 数値 | `py_compile`、`verify_numerics.py` 成功、8群 |
| 最小二乗 | 正規方程式残差ノルム 8.02546e-14、ノイズ分散 0.008904525602 |
| 補足の数値 | 予測 `[1,2]`、有限差分勾配 `[1.5,1.5]`、一点の誤差 `1.125 → 0.405` |
| フルレンダリング | キャッシュ無効・flush付き `-ql`、89 animations、終了コード0 |
| 映像 | H.264、854×480、15fps、615.399678秒 → 660.733011秒 |
| 増加 | +45.333333秒、+7.366486% |
| 音声 | AAC、660.757333秒、映像との差 0.024322秒 |
| 無音 / 音量 | `silencedetect=noise=-45dB:d=3`：0件、平均 -26.5 dB、最大 -6.0 dB |
| 読み | 全142文のaudio_query取得、新規9文・変更2文を確認。「一行」イッコオ→イチギョウ |
| 整合性 | 台本・manifest・読み記録の142文のID/display/speechが一致 |
| 字幕配置 | 全142文生成、最大幅 9.349219 / 許容12.9、最大高 0.704688 / 許容0.9 |
| 読み仮名検索 | displayのみをrg検索し0件 |
| 既存音声 | 未変更131文のPCMが元とバイト単位で一致。scene02/05/09のWAV保持 |
| 画像 | 最終MP4から139枚抽出し54枚目視。全追加場面の前後・全操作、参照表示2か所、既存9シーンを含む |
| シーンとWAV | 尺の差は最大 5.92e-11秒、142字幕の開始終了をPCMと照合 |
| 差分 | `git diff --check` 成功 |
| 発話と動作：R1.1 sum squares | 発話 21.377667秒、動作 21.400000秒、差 +0.022333秒 |
| 発話と動作：V08b multiply negative term | 発話 257.039333秒、動作 257.066667秒、差 +0.027333秒 |

動画SHA-256: `ced75e5d47472b3fce330e031acdf1577b3911f677a75d265431e9f95dc2b123`。

目視したフレームキー（再生成は `review_video.py`）：

`added-scene01-before`, `added-scene01-after`, `added-scene01-entry`, `added-scene01-exit`, `added-scene01-action00`, `added-scene01-action01`, `added-scene01-action02`, `added-scene01-action02-0.15`, `added-scene01-action02-0.85`, `added-scene01-action03`, `added-scene03-before`, `added-scene03-after`, `added-scene03-entry`, `added-scene03-exit`, `added-scene03-action00`, `added-scene03-action01`, `added-scene03-action02`, `added-scene03-action03`, `added-scene03-action04`, `added-scene04-before`, `added-scene04-after`, `added-scene04-entry`, `added-scene04-exit`, `added-scene04-action00`, `added-scene04-action01`, `added-scene04-action02`, `added-scene04-action02-0.15`, `added-scene04-action02-0.85`, `added-scene04-action03`, `added-scene04-action04`, `added-scene06-before`, `added-scene06-after`, `added-scene06-entry`, `added-scene06-exit`, `added-scene06-action00`, `added-scene06-action01`, `added-scene06-action01-0.15`, `added-scene06-action01-0.85`, `added-scene06-action02`, `reference-scene07-before`, `reference-scene07-during`, `reference-scene07-after`, `reference-scene08-before`, `reference-scene08-during`, `reference-scene08-after`, `scene01-beat05`, `scene02-beat05`, `scene03-beat05`, `scene04-beat05`, `scene05-beat05`, `scene06-beat05`, `scene07-beat05`, `scene08-beat05`, `scene09-beat05`

検証コマンドはREADME記載。追加場面4件の前後、各操作、数式字幕と参照表示を画像で確認。発話の句境界はaudio_queryの音素長と話速1.08から算出し、アニメーション記録と2か所で1フレーム以内を確認した。これは実音声の通し聴取を意味しない。

## 7. 指示へのfit評価

| 評価軸 | 評価 | 根拠 |
|---|---:|---|
| 指示網羅性 | 5/5 | 4件・参照・台本・音声・動画・検証を実施 |
| 制約遵守 | 5/5 | 名前、配色、尺、VOICEVOX話者、生成物管理を維持 |
| 成果物品質 | 4.5/5 | 数値・表示・同期を検証。通し聴取と全フレーム目視は未実施 |
| 説明責任 | 5/5 | 原文対応、計画変更、実測と未検証範囲を明記 |
| 検収容易性 | 5/5 | 位置・秒数・動画・検証結果を保存 |

**総合fit: 4.9 / 5.0（約98%）**

理由: 明示された補足・復習と検証を満たした。知覚品質の確認範囲には下記の限界がある。

## 8. 未対応・制約・リスク

- 既知の未解決不具合なし。
- 全編の通し聴取、全フレームの目視、高解像度版は未実施。アクセントの自然さを全編保証するものではない。
- Git管理領域は読み取り専用だったため、許可されたGit操作を権限付きで実施。VOICEVOX Engine 0.25.2への接続成功、コンテナ操作なし。
- PR作成・マージと後片付けは、このレポートをcommit/push後に実行し、確定した番号・ハッシュ・削除結果を最終回答で報告する。

## 9. 節目のコミット

```text
478335c 📝 docs(prml-3.1): 視覚補足の台本と読みを整える
d9e5b61 🔊 feat(prml-3.1): 視覚補足に同期する音声を生成
fa81397 ✨ feat(prml-3.1): 共通図法の補足カードを実装
bf2e702 🩹 fix(prml-3.1): 復習の説明を所定の尺に整える
cacbec0 🐛 fix(prml-3.1): 補足の動作を発話位置に合わせる
2c5803f 🩹 fix(prml-3.1): 次の行への計算を音声で案内する
017bd55 ✅ test(prml-3.1): 補足の動作前後を抽出対象にする
3ddbdba ✅ test(prml-3.1): 補足の発話時刻と動作時刻を照合する
ff22de6 🎬 feat(prml-3.1): 視覚補足付き480p動画を更新
```

本レポートを含む最終ドキュメントコミットとPRのマージコミットは最終回答に記載する。
