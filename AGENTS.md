# Repository Agent Instructions

このリポジトリで作業する Codex / AI agent は、以下を守る。

## 共通

- 指定 skill が利用可能一覧に出ない場合も、リポジトリローカルの明示ルールとして該当 `SKILL.md` を読む。
- 実施していないテスト、確認、検証を実施済みとして書かない。
- `.working/` は参照用・一時作業用の領域であり、Git 管理対象に含めない。

## Post Task Work Report

- 対象: ファイル編集、コマンド実行、調査、検証、ドキュメント作成など、リポジトリへの実作業。
- 必読: `skills/post-task-fit-report/SKILL.md`
- ユーザーが「レポート不要」と明示しない限り、主作業完了後かつ最終回答前に `reports/working/` へ作業完了レポートを 1 件残す。
- ファイル名は `YYYYMMDD-HHMM-<task-summary>.md` とし、summary は ASCII 小文字とハイフンで短く表す。
- 最終回答では生成したレポートの保存先パスを明示する。

## Git Commit Message

- 対象: Git commit message、コミットメッセージ、コミットコメント、`git commit`。
- 必読: `skills/japanese-git-commit-gitmoji/SKILL.md`
- commit 前に `git diff --cached --name-only` でステージ済みファイルを確認する。
- 1 行目は原則 `<emoji> <type>(<scope>): <日本語の要約>` とする。scope 不要または不明なら省略可。
- 変更目的が複数に分かれる場合は、1 commit にまとめず目的別分割を検討する。
