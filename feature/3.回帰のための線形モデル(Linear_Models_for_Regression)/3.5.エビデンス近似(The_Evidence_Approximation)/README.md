# PRML 3.5 エビデンス近似

PRML Chapter 3, Section 3.5 "The Evidence Approximation" の日本語解説動画です。

## 成果物

- `prml_3_5_evidence_approximation.py`: Manim アニメーション
- `make_voicevox_narration.py`: VOICEVOX ナレーション生成
- `narration_script.md`: 台本と scene 設計
- `assets/voicevox/`: 生成済みナレーション音声
- `media/videos/prml_3_5_evidence_approximation/480p15/PRML35EvidenceApproximation.mp4`: レンダリング済み動画

## 実行

```bash
python3 make_voicevox_narration.py
uv run manim --progress_bar none --disable_caching --flush_cache -ql prml_3_5_evidence_approximation.py PRML35EvidenceApproximation
```

VOICEVOX Engine は既定で `http://127.0.0.1:50021` を参照します。途中から音声生成を再開する場合は、次のように実行します。

```bash
python3 make_voicevox_narration.py --from-scene scene04
```

## 内容

1. 完全ベイズとエビデンス近似の違い
2. evidence を `p(t|w,beta)p(w|alpha)` の積分として見る直感
3. log evidence の項分解
4. evidence 最大化による `alpha` の選択
5. 有効パラメータ数 `gamma`
6. `beta` 更新と `N - gamma` の自由度補正
7. `alpha`、`beta`、`gamma` の反復再推定
8. 固定基底関数の限界への橋渡し

## クレジット

ナレーション音声: VOICEVOX:WhiteCUL
