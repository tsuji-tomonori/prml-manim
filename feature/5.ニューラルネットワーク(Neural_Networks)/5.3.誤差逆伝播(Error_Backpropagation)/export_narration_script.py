"""Export the independent caption/speech storyboard and measured durations."""
import json
from pathlib import Path
from narration_content import SCENES, script_hash
ROOT=Path(__file__).parent
path=ROOT/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(path.read_text())['scenes']} if path.exists() else {}
lines=['# PRML 5.3 誤差逆伝播：問いから動き、式へ','',
'原文：Bishop (2006), 印刷 pp.241–249（PDF pp.261–269）、§5.3.1–5.3.4。Fig.5.7 の分岐と Fig.5.8 のモジュール結合を自作例で再構成。本節に表はない。',
'前提：高校数学。微分は小さな変更の倍率として導入。字幕の $...$ は MathTex。各文の音声長に合わせて動作する。','']
for s in SCENES:
    e=entries.get(s['id'],{})
    duration=e.get('duration','生成前') if e.get('script_sha256') == script_hash(s) else '生成前'
    lines += [f"## {s['id']}：{s['title']}",'',f"原文：{s['reference']}。音声尺：{duration} 秒。",'']
    for i,b in enumerate(s['beats'],1):
        lines += [f"### {i}. {b['visual_note']}",'']
        for v in b['segments']:lines += [f"- 字幕：{v['display']}",f"- 読み上げ：{v['speech']}"]
        lines.append('')
lines += ['## 成立条件と補足','',
'- (5.54) の出力差は線形出力＋二乗和、sigmoid＋二値交差エントロピー、softmax＋多クラス交差エントロピーなど対応する組で成立。一般の組では連鎖律を使う。',
'- δ は誤差の総入力 a に関する微分。(5.75)–(5.76) の δ_kj はクロネッカーのデルタで、感度 δ_j とは別。画面で δ_kj=1 (k=j), 0 (k≠j) と定義。',
'- (5.56) の和は、そのユニットから直接接続する全ての先を対象とする。隠れ層と出力層のどちらも含み得る。異なる活性化関数でも局所微分を使える。',
'- 重み行列は行が接続先、列が接続元。バイアスは先頭列、入力 x_0=1、隠れ z_0=1。',
'- (5.57) は誤差の和に対応する勾配の和。平均誤差を使う場合は勾配も N で割る。',
'- 計算量は一例、固定費用の活性化と接続の計算を前提とする漸近評価。図中 W と 2W² は増え方の模型で、実測時間ではない。中心差分の誤差は滑らかな関数・正確な算術で O(ε²)。実際は丸め誤差もある。',
'- (5.72) は小さい入力変化の局所近似で、J は基準入力ごとに再計算。(5.73)–(5.74) は選択した出力から逆伝播、(5.75)–(5.76) は出力側の開始条件。(5.77) の入力中心差分でも数値確認する。',
'- Fig.5.8 は二つの前処理から一つのモジュールへ合流する構造を保持。前処理のパラメータ微分に途中のヤコビ行列を使う (5.71)。','']
(ROOT/'narration_script.md').write_text('\n'.join(lines))
