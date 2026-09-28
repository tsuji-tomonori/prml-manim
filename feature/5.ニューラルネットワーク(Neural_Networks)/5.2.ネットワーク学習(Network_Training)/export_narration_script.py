"""Export the editable source into a readable storyboard."""
import json
from pathlib import Path
from narration_content import SCENES
p=Path(__file__).parent
manifest=p/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(manifest.read_text())['scenes']} if manifest.exists() else {}
lines=['# PRML 5.2 ネットワーク学習：問いから動き、式へ','',
'原文：Bishop (2006) 印刷 pp.232–241（PDF pp.252–261）。§5.2.1–5.2.4を含む。図5.5・5.6は自作数値例で再構成。この節に表はない。',
'前提：高校数学。微分はその場の傾き、勾配は各重みに関する傾き、ヘッセ行列は曲がり方として導入する。',
'各段階は対応する文のPCM長で進行し、末尾だけ約0.35秒の呼吸を置く。字幕の $...$ は MathTex。','']
for s in SCENES:
    e=entries.get(s['id'],{})
    lines += [f"## {s['id']}：{s['title']}",'',f"原文：{s['reference']}。音声尺：{e.get('duration','生成前')} 秒。",'']
    for i,b in enumerate(s['beats'],1):
        lines += [f"### {i}. {b['visual_note']}",'']
        for v in b['segments']:
            lines += [f"- 字幕：{v['display']}",f"- 読み上げ：{v['speech']}"]
        lines.append('')
lines += ['## 成立条件と原文との対応','',
'- (5.13) は β を含む負の対数尤度。(5.14) は重み最適化で β>0 を固定した二乗和。(5.15) は残差二乗和 / N、複数出力 (5.17) は / NK。',
'- (5.18) は一例の対応する出力と誤差の組に対する ∂E_n/∂a_k=y_k−t_k。全重みの勾配そのものではない。',
'- 複数二値出力は条件付き独立の仮定、softmax は排他的なクラス。隠れ層の特徴共有は独立仮定と矛盾しない。',
'- 図5.5に対応する多峰性の曲線は教育用関数であり、冒頭ネットワークの誤差とは別。(5.26) は停留条件であり十分条件ではない。',
'- (5.28)–(5.32) は局所二次近似。図5.6と(5.33)–(5.39)を楕円と回転で説明。(5.37)の正定値はゼロでない任意のベクトルに対する条件。',
'- 正定値ヘッセ行列は厳密局所最小の十分条件。ゼロ固有値を含む最小点も存在する。',
'- §5.2.3の O(W³) と O(W²) は原文の二次近似での情報量の概算。任意のネットワークに対する収束回数の保証ではない。固定データ数・一例あたりの計算を区別。',
'- (5.41) は全データの和の勾配。(5.43) は一例の勾配。オンラインの揺らぎからの脱出は可能性であり保証ではない。','']
(p/'narration_script.md').write_text('\n'.join(lines))
