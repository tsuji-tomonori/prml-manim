"""Export the authoritative display/speech data and measured durations."""
import json
from pathlib import Path
from narration_content import SCENES, script_hash

ROOT = Path(__file__).resolve().parent

def main():
    path = ROOT / 'assets/voicevox/manifest.json'
    entries = {s['id']: s for s in json.loads(path.read_text())['scenes']} if path.exists() else {}
    lines = ['# PRML 4.3 確率的識別モデル：収録台本', '',
             '問い → 連続した視覚実験 → 数式。字幕 display と読み上げ speech は別管理。',
             '原文は印刷 pp.203–213（手元PDF pp.223–233）。本節に表はない。',
             'Fig.4.12 は二乗基底と円による独自例、Fig.4.13 は標準ガウスによる独自例。', '',
             '## 原文の訂正と成立条件', '',
             '- pp.207–208: 誤差は convex（凸）。有限重みで Hessian は半正定値、特徴行列の列が独立なら正定値。完全分離では有限の最尤解は存在しない。',
             '- Eq.(4.110): 先頭のマイナスを除く。softmax 全クラス共通シフトの冗長性にも注意。',
             '- Eqs.(4.114)–(4.116): 活性化は標準正規CDF（inverse probit）、リンクはその逆関数。標準の erf で Phi(a)=(1+erf(a/sqrt(2)))/2。',
             '- Eq.(4.117): 観測ラベルが1となる確率として表示。独立な反転率 0<=epsilon<1/2。',
             '- Eq.(4.124): 左辺は grad E。尺度 1/s を保持する。',
             '- [著者の正誤表](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf) p.12。', '',
             'パラメータ数の比較、二乗誤差Newtonの導出(4.93–95)、erfの積分定義、多クラスHessianの詳細展開は台本の主線から省く。',
             '重要式は映像に表示し、専門語は先に意味を説明する。途中の連続補間は視覚化で、IRLSの更新値は各反復の整数位置。', '']
    for scene in SCENES:
        entry = entries.get(scene['id'], {})
        if entry.get('script_sha256') != script_hash(scene):
            entry = {}
        lines += [f"## {scene['id']} {scene['title']}", '', f"原文: {scene['reference']}",
                  f"音声尺: {entry.get('duration', '未生成')} 秒", '']
        for i, beat in enumerate(scene['beats']):
            duration = entry.get('beat_durations', [None]*len(scene['beats']))[i]
            lines += [f"### Beat {i+1}: {beat['visual_note']}", '', f'実測: {duration} 秒', '']
            for s in beat['segments']:
                lines += [f"- {s['id']} 字幕: {s['display']}", f"  読み上げ: {s['speech']}"]
            lines += ['']
    (ROOT/'narration_script.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()
