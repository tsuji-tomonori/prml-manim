"""Render the reviewed display/speech source as a readable storyboard."""
import json
from pathlib import Path
from narration_content import SCENES

ROOT = Path(__file__).resolve().parent

def main():
    p = ROOT/'assets/voicevox/manifest.json'
    manifest = json.loads(p.read_text()) if p.exists() else {}
    entries = {e['id']: e for e in manifest.get('scenes', [])} if isinstance(manifest, dict) else {}
    lines=['# PRML 1.3 モデル選択：台本', '',
           '原文：Bishop (2006) 印刷 pp.32–33（PDF の52–53ページ）。Fig.1.18 / Eq.(1.73)。この節に表はない。',
           '自作の正弦波＋独立ガウスノイズを使用。問い → 実験 → 式の順で進む。',
           '字幕の数式は $...$ 内を MathTex で表示する。speech は音声専用。', '',
           '## 原文との対応上の注意', '',
           '- 多項式の次数は d。式 (1.73) の M は調整可能なパラメータ数。既知分散の実験では M=d+1。',
           '- p(D|w_ML) は尤度、その対数 ln p(D|w_ML) が対数尤度。原文本文の語の省略と区別する。',
           '- Eq.(1.73) は最大化するスコア。通常の AIC=-2 ln p+2M は最小化する。',
           '- 平均二乗誤差と既知分散の対数尤度の展開式は説明用。1.3 節の番号付き式としない。',
           '- 各次数の間の連続変形は係数補間。整数の候補で評価する。',
           '- Fig.1.18 は4組の役割交替として再構成し、毎回再学習する。等しい組の誤差を平均する。',
           '- 最終評価は選択済みモデルのみ。CV の開発データは別の自作24点。テストはCVに含めない。',
           '- 計算量の格子は全組合せ探索の例。Bayes の曲線群は模式図で、事後分布の標本ではない。', '',
           '| シーン | 題名 | 音声実測秒 | 原文 |','|---|---|---:|---|']
    for s in SCENES:
        e=entries.get(s['id'],{})
        lines.append(f"| {s['id']} | {s['title']} | {e.get('duration','生成後に記録')} | {s['reference']} |")
    for s in SCENES:
        e=entries.get(s['id'],{})
        lines += ['',f"## {s['id']} {s['title']}",'',f"原文参照：{s['reference']}",'']
        for i,b in enumerate(s['beats']):
            duration=e.get('beat_durations',[None]*len(s['beats']))[i]
            lines += [f"### Beat {i+1}（{duration if duration else '音声生成後に確定'} 秒）",'',f"画面の焦点：{b['visual_note']}",'']
            for c in b['segments']:
                lines += [f"- {c['id']} 字幕：{c['display']}",f"- 読み上げ：{c['speech']}"]
            lines.append('')
    (ROOT/'narration_script.md').write_text('\n'.join(lines).rstrip()+'\n')

if __name__=='__main__':
    main()
