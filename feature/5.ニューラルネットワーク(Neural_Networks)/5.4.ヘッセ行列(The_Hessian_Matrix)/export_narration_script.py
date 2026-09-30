"""Export source-linked storyboard from the canonical display/speech data."""
import json
from pathlib import Path
from narration_content import SCENES,script_hash
ROOT=Path(__file__).parent
p=ROOT/'assets/voicevox/manifest.json'
entries={e['id']:e for e in json.loads(p.read_text())['scenes']} if p.exists() else {}
lines=['# PRML 5.4 ヘッセ行列：傾きの変化を、動かして見る','',
'Bishop (2006), 印刷 pp.249–256（PDF pp.269–276）を全文抽出して照合。§5.4 に独自の図・表はない。Fig.5.1（p.228）の二層構造と Fig.5.6（p.239）の曲率を補助参照し、自作数値例で描く。',
'前提：高校数学。微分は傾きと小さな変化の倍率として導入。数式字幕と読み上げは分離。各文のPCM長に同期。','']
for s in SCENES:
    e=entries.get(s['id'],{})
    duration=e.get('duration','生成前') if e.get('script_sha256')==script_hash(s) else '生成前'
    lines += [f"## {s['id']}：{s['title']}",'',f"原文：{s['reference']}。尺：{duration} 秒。",'']
    for i,b in enumerate(s['beats'],1):
        lines += [f"### {i}. {b['visual_note']}",'']
        for v in b['segments']:lines += [f"- 字幕：{v['display']}",f"- 読み上げ：{v['speech']}"]
        lines.append('')
lines += ['## 原文との対応・成立条件','',
'- 公式正誤表（2011-09-21）を適用。(5.83) は外積の後半に転置が必要。(5.95) は H を M へ置換し、右辺の j と j′ を交換。本映像は修正版を一入力・一隠れ・一出力の例へ特殊化する。',
'- scene01 は E(s)=1+s+ks²/2 の説明用断面。scene02–03 は H=[[3,c],[c,3]]（c=2 で固有値1,5）。停留点と正定値の条件を併記し、ゼロ固有値だけでは判定しない。二次近似は一般には局所近似。',
'- (5.79)–(5.81): 正確な対角成分を抽出する操作と、再帰中の交差二階微分を捨てる O(W) 近似を区別。滑らかな活性化を前提とする。',
'- (5.82)–(5.84): 一出力・二乗誤差の H=Σbbᵀ+Σr∇²y。残差を縮める映像は目標を変える説明実験であり訓練ではない。平均ゼロかつ無相関という別の根拠も説明。',
'- (5.85): sigmoid＋二値交差エントロピーでは b=∇a（確率になる前の総入力）。多クラスsoftmaxへの拡張は本文の言及に合わせREADMEで補足し、導出は省略。',
'- (5.86)–(5.89): H₀=αI、α=0.3。最終結果は外積和＋αIの逆。整数間は次の外積を0〜1倍する。楕円は dᵀHd=1 の線であり、特定の確率質量を主張しない。',
'- (5.90)–(5.91): 4点中心差分は異なる重みの組、対角は標準3点差分で検証。打ち切り誤差O(ε²)と丸め誤差を区別。計算量は一例あたり、固定費用の活性化を前提とし、実測時間ではない。',
'- (5.92)–(5.95): x=0.8, w=(u,v), z=tanh(ux), y=vz, E=(y−t)²/2。出力曲率 M=1, δ=y−t。Hvv=z², Huu=x²[(vh′)²+δvh″], Huv=xh′(vz+δ)。一般式の三ブロックに対応。',
'- (5.96)–(5.111): R{f}=d f(w+εv)/dε|₀。前向きの値とR値、逆向きのδとRδをすべて計算しHを作らずHvを得る。原文はvᵀH、本映像は対称性による転置Hvを使用。基底方向の反復で全列を復元できる。',
'- 視覚復習: scene05 は 4.4 multivariate（pp.215–216, (4.131)–(4.134)）の赤い楕円・金と青の主軸を縮約。精度の固有値1→5に対し標準偏差1→1/√5。正定値を明示し、A→H、Σ→H⁻¹を対応させる。',
'- 視覚復習: scene07 は 5.3 chain（p.243, (5.50)–(5.53)）の箱の列で、値は前へ、感度は後ろへ。w→uと本編の色への対応を示し、微分gから二階微分Hへ渡す。',
'- 視覚補足 V14b: scene08 は本編の y=vz を面積にする。正の辺・増分は説明用の条件であり積の微分の一般的な制限ではない。Δv=εRv、Δz=εRz、増分の積はε²RvRz。εで割ると角の寄与はεに比例して消え、R(vz)=zRv+vRzが残る。(5.103)へ接続する。',
'- 本節の式の全添字展開を読み上げず、重要式と特殊化した三ブロック、R再帰を画面に表示する。最適化・再学習・刈り込み・Laplaceの用途は紹介に留める。',
'- 正誤表：https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-1st-20110921.pdf','']
(ROOT/'narration_script.md').write_text('\n'.join(lines))
