"""Independent enumeration of Bishop's gauge table and graph implications."""
import json
from fractions import Fraction as Q
from pathlib import Path
from gauge_model import JOINT_G0,P_G0,P_F0_G0,P_F0_G0_B0,RAIN

p_b={0:Q(1,10),1:Q(9,10)};p_f={0:Q(1,10),1:Q(9,10)}
g1={(1,1):Q(8,10),(1,0):Q(2,10),(0,1):Q(2,10),(0,0):Q(1,10)}
states={(b,f,g):p_b[b]*p_f[f]*(g1[b,f] if g else 1-g1[b,f])
        for b in (0,1) for f in (0,1) for g in (0,1)}
assert sum(states.values())==1
assert {k:states[k[0],k[1],0] for k in JOINT_G0}==JOINT_G0
assert P_G0==Q(315,1000)==sum(v for (b,f,g),v in states.items() if g==0)
assert P_F0_G0==Q(9,35)==Q(81,315)
assert P_F0_G0_B0==Q(1,9)==Q(9,81)
assert Q(1,10)<P_F0_G0_B0<P_F0_G0
# Initial independence, followed by dependence after selecting the collider G=0.
assert sum(v for (b,f,g),v in states.items() if b==0 and f==0)==Q(1,100)
assert P_F0_G0_B0!=P_F0_G0
rain=RAIN
assert rain['p_umbrella_rain']==Q(8,10) and rain['p_umbrella_dry']==Q(2,10)
p_r={0:1-rain['p_rain'],1:rain['p_rain']}
p_d1={0:rain['p_wet_dry'],1:rain['p_wet_rain']}
p_u1={0:rain['p_umbrella_dry'],1:rain['p_umbrella_rain']}
fork={(r,d,u):p_r[r]*(p_d1[r] if d else 1-p_d1[r])*(p_u1[r] if u else 1-p_u1[r])
      for r in (0,1) for d in (0,1) for u in (0,1)}
for r in (0,1):
    conditional=fork[r,1,1]/p_r[r]
    assert conditional==p_d1[r]*p_u1[r]
p_du=sum(fork[r,1,1] for r in (0,1))
p_d=sum(fork[r,1,u] for r in (0,1) for u in (0,1))
p_u=sum(fork[r,d,1] for r in (0,1) for d in (0,1))
assert p_du!=p_d*p_u

# Separate illustrative chain; the displayed 0.62 is p(b=1|c=1).
p_a={0:Q(1,2),1:Q(1,2)}
p_c1={0:Q(2,10),1:Q(8,10)}
p_right1={0:Q(15,100),1:Q(62,100)}
chain={(a,c,b):p_a[a]*(p_c1[a] if c else 1-p_c1[a])*(p_right1[c] if b else 1-p_right1[c])
       for a in (0,1) for c in (0,1) for b in (0,1)}
for c in (0,1):
    for a in (0,1):
        assert chain[a,c,1]/sum(chain[a,c,b] for b in (0,1))==p_right1[c]
assert sum(chain[1,c,1] for c in (0,1))!=p_a[1]*sum(chain[a,c,1] for a in (0,1) for c in (0,1))

# Observing a noisy descendant of a collider induces dependence too.
p_c1_given_e={0:Q(1,10),1:Q(9,10)}
desc={(a,b,c):p_a[a]*p_a[b]*(p_c1_given_e[a or b] if c else 1-p_c1_given_e[a or b])
      for a in (0,1) for b in (0,1) for c in (0,1)}
pc=sum(desc[a,b,1] for a in (0,1) for b in (0,1))
pab=desc[1,1,1]/pc
pa=sum(desc[1,b,1] for b in (0,1))/pc
pb=sum(desc[a,1,1] for a in (0,1))/pc
assert pab!=pa*pb
result=dict(source='Bishop (2006) §8.2 pp.377–378, Eqs.8.30–8.33',
            joint_g0={f'B={b},F={f}':float(v) for (b,f),v in sorted(JOINT_G0.items())},
            p_g0=float(P_G0),p_f0_g0=float(P_F0_G0),p_f0_g0_b0=float(P_F0_G0_B0),
            rational=dict(p_g0=str(P_G0),p_f0_g0=str(P_F0_G0),p_f0_g0_b0=str(P_F0_G0_B0)),
            checks=['joint probability sums to 1','printed gauge result','before/after collider dependence',
                    '8.1 rain fork: conditional independent, marginal dependent',
                    'illustrative chain: conditional independent, marginal dependent',
                    'illustrative noisy collider descendant: conditional dependent'])
(Path(__file__).resolve().parent/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',result)
