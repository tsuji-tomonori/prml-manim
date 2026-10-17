"""Bishop §8.2's four-state fuel-gauge example, in exact rational arithmetic."""
from fractions import Fraction

P_B = {0: Fraction(1,10), 1: Fraction(9,10)}
P_F = {0: Fraction(1,10), 1: Fraction(9,10)}
P_G1 = {(1,1):Fraction(8,10), (1,0):Fraction(2,10),
        (0,1):Fraction(2,10), (0,0):Fraction(1,10)}
JOINT_G0 = {(b,f):P_B[b]*P_F[f]*(1-P_G1[b,f]) for b in (0,1) for f in (0,1)}
P_G0 = sum(JOINT_G0.values())
P_F0_G0 = sum(w for (b,f),w in JOINT_G0.items() if f==0)/P_G0
P_F0_G0_B0 = JOINT_G0[0,0]/sum(w for (b,f),w in JOINT_G0.items() if b==0)

# The 8.1 rain/road/umbrella model is reused only as a brief same-chapter link.
RAIN = dict(p_rain=Fraction(3,10), p_wet_rain=Fraction(9,10),
            p_wet_dry=Fraction(1,10), p_umbrella_rain=Fraction(8,10),
            p_umbrella_dry=Fraction(2,10))
