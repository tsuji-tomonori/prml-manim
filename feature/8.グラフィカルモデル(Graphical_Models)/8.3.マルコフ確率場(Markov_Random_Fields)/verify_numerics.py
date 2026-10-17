"""Independent checks of the picture, local energies and ICM update record."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from image_model import BETA, ETA, FLIPS, H, SIDE, TARGET, energy, icm, local_energy, observed, original

clean = original()
y = observed()
assert clean.shape == y.shape == (12,12)
assert len(FLIPS) == len(set(FLIPS)) == 14
assert int((clean != y).sum()) == 14
assert int((clean == y).sum()) == 130
assert int(y[TARGET]) == 1 and int(clean[TARGET]) == -1
assert [int(y[r,c]) for r,c in ((4,5),(6,5),(5,4),(5,6))] == [-1]*4

# Evaluate the complete energy without calling image_model.energy.
def separate_energy(x):
    total = 0.
    for r in range(SIDE):
        for c in range(SIDE):
            value = int(x[r,c])
            total += H*value - ETA*value*int(y[r,c])
            if r+1<SIDE:
                total -= BETA*value*int(x[r+1,c])
            if c+1<SIDE:
                total -= BETA*value*int(x[r,c+1])
    return total

assert math.isclose(energy(y,y),separate_energy(y),abs_tol=1e-8)
candidate={}
for value in (-1,1):
    image=y.copy()
    image[TARGET]=value
    candidate[value]=dict(local=local_energy(y,y,*TARGET,value),global_energy=separate_energy(image))
assert math.isclose(candidate[1]['local'],1.9,abs_tol=1e-8)
assert math.isclose(candidate[-1]['local'],-1.9,abs_tol=1e-8)
assert math.isclose(candidate[1]['global_energy']-candidate[-1]['global_energy'],3.8,abs_tol=1e-8)

x,history,changes=icm(y)
assert [len(c) for c in changes] == [8,0]
assert int((x==clean).sum()) == 138
assert int(x[TARGET]) == -1
energies=[separate_energy(state) for state in history]
assert all(b<=a+1e-8 for a,b in zip(energies,energies[1:]))
for row in range(SIDE):
    for col in range(SIDE):
        now=int(x[row,col])
        other=-now
        assert local_energy(x,y,row,col,now)<=local_energy(x,y,row,col,other)+1e-9

# A separate four-state example checks the positive-potential normalization.
small=[(-1,-1),(-1,1),(1,-1),(1,1)]
weights=[math.exp(BETA*a*b) for a,b in small]
z=sum(weights)
assert all(v>0 for v in weights)
assert math.isclose(sum(v/z for v in weights),1.,abs_tol=1e-12)

# The 16 bars in scene07 use a separate, explicitly stated 2x2 Ising model.
grid_weights=[]
for state in itertools.product((-1,1),repeat=4):
    a,b,c,d=state
    grid_weights.append(math.exp(.13*(a*b+a*c+b*d+c*d)+.10*(a+b+c+d)))
grid_z=sum(grid_weights)
assert math.isclose(grid_z,16.986362382408668,abs_tol=1e-12)
assert math.isclose(sum(v/grid_z for v in grid_weights),1.,abs_tol=1e-12)

result=dict(shape=[SIDE,SIDE], flipped=len(FLIPS), flip_fraction=len(FLIPS)/SIDE**2,
            target=list(TARGET), target_observation=int(y[TARGET]),
            target_neighbours=[int(y[r,c]) for r,c in ((4,5),(6,5),(5,4),(5,6))],
            target_local_energy=dict(black=candidate[-1]['local'],white=candidate[1]['local']),
            sweep_changes=[len(c) for c in changes], initial_agreement=130,
            restored_agreement=138, restored_fraction=138/144,
            energies=energies, small_partition=z, grid_2x2_partition=grid_z)
Path('numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS',result)
