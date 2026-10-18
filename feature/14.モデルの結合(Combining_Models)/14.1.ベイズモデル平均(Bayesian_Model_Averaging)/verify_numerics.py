"""Check exact probabilities and the placement of product and summation."""
import json
from fractions import Fraction
from pathlib import Path
from itertools import product
from model_average import (OBSERVATIONS, P_RED, PRIOR, point_probability,
    data_likelihood, mixture_probability, posterior, predictive_red,
    local_data_probability, global_data_probability)

assert (data_likelihood('A'), data_likelihood('B')) == (Fraction(64,625), Fraction(4,625))
assert posterior() == {'A':Fraction(16,17),'B':Fraction(1,17)}
assert predictive_red() == Fraction(13,17)
assert global_data_probability((*OBSERVATIONS, '赤')) / global_data_probability() == Fraction(13,17)
assert local_data_probability((*OBSERVATIONS, '赤')) / local_data_probability() == Fraction(1,2)
assert mixture_probability('赤') == Fraction(1,2)
assert local_data_probability() == Fraction(1,16)
assert global_data_probability() == Fraction(34,625)
# Enumerate all 2^4 latent assignments independently of the closed product form.
local_direct=sum((__import__('math').prod(PRIOR[h]*point_probability(x,h) for x,h in zip(OBSERVATIONS,hs))
                  for hs in product(PRIOR,repeat=len(OBSERVATIONS))),Fraction(0))
assert local_direct == local_data_probability()
# Enumerate the single global selector separately.
global_direct=sum((PRIOR[h]*__import__('math').prod(point_probability(x,h) for x in OBSERVATIONS)
                   for h in PRIOR),Fraction(0))
assert global_direct == global_data_probability()
assert sum(posterior().values()) == 1
result={'observations':OBSERVATIONS,'local_probability':str(local_direct),'global_probability':str(global_direct),
        'likelihood':{h:float(data_likelihood(h)) for h in PRIOR},
        'posterior':{h:float(p) for h,p in posterior().items()},
        'predictive_red':float(predictive_red()),'exact_predictive_red':str(predictive_red()),
        'enumeration':'passed'}
Path('numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(result)
