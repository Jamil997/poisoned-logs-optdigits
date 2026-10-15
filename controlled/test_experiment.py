import unittest,itertools
import numpy as np
from scipy.stats import binom
from experiment import TYPES,probabilities,favorable_types,lower_tables,attack_counts,choose,score

class ExactGateTests(unittest.TestCase):
 def test_analytic_values(self):
  s,z,a,r=TYPES.T;probs=probabilities(.03,.001);fav=favorable_types()
  np.testing.assert_allclose((fav@probs-.5)*[.1,.9],[.003,-.0009],atol=1e-16)
  self.assertAlmostEqual(sum(probs),1)
 def test_clopper_pearson_coverage_by_enumeration(self):
  for n in [5,20,100]:
   tables,_=lower_tables(n,np.array([.1,.9]));lower=tables['binomial'][0]/.1+.5
   for p in [.001,.05,.2,.5,.8,.95,.999]:
    error=binom.pmf(np.arange(n+1),n,p)[lower>p+1e-13].sum();self.assertLessEqual(error,.025+1e-12)
 def test_exact_attacks_against_exhaustive_row_flips(self):
  # Both required attack groups have enough rows. Enumerate actual edited rows.
  counts=np.full(16,2);n=sum(counts);fav=favorable_types();expanded=np.repeat(np.arange(16),counts);base=fav@counts
  t=np.arange(n+1)
  tables=[np.stack([.1*(t/n-.45),.9*(t/n-.52)]),np.stack([.1*(t/n-.4)**3,.9*(t/n-.5)**3])]
  rank={1:0,-1:1,0:2}
  for k in [1,2]:
   for table in tables:
    for kd in [0,k]:
     ans=attack_counts(counts,k,table,kd);best=2
     for size in range(k+1):
      for ids in itertools.combinations(range(n),size):
       edits=np.bincount(expanded[list(ids)],minlength=16) if size else np.zeros(16,dtype=int)
       tt=base+(1-2*fav.astype(int))@edits;sel=choose(score(table,tt,kd));best=min(best,rank[sel])
     self.assertEqual(rank[ans[0]],best)
 def test_corruption_monotonicity(self):
  tables,_=lower_tables(100,np.array([.1,.9]))
  for family,table in tables.items():
   self.assertTrue(np.all(np.diff(table,axis=1)>=0))
   for clean in range(101):
    for k in [1,5,10]:
     for observed in range(max(0,clean-k),min(100,clean+k)+1):
      self.assertTrue(np.all(table[:,max(0,observed-k)]<=table[:,clean]+1e-15))
if __name__=='__main__':unittest.main()
