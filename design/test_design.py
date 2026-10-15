import unittest,itertools
import numpy as np
from confirmation import all_designs,spec,core

class DesignTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(731);self.p=rng.dirichlet(np.ones(10),size=(4,7));self.y=np.arange(7)
 def test_importance_identity_and_exact_binary_second_moment(self):
  p=self.p;delta=p[1:]-p[0];truth=delta[:,np.arange(7),self.y].mean(axis=1)
  for Q in all_designs(p).values():
   self.assertAlmostEqual(Q.sum(),1);self.assertGreater(Q.min(),0)
   s=spec(p,Q,20000);r=np.eye(10)[self.y];z=s['c']*(r-.5)
   np.testing.assert_allclose((Q*z).sum(axis=(1,2)),truth,atol=1e-14)
   np.testing.assert_allclose((Q*z*z).sum(axis=(1,2)),s['V'],atol=1e-14)
 def test_general_rewards_and_moment_bound(self):
  rng=np.random.default_rng(313);r=rng.random((7,10));delta=self.p[1:]-self.p[0]
  for Q in all_designs(self.p).values():
   s=spec(self.p,Q,20000);z=s['c']*(r-.5)
   np.testing.assert_allclose((Q*z).sum(axis=(1,2)),(delta*r).sum(axis=(1,2))/7,atol=1e-14)
   self.assertTrue(np.all((Q*z*z).sum(axis=(1,2))<=s['V']+1e-14))
 def test_global_influence_bound_and_minimax_normalizer(self):
  delta=(self.p[1:]-self.p[0])/7;w=np.max(abs(delta),axis=0);W=w.sum()
  for name,Q in all_designs(self.p).items():
   if name.startswith('joint_') and name!='joint_rms' and float(name[6:])>0:
    lam=float(name[6:]);self.assertLessEqual(np.max(abs(delta/Q)),W/(.99*lam)+1e-12)
  Q=w/W;self.assertAlmostEqual(np.max(abs(delta/Q)),W)
 def test_uniform_context_equivalence(self):
  p=self.p;mu=.8*p[0]+.02;Q=all_designs(p)['joint_0'];s=spec(p,Q,20000)
  np.testing.assert_allclose(s['c'],(p[1:]-p[0])/mu)
 def test_fractional_reward_envelopes(self):
  rng=np.random.default_rng(81);gate=dict(c=rng.normal(size=(3,5)),off=rng.normal(size=3));r=np.array([0,1,0,1,1.])
  for ka,kd in [(1,0),(1,1),(2,2)]:
   lo,hi=core.envelopes(gate,r,ka,kd)
   for indices in itertools.combinations(range(5),ka):
    for values in itertools.product([0,.19,.55,1],repeat=ka):
     rp=r.copy();rp[list(indices)]=values;s=core.score(gate,rp,kd)
     self.assertTrue(np.all(s>=lo-1e-12));self.assertTrue(np.all(s<=hi+1e-12))
 def test_defender_functions_have_no_label_argument(self):
  # Only policy arrays are accepted, and all designs/specifications work on them.
  ds=all_designs(self.p)
  for Q in ds.values():self.assertTrue(np.isfinite(spec(self.p,Q,20000)['rad']).all())
if __name__=='__main__':unittest.main()
