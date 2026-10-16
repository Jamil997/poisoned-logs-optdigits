import unittest,itertools
import numpy as np
from scipy.stats import binom
from method import tail,threshold,design,weights_for,margins,select,damage
class EvidenceTests(unittest.TestCase):
 def test_exact_tail_against_exhaustive_distribution(self):
  for n,m,a,c in [(4,3,1,3),(3,2,4,1),(5,0,1,0),(0,4,0,1)]:
   for p in [.2,.5,.8]:
    for h in range(a*n+c*m+2):
     direct=sum(binom.pmf(u,n,p)*binom.pmf(v,m,p) for u in range(n+1) for v in range(m+1) if a*u+c*v>=h)
     self.assertAlmostEqual(tail(h,n,m,a,c,p),direct,places=13)
 def test_uniform_null_control(self):
  n,m,a,c=10,4,1,3;h=threshold(n,m,a,c,.05)
  for p in np.linspace(0,.5,21):self.assertLessEqual(tail(h,n,m,a,c,p),.05)
  self.assertGreater(tail(h-1,n,m,a,c),.05-1e-12)
 def test_permutation_equivariance(self):
  for method in ['pooled_equal','pooled_influence','evidence_weighted','pilot_weight_main','pilot_select_main']:
   B=np.array([.1,.9]);U=np.array([[10800,10000],[10000,10001]]);V=np.array([[520,510],[500,520]])
   r=weights_for(method,19000,1000,200,.01,B,V);rp=weights_for(method,19000,1000,200,.01,B[::-1],V[::-1])
   np.testing.assert_allclose(margins(U,V,B,r)[::-1],margins(U[::-1],V[::-1],B[::-1],rp))
 def test_design_and_declared_cap(self):
  d,_=design(19000,1000,200,.01);self.assertEqual(d['fraction'],.2)
  self.assertEqual(damage(4,19,200,0),(800,200,0));self.assertEqual(damage(4,19,200,10),(950,190,10))
 def test_adaptive_extremes_exhaustive(self):
  patterns=np.array([[0,0],[0,1],[1,0],[1,1]]);U=np.repeat(patterns,2,axis=0);V=patterns.copy();B=np.array([.1,.9]);rank={1:0,-1:1,0:2};baseU=U.sum(axis=0);baseV=V.sum(axis=0)
  for cap in [0,1]:
   for a,c in [(1,1),(1,3),(0,1)]:
    k=2;cor,ku,kt=damage(a,c,k,cap)
    for hs in [[4,5],[8,7],[1,3]]:
     rule=(a,c,20,np.array(hs)[:,None],cor);choices=[]
     for shift in [np.array([-1,1]),np.array([-1,-1]),np.array([0,0])]:choices.append(int(select(margins((baseU+ku*shift)[:,None],(baseV+kt*shift)[:,None],B,rule))[0]))
     predicted=min(rank[x] for x in choices);actual=2
     for size in range(k+1):
      for ids in itertools.combinations(range(len(U)+len(V)),size):
       ui=[x for x in ids if x<len(U)];vi=[x-len(U) for x in ids if x>=len(U)]
       if len(vi)>cap:continue
       tu=baseU+(1-2*U[ui]).sum(axis=0) if ui else baseU;tv=baseV+(1-2*V[vi]).sum(axis=0) if vi else baseV
       sel=int(select(margins(tu[:,None],tv[:,None],B,rule))[0]);actual=min(actual,rank[sel])
     self.assertEqual(predicted,actual)
if __name__=='__main__':unittest.main()
