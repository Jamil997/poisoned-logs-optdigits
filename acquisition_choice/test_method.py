import itertools,unittest
import numpy as np
from scipy.stats import binom
from method import critical,design_rows,select,scores,plans
from audit import direct_tail
from pathlib import Path
import json
CFG=json.loads((Path(__file__).parent/'protocol.json').read_text())

class Tests(unittest.TestCase):
    def test_calibration_exhaustive(self):
        for n,m,a,c in [(4,3,1,3),(4,3,4,19),(0,12,0,1),(12,0,1,0)]:
            h=critical(n,m,a,c,.05)
            p=sum(binom.pmf(u,n,.5)*binom.pmf(v,m,.5) for u in range(n+1) for v in range(m+1) if a*u+c*v>=h)
            self.assertLessEqual(p,.05)
            for null in [.1,.3,.5]:
                pp=sum(binom.pmf(u,n,null)*binom.pmf(v,m,null) for u in range(n+1) for v in range(m+1) if a*u+c*v>=h)
                self.assertLessEqual(pp,.05)
    def test_large_calibration_and_truncation(self):
        for n,m,a,c in [(19000,1000,4,19),(16000,80,1,32),(0,80000,0,1)]:
            h=critical(n,m,a,c,.0005);p,e=direct_tail(n,m,a,c,h)
            self.assertLessEqual(p+e,.0005)
    def test_trusted_endpoint_and_cost(self):
        for accounting in CFG['accounting_models']:
            pp,_=plans(CFG,20000,5,accounting)
            self.assertIn((0,4000),pp)
            d,_,_,rows=design_rows(CFG,20000,5,accounting,'fixed_200',.01)
            trusted=[r for r in rows if r['n']==0 and r['m']==4000][0]
            self.assertLessEqual(d['objective'],trusted['objective'])
            self.assertTrue(all(r['actual_cost']<=20000 for r in rows))
    def test_source_choice_does_not_receive_quality(self):
        import inspect
        for f in [design_rows,plans]:
            self.assertFalse(any(x in inspect.signature(f).parameters for x in ['beta','theta','g','b','truth','rewards']))
        d=design_rows(CFG,10000,1,'total_purchase','fixed_200',.01)[0]
        self.assertEqual(d['reference_weight'],1)
    def test_exhaustive_attack_and_pilot_conditioning(self):
        patterns=np.array([[0,0],[0,1],[1,0],[1,1]])
        U=np.repeat(patterns,2,axis=0);V=patterns.copy();u=U.sum(axis=0);v=V.sum(axis=0);B=np.array([.1,.9]);k=2
        for a,c in [(1,0),(1,1),(4,19),(0,1)]:
            for hh in [[1,3],[4,5],[8,7]]:
                h=np.array(hh)[:,None];d=max(1,a*len(U)+c*len(V));candidates=[]
                for shift in [[-k,k],[-k,-k],[0,0]]:
                    candidates.append(int(select(scores((u+shift)[:,None],v[:,None],B,a,c,d,h,a*k))[0]))
                rank=lambda s:0 if s==1 else 1 if s<0 else 2
                actual=[]
                for count in range(k+1):
                    for ids in itertools.combinations(range(len(U)),count):
                        uu=u+(1-2*U[list(ids)]).sum(axis=0) if ids else u
                        actual.append(int(select(scores(uu[:,None],v[:,None],B,a,c,d,h,a*k))[0]))
                self.assertEqual(min(map(rank,candidates)),min(map(rank,actual)))

if __name__=='__main__':unittest.main()
