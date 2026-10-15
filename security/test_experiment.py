"""Small exhaustive checks for the attack and the confidence/correction machinery."""
import itertools, json, unittest
import numpy as np
from experiment import ROOT, active_branch, attack, build_specs, envelopes, possible_outcomes, score, choose, utility

class MathematicalChecks(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(81)
        self.config=json.loads((ROOT/'protocol.json').read_text())

    def test_topk_against_exhaustive_replacements(self):
        for n in [4,7]:
          for _ in range(10):
            r=self.rng.random(n);c=self.rng.normal(size=(3,n))
            gate=dict(c=c,off=self.rng.normal(size=3))
            for k in [0,1,2]:
                values=[]
                for support in itertools.combinations(range(n),k):
                  for replacements in itertools.product([0.,1.],repeat=k):
                    rp=r.copy();rp[list(support)]=replacements;values.append(score(gate,rp,0))
                np.testing.assert_allclose(score(gate,r,k),np.min(values,axis=0),atol=1e-12)

    def test_all_real_envelopes_and_possible_outcomes(self):
        n=10
        for _ in range(100):
            r=self.rng.random(n);gate=dict(c=self.rng.normal(size=(4,n)),off=self.rng.normal(size=4))
            for ka,kd in [(2,0),(2,1),(2,2),(2,3)]:
                lo,hi=envelopes(gate,r,ka,kd)
                rp=r.copy();rp[self.rng.choice(n,ka,replace=False)]=self.rng.random(ka)
                ss=score(gate,rp,kd)
                self.assertTrue(np.all(ss>=lo-1e-12) and np.all(ss<=hi+1e-12))
                self.assertIn(choose(ss),possible_outcomes(lo,hi))

    def test_binary_active_branch_identity(self):
        n=7;r=self.rng.integers(2,size=n).astype(float)
        gate=dict(c=self.rng.normal(size=(3,n)),off=self.rng.normal(size=3))
        for kd in [0,1,3]:
          allz=np.array(list(itertools.product([0,1],repeat=n)))
          for j in range(3):
            branches=[active_branch(gate,r,z,kd,j) for z in allz]
            for z in allz:
                exact=score(gate,r+(1-2*r)*z,kd)[j]
                self.assertAlmostEqual(max(a+b@z for a,b in branches),exact,places=11)

    def test_attack_witness_and_certificate_against_bruteforce(self):
        for _ in range(12):
            n=7;r=self.rng.integers(2,size=n).astype(float)
            gate=dict(c=self.rng.normal(size=(3,n)),off=self.rng.normal(size=3)*.1)
            truth=np.array([-.2,.1,.3]);ka=2
            for kd in [0,2]:
                result=attack(gate,r,ka,kd,truth,self.config)
                possible=[]
                for k in range(ka+1):
                  for ids in itertools.combinations(range(n),k):
                    rp=r.copy();rp[list(ids)]=1-rp[list(ids)]
                    possible.append(utility(choose(score(gate,rp,kd)),truth))
                optimum=min(possible)
                self.assertLessEqual(result['floor'],optimum+1e-10)
                self.assertGreaterEqual(utility(result['selected'],truth),optimum-1e-10)
                if result['gap']<1e-10: self.assertAlmostEqual(utility(result['selected'],truth),optimum)
                if kd==0: self.assertAlmostEqual(utility(result['selected'],truth),optimum)

    def test_dr_unbiasedness_and_shrinkage_bias_bound(self):
        X=13;M=6;A=10
        p=self.rng.dirichlet(np.ones(A),size=(M+1,X));q=self.rng.dirichlet(np.ones(A),size=X)
        y=self.rng.integers(A,size=X);r=np.eye(A)[y]
        data=dict(p=p,q=q,pv=p,qv=q,yv=y)
        mu,specs,_=build_specs(data,5000,.2,self.config)
        truth=((p[1:]-p[0])*r).sum(axis=2).mean(axis=1)
        for name,s in specs.items():
            z=s['z0']+s['c']*r
            expectation=(mu*z).sum(axis=2).mean(axis=1)
            if name!='dros_b': np.testing.assert_allclose(expectation,truth,atol=1e-12)
            self.assertTrue(np.all(expectation-truth<=s['bias']+1e-12))
            second=(mu*z*z).sum(axis=2).mean(axis=1)
            self.assertTrue(np.all(second<=s['V']+1e-12))
            # Adversarial alternative rewards: the label-free moment/bias limits
            # must hold without any correct reward-model assumption.
            for _ in range(20):
                other=self.rng.random((X,A));z=s['z0']+s['c']*other
                target=((p[1:]-p[0])*other).sum(axis=2).mean(axis=1)
                exp=(mu*z).sum(axis=2).mean(axis=1)
                self.assertTrue(np.all(exp-target<=s['bias']+1e-12))
                self.assertTrue(np.all((mu*z*z).sum(axis=2).mean(axis=1)<=s['V']+1e-12))

if __name__=='__main__': unittest.main(verbosity=2)
