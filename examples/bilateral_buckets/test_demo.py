import copy
import random
import unittest
from . import demo as d


class Tests(unittest.TestCase):
    def test_dense_oracle(self):
        rng=random.Random(48)
        for n in [1,2,3,4,7,8,15,16,33]:
            mem=d.ReverseMemory(3,2)
            history=[]
            for t in range(n):
                q=[rng.uniform(-1,1) for _ in range(3)]
                k=[rng.uniform(-1,1) for _ in range(3)]
                v=[rng.uniform(-1,1) for _ in range(2)]
                gate=rng.random(); weights=[rng.random()*2 for _ in range(d.level(t)+1)]
                history.append((k,v,gate))
                actual=mem.step(q,k,v,gate,weights)
                # Explicit token-pair operator, no matrix-state reuse.
                expected=[sum(weights[(s+1).bit_length()-1]*g*sum(x*y for x,y in zip(q,key))*value[a]
                              for s,(key,value,g) in enumerate(history)) for a in range(2)]
                for a,b in zip(actual,expected): self.assertAlmostEqual(a,b,places=12)

    def test_partition(self):
        for t in range(256):
            for reverse in (True,False):
                self.assertEqual(sum(d.buckets(t,reverse),[]),list(range(t+1)))
            self.assertEqual(len(d.buckets(t)),(t+1).bit_length())

    def test_gate_once_not_decay(self):
        m=d.ReverseMemory(1,1)
        self.assertEqual(m.step([1],[1],[8],.5,[1]),[4])
        self.assertEqual(m.step([1],[1],[100],0,[1,1]),[4])
        self.assertEqual(m.step([1],[1],[2],1,[2,3]),[14])

    def test_completed_bucket_unchanged(self):
        m=d.ReverseMemory(1,1)
        for t in range(3): m.step([1],[1],[t],1,[1]*(d.level(t)+1))
        old=copy.deepcopy(m.states)
        for t in range(3,15): m.step([1],[1],[t],1,[1]*(d.level(t)+1))
        self.assertEqual(m.states[:2],old)

    def test_current_token_and_zero_read(self):
        m=d.ReverseMemory(1,1)
        self.assertEqual(m.step([2],[3],[4],.5,[1]),[12])
        self.assertEqual(m.step([2],[3],[4],1,[0,0]),[0])

    def test_invalid_atomic(self):
        m=d.ReverseMemory(1,1)
        for args in [([1],[1],[1],-1,[1]),([1],[1],[1],1,[-1]),([1],[1],[1],1,[]),([float('nan')],[1],[1],1,[1])]:
            with self.assertRaises(ValueError): m.step(*args)
            self.assertEqual((m.states,m.count),([],0))
        for bad in [-1,True,1.5]:
            with self.assertRaises(ValueError): d.level(bad)
        with self.assertRaises(ValueError): d.ReverseMemory(0,1)

    def test_showcase_matches_recurrence(self):
        for target,distractor in [(0,1),(14,13),(4,5)]:
            m=d.ReverseMemory(1,1)
            values=[0.0]*15
            values[target],values[distractor]=1.0,-1.0
            for t,v in enumerate(values):
                weights=[float(l==d.level(target)) for l in range(d.level(t)+1)]
                out=m.step([1],[1],[v],1,weights)
            self.assertEqual(out,[d.selected_bucket_read(14,target,values,True)])

    def test_collision_controls(self):
        r=d.demo()['hand_selected_bucket_reads']
        self.assertEqual((r['target_0_distractor_1']['forward'],r['target_0_distractor_1']['reverse']),(0,1))
        self.assertEqual((r['target_14_distractor_13']['forward'],r['target_14_distractor_13']['reverse']),(1,0))
        self.assertEqual((r['target_4_distractor_5']['forward'],r['target_4_distractor_5']['reverse']),(0,0))

if __name__=='__main__': unittest.main()
