"""Raw reverse-branch recurrence; fixed, untrained features and weights."""
import json
import math


def level(s):
    if type(s) is not int or s < 0:
        raise ValueError('position must be a nonnegative integer')
    return (s + 1).bit_length() - 1


def buckets(t, reverse=True):
    level(t)
    if reverse:
        return [list(range(2**l-1, min(t+1, 2**(l+1)-1))) for l in range(level(t)+1)]
    # Binary decomposition of the historical prefix, plus current token.
    result, start = [], 0
    for l in range(t.bit_length()-1, -1, -1):
        if t & (1 << l):
            result.append(list(range(start, start + 2**l)))
            start += 2**l
    return result + [[t]]


class ReverseMemory:
    def __init__(self, key_dim, value_dim):
        if any(type(d) is not int or d < 1 for d in (key_dim, value_dim)):
            raise ValueError('dimensions must be positive integers')
        self.dk, self.dv = key_dim, value_dim
        self.states, self.count = [], 0

    @staticmethod
    def vector(x, n):
        if len(x) != n or any(not math.isfinite(v) for v in x):
            raise ValueError('invalid vector')

    def step(self, q, k, v, gate, weights):
        l = level(self.count)
        self.vector(q, self.dk); self.vector(k, self.dk); self.vector(v, self.dv)
        self.vector(weights, l+1)
        if not math.isfinite(gate) or not 0 <= gate <= 1 or any(w < 0 for w in weights):
            raise ValueError('invalid gate or level weight')
        if l == len(self.states):
            self.states.append([[0.0]*self.dk for _ in range(self.dv)])
        for a in range(self.dv):
            for b in range(self.dk):
                self.states[l][a][b] += gate*v[a]*k[b]
        self.count += 1
        return [sum(w * sum(state[a][b]*q[b] for b in range(self.dk))
                    for w, state in zip(weights, self.states))
                for a in range(self.dv)]


def selected_bucket_read(t, target, values, reverse):
    # Identical scalar keys: a bucket gate cannot distinguish its members.
    selected = next(b for b in buckets(t, reverse) if target in b)
    return sum(values[s] for s in selected)


def demo():
    t = 14
    results = {}
    for target, distractor in [(0,1),(14,13),(4,5)]:
        values = [0.0]*15
        values[target], values[distractor] = 1.0, -1.0
        results[f'target_{target}_distractor_{distractor}'] = {
            'forward': selected_bucket_read(t,target,values,False),
            'reverse': selected_bucket_read(t,target,values,True),
            'desired': 1.0}
    memory = ReverseMemory(1,1)
    for s in range(1024):
        memory.step([1],[1],[1],1,[1]*(level(s)+1))
    return {'forward_buckets_at_14':buckets(14,False),
            'reverse_buckets_at_14':buckets(14),
            'hand_selected_bucket_reads':results,
            'reverse_states_after_1024_tokens':len(memory.states),
            'note':'Synthetic oracle bucket selection, not learned retrieval or a benchmark.'}


if __name__ == '__main__':
    print(json.dumps(demo(),indent=2,sort_keys=True))
