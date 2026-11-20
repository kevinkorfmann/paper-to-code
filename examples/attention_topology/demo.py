"""Original finite H0 attention diagnostic; standard library only."""
import json
import math
import statistics


def normalize(attention):
    n = len(attention)
    if n < 2 or any(len(row) != n for row in attention):
        raise ValueError('need a square matrix with at least two tokens')
    result = []
    for row in attention:
        if any(not math.isfinite(x) or x < 0 for x in row):
            raise ValueError('entries must be finite and nonnegative')
        total = math.fsum(row)
        if not math.isfinite(total) or total <= 0:
            raise ValueError('each row needs a finite positive sum')
        result.append([x / total for x in row])
    return result


def dissimilarities(attention):
    p = normalize(attention)
    n = len(p)
    return [[0.0 if i == j else max(0.0, min(1.0, 1 - (p[i][j] + p[j][i]) / 2))
             for j in range(n)] for i in range(n)]


def merge_values(attention):
    """Kruskal MST weights = finite H0 deaths; omit the essential component."""
    d = dissimilarities(attention)
    n = len(d)
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    deaths = []
    for weight, i, j in sorted((d[i][j], i, j) for i in range(n) for j in range(i)):
        a, b = root(i), root(j)
        if a != b:
            parent[a] = b
            deaths.append(weight)
            if len(deaths) == n - 1:
                break
    return deaths


def features(attention):
    deaths = merge_values(attention)
    total = math.fsum(deaths)
    def quantile(q):
        position = (len(deaths) - 1) * q
        lo = math.floor(position)
        hi = math.ceil(position)
        return deaths[lo] + (position - lo) * (deaths[hi] - deaths[lo])
    return dict(mean=statistics.mean(deaths), std=statistics.pstdev(deaths),
                q10=quantile(.1), q25=quantile(.25), q50=quantile(.5),
                q75=quantile(.75), q90=quantile(.9), maximum=max(deaths),
                total=total, entropy=-math.fsum((x / total) * math.log(x / total)
                    for x in deaths if x > 0) if total else 0.0)


def readout(attention, values):
    p = normalize(attention)
    if len(values) != len(p) or any(not math.isfinite(v) for v in values):
        raise ValueError('one finite scalar value per token required')
    return [math.fsum(a * v for a, v in zip(row, values)) for row in p]


def fixture():
    base = [[.5, .125, .125, .25], [.125, .5625, .25, .0625],
            [.125, .25, .5, .125], [.25, .0625, .125, .5625]]
    circulated = [row[:] for row in base]
    for i in range(4):
        j = (i + 1) % 4
        circulated[i][j] += .0625
        circulated[j][i] -= .0625
    return base, circulated


def run():
    base, circulated = fixture()
    f = features(base)
    return {'scope': 'original nonuniform four-token diagnostic; no trained model',
            'finite_merge_values': merge_values(base), 'features': f,
            'identical_dissimilarities': dissimilarities(base) == dissimilarities(circulated),
            'identical_features': f == features(circulated),
            'values': [0, 1, 2, 3], 'base_AV': readout(base, [0, 1, 2, 3]),
            'circulated_AV': readout(circulated, [0, 1, 2, 3]),
            'recovered_token_count': 1 + f['total'] / f['mean']}


def main():
    def rounded(x):
        if isinstance(x, float):
            return round(x, 12)
        if isinstance(x, list):
            return [rounded(v) for v in x]
        if isinstance(x, dict):
            return {k: rounded(v) for k, v in x.items()}
        return x
    print(json.dumps(rounded(run()), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
