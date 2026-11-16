"""A constructed diagnostic: average coverage is not worst-query coverage."""
import json
from .core import attend


def run():
    queries = [[0.0]] * 99 + [[-8.0]]
    keys, values, bias = [[1.0], [-1.0]], [[1.0], [-1.0]], [4.0, 0.0]
    full = attend(queries, keys, values, bias)
    pruned = attend(queries, keys, values, bias, 1)
    all_kept = attend(queries, keys, values, bias, 2)
    mass = [sum(row[j] for j in pruned['indices']) for row in full['weights']]
    error = [abs(x[0]-y[0]) for x, y in zip(full['outputs'], pruned['outputs'])]
    constant_full = attend(queries, keys, [[1.0], [1.0]], bias)
    constant_pruned = attend(queries, keys, [[1.0], [1.0]], bias, 1)
    return {
        'inputs': {'query_counts': {'0': 99, '-8': 1}, 'keys': keys,
                   'values': values, 'bias': bias, 'budget': 1},
        'kept_indices': pruned['indices'],
        'mean_retained_mass': round(sum(mass)/len(mass), 9),
        'min_retained_mass': round(min(mass), 9),
        'common_query': {'dense_output': round(full['outputs'][0][0], 9),
                         'pruned_output': pruned['outputs'][0][0]},
        'rare_query': {'dense_output': round(full['outputs'][-1][0], 9),
                       'pruned_output': pruned['outputs'][-1][0]},
        'max_absolute_output_error': round(max(error), 9),
        'all_keys_max_error': max(abs(x[0]-y[0]) for x,y in zip(full['outputs'],all_kept['outputs'])),
        'constant_values_max_error': round(max(abs(x[0]-y[0]) for x,y in zip(constant_full['outputs'],constant_pruned['outputs'])), 9),
    }

if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))
