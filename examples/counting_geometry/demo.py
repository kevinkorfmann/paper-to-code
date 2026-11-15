"""Deterministic supplied-coordinate experiment."""
import json
from .core import classify, count_coordinate, prototypes, quantize, regression


def run():
    rows = []
    for limit in (16, 64, 256):
        centers = prototypes(limit)
        rows.append(dict(limit=limit, classes=len(centers),
                         max_head_weight=round(2*centers[-1],9),
                         min_exact_formula_margin=round((centers[-1]-centers[-2])**2,12),
                         float_correct=sum(classify(x,centers)[0]==n for n,x in enumerate(centers)),
                         grid_8bit_correct=sum(classify(quantize(x,8),centers)[0]==n for n,x in enumerate(centers))))
    centers = prototypes(256)
    return dict(rows=rows, collision=dict(counts=[255,256], grid_coordinates=[quantize(centers[i],8) for i in (255,256)], predictions=[classify(quantize(centers[i],8),centers)[0] for i in (255,256)]),
                outside_class_range=dict(true_count=300,prediction=classify(count_coordinate(300),centers)[0]),
                distractor_collision=dict(inputs=[[1,0],[2,2]], coordinates=[count_coordinate(1,2,0),count_coordinate(2,2,2)]),
                regression=dict(limit=64,gamma=8192,prediction=round(regression(64,8192),9),error=round(64-regression(64,8192),9)))

if __name__ == '__main__':
    print(json.dumps(run(),indent=2,sort_keys=True))
