"""Transformer-FSQ forward scalar coding and linear value pooling."""
import json
import math

LEVELS = 16
OFFSET = LEVELS // 2


def quantize(u):
    if not math.isfinite(u):
        raise ValueError("expected finite latent")
    h, delta = (LEVELS - 1) * 1.001 / 2, 0.5
    return round(math.tanh(u + math.atanh(delta / h)) * h - delta)


def pack(levels):
    if not levels or any(type(x) is not int or not -8 <= x <= 7 for x in levels):
        raise ValueError("expected nonempty signed 4-bit levels")
    out = bytearray((len(levels) + 1) // 2)
    for i, x in enumerate(levels):
        out[i // 2] |= (x + OFFSET) << (4 * (i % 2))
    return bytes(out)


def unpack(payload, width):
    if type(width) is not int or width < 1 or len(payload) != (width + 1) // 2:
        raise ValueError("invalid latent width or byte count")
    if width % 2 and payload[-1] >> 4:
        raise ValueError("nonzero padding")
    return tuple(((payload[i // 2] >> (4 * (i % 2))) & 15) - OFFSET for i in range(width))


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def project(matrix, x):
    return tuple(dot(row, x) for row in matrix)


def weights(query, latents, key_matrix):
    # A single query reads its supplied visible prefix; no positional encoding.
    scale = math.sqrt(len(query))
    scores = [dot(query, project(key_matrix, z)) / scale for z in latents]
    peak = max(scores)
    exp_scores = [math.exp(s - peak) for s in scores]
    total = math.fsum(exp_scores)
    return tuple(x / total for x in exp_scores)


def mix(rows, a):
    return tuple(dot(a, column) for column in zip(*rows, strict=True))


def read(query, latents, key_matrix, value_matrix, pool_first):
    a = weights(query, latents, key_matrix)
    return project(value_matrix, mix(latents, a)) if pool_first else mix(
        [project(value_matrix, z) for z in latents], a
    )


def read_packed(query, payloads, width, key_matrix, value_matrix):
    latents = [unpack(payload, width) for payload in payloads]
    return read(query, latents, key_matrix, value_matrix, pool_first=True)


def demo():
    source = [(-.44, .26, .91), (.63, -.57, .07), (.02, .44, -.63)]
    continuous = [
        tuple(math.tanh(u + math.atanh(.5 / 7.5075)) * 7.5075 - .5 for u in x)
        for x in source
    ]
    levels = [tuple(quantize(u) for u in x) for x in source]
    packed = [pack(x) for x in levels]
    q = (.7, -.2)
    wk = ((.3, -.4, .2), (.7, .1, .5))
    wv = ((.5, 1., -.25), (1., -.75, .5))
    lhs = read(q, levels, wk, wv, pool_first=False)
    rhs = read_packed(q, packed, 3, wk, wv)
    ref = read(q, continuous, wk, wv, pool_first=False)
    return {
        "signed_levels": levels, "packed_tokens_hex": [b.hex() for b in packed],
        "packed_payload_bytes": sum(map(len, packed)),
        "illustrative_fp16_kv_payload_bytes": len(source) * 2 * (len(wk) + len(wv)),
        "pool_then_decode": [round(x, 12) for x in rhs],
        "decode_then_pool": [round(x, 12) for x in lhs],
        "continuous_bounded_same_decoders": [round(x, 12) for x in ref],
        "quantization_max_abs_error": round(max(abs(x-y) for x,y in zip(rhs, ref)), 12),
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
