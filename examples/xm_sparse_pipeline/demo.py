"""Original timeline toy for resident-first sparse KV tile pipelining."""

import json


def schedule(tiles, compute, transfer, *, slots=2, mode="pipeline"):
    """Return timed (tile, start, finish) events; 'H' is resident, 'M' remote."""
    tiles = tuple(tiles)
    if mode not in {"barrier", "pipeline"}:
        raise ValueError("unknown mode")
    if any(type(t) is not int or t <= 0 for t in (compute, transfer, slots)):
        raise ValueError("stage costs and slots must be positive integers")
    if any(tile not in {"H", "M"} for tile in tiles):
        raise ValueError("unknown tile type")

    # One upload engine, one consumer. Each remote tile holds its buffer
    # from the beginning of transfer until the consumer finishes reading it.
    downloads, attention, releases = [], [], []
    miss_count = tiles.count("M")
    available_engine = 0
    available_consumer = transfer * miss_count if mode == "barrier" else 0
    for i, tile in enumerate(tiles):
        ready = 0
        if tile == "M":
            if mode == "pipeline" and len(releases) >= slots:
                reuse = releases[-slots]
            else:
                reuse = 0
            start = max(available_engine, reuse)
            ready = start + transfer
            downloads.append((i, start, ready))
            available_engine = ready
        start = max(available_consumer, ready)
        available_consumer = start + compute
        attention.append((i, start, available_consumer))
        if tile == "M":
            releases.append(available_consumer)
    return {
        "finish": available_consumer,
        "transfer": downloads,
        "attention": attention,
    }


def demo():
    return {
        "units": "abstract time units",
        "all_fetches_before_attention": schedule("HHMMM", 3, 4, mode="barrier"),
        "resident_first_one_slot": schedule("HHMMM", 3, 4, slots=1),
        "resident_first_two_slots": schedule("HHMMM", 3, 4),
        "remote_first_two_slots": schedule("MMMHH", 3, 4),
        "all_resident": schedule("HHHHH", 3, 4),
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
