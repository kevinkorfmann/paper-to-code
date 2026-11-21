"""Independent reference and fixture checks for the timeline toy."""

import itertools
import unittest

from .demo import demo, schedule


def tick_until_finished(tiles, compute, transfer, slots):
    """Independent, unit-step simulation, without event-time recurrence."""
    pending = [i for i, tile in enumerate(tiles) if tile == "M"]
    t, next_attention = 0, 0
    lock, ready = set(), set()
    io = cpu = None
    while next_attention < len(tiles):
        if io is not None and io[1] == t:
            ready.add(io[0])
            io = None
        if cpu is not None and cpu[1] == t:
            lock.discard(cpu[0])
            next_attention += 1
            cpu = None
        if next_attention == len(tiles):
            break
        if io is None and pending and len(lock) < slots:
            tile = pending.pop(0)
            lock.add(tile)
            io = (tile, t + transfer)
        if cpu is None and (
            tiles[next_attention] == "H" or next_attention in ready
        ):
            cpu = (next_attention, t + compute)
        t += 1
    return t


class PipelineTests(unittest.TestCase):
    def test_fixture(self):
        answers = demo()
        expected = {
            "all_fetches_before_attention": 27,
            "resident_first_one_slot": 23,
            "resident_first_two_slots": 16,
            "remote_first_two_slots": 21,
            "all_resident": 15,
        }
        self.assertEqual({key: answers[key]["finish"] for key in expected}, expected)

    def test_timeline(self):
        result = schedule("HHMMM", 3, 4)
        self.assertEqual(result["transfer"], [(2, 0, 4), (3, 4, 8), (4, 9, 13)])
        self.assertEqual(result["attention"], [
            (0, 0, 3), (1, 3, 6), (2, 6, 9), (3, 9, 12), (4, 13, 16)
        ])

    def test_serial_control(self):
        for tiles in ("", "H", "M", "MH", "HHMMM"):
            self.assertEqual(schedule(tiles, 2, 7, mode="barrier")["finish"],
                             2 * len(tiles) + 7 * tiles.count("M"))

    def test_reference_exhaustively(self):
        for n in range(6):
            for configuration in itertools.product("HM", repeat=n):
                for compute, transfer, slots in itertools.product(range(1, 4), repeat=3):
                    args = (configuration, compute, transfer)
                    self.assertEqual(schedule(*args, slots=slots)["finish"],
                                     tick_until_finished(*args, slots))

    def test_invalid(self):
        for args in (("X", 3, 4), ("H", 0, 4), ("H", 3, -4), ("H", True, 4)):
            with self.assertRaises(ValueError):
                schedule(*args)
        with self.assertRaises(ValueError):
            schedule("HM", 3, 4, slots=0)
        with self.assertRaises(ValueError):
            schedule("HM", 3, 4, mode="mystery")


if __name__ == "__main__":
    unittest.main()
