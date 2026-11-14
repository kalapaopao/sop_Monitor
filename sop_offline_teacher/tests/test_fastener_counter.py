import unittest

from src.fastener_counter import CounterConfig, FastenerCounter, CounterState


class FastenerCounterTests(unittest.TestCase):
    def test_repeated_presses_count_once(self):
        c = FastenerCounter(CounterConfig(min_fastening_ticks=2))
        for sample in [
            {"timestamp": 0, "placement": .9},
            {"timestamp": 1, "tool": .9},
            {"timestamp": 2, "tool": .95},
            {"timestamp": 3, "tool": .85},
            {"timestamp": 4, "release": .9},
        ]:
            c.update(sample)
        self.assertEqual(c.count, 1)
        self.assertEqual(len(c.events), 1)

    def test_tool_without_new_placement_does_not_count(self):
        c = FastenerCounter()
        for _ in range(5):
            c.update({"tool": .99})
        self.assertEqual(c.count, 0)

    def test_long_occlusion_becomes_unknown(self):
        c = FastenerCounter(CounterConfig(unknown_timeout_ticks=2))
        c.update({"placement": .9, "visible": True})
        c.update({"visible": False})
        result = c.update({"visible": False})
        self.assertEqual(result["state"], CounterState.UNKNOWN.value)


if __name__ == "__main__":
    unittest.main()
