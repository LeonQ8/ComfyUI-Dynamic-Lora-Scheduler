import sys
import unittest
from unittest.mock import MagicMock
import math
import os

# --- Mocking Infrastructure ---
class MockArray(list):
    def __init__(self, data):
        super().__init__(data)
        
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x + other for x in self])
        if isinstance(other, (list, MockArray)):
            return MockArray([x + y for x, y in zip(self, other)])
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
        
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x - other for x in self])
        if isinstance(other, (list, MockArray)):
            return MockArray([x - y for x, y in zip(self, other)])
        return NotImplemented
        
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([other - x for x in self])
        if isinstance(other, (list, MockArray)):
            return MockArray([y - x for x, y in zip(other, self)])
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x * other for x in self])
        if isinstance(other, (list, MockArray)):
            return MockArray([x * y for x, y in zip(self, other)])
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x / other for x in self])
        return NotImplemented
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([other / x for x in self])
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x ** other for x in self])
        return NotImplemented
        
    def __neg__(self):
        return MockArray([-x for x in self])
    
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            # Calculate length of the slice
            if step == 1 or step is None:
                width = stop - start
            else:
                width = len(range(start, stop, step))
            
            # If value is scalar, broadcast
            if isinstance(value, (int, float)):
                replacement = [float(value)] * width
                super().__setitem__(key, replacement)
            else:
                super().__setitem__(key, value)
        else:
            super().__setitem__(key, value)
        
    def tolist(self):
        return list(self)

class MockNumpy:
    def linspace(self, start, stop, num):
        if num <= 1: return MockArray([float(start)])
        step = (stop - start) / (num - 1)
        return MockArray([start + step * i for i in range(num)])
    
    def exp(self, x):
        if isinstance(x, (list, MockArray)):
            return MockArray([math.exp(v) for v in x])
        return math.exp(x)
        
    def interp(self, x, xp, fp):
        res = []
        for val in x:
            if val <= xp[0]: res.append(fp[0])
            elif val >= xp[-1]: res.append(fp[-1])
            else:
                for i in range(len(xp)-1):
                    if xp[i] <= val <= xp[i+1]:
                        # Avoid div by zero
                        denom = (xp[i+1] - xp[i])
                        if denom == 0:
                            t = 0
                        else:
                            t = (val - xp[i]) / denom
                        res.append(fp[i] * (1-t) + fp[i+1] * t)
                        break
        return MockArray(res)
        
    def clip(self, a, a_min, a_max):
        if isinstance(a, (list, MockArray)):
            return MockArray([max(a_min, min(v, a_max)) for v in a])
        return max(a_min, min(a, a_max))
        
    def zeros(self, num):
        return MockArray([0.0] * num)

# Inject Mocks
sys.modules['numpy'] = MockNumpy()
sys.modules['torch'] = MagicMock()

# --- Import Node ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nodes.adaptive_lora_scheduler import AdaptiveLoraScheduler

# --- Tests ---
class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = AdaptiveLoraScheduler()
        # Mock standard WANVIDLORA input (list of dicts)
        self.high = [{"name": "high", "strength": 1.0, "path": "path/high"}]
        self.low = [{"name": "low", "strength": 1.0, "path": "path/low"}]

    def test_run_output_structure(self):
        steps = 10
        # Signature: lora_high, lora_low, steps, end_step, start_step, blend_strategy, invert, preview_only, custom_curve=None, unique_id=None
        # Arguments must match this order or be keyword args.
        start = 0
        end = steps
        
        high_out, low_out = self.scheduler.run(
            self.high, self.low, steps, end, start, "linear", False, False
        )
        
        self.assertIsInstance(high_out, list)
        self.assertIsInstance(low_out, list)
        
        # Default (False): High->Low (1.0 -> 0.0)
        high_sched = high_out[0]["strength"]
        self.assertAlmostEqual(high_sched[0], 1.0)
        self.assertAlmostEqual(high_sched[-1], 0.0)

    def test_invert(self):
        steps = 10
        start = 0
        end = 10
        # Invert=True: Low->High (0.0 -> 1.0)
        high_out, _ = self.scheduler.run(
            self.high, self.low, steps, end, start, "linear", True, False # Invert=True
        )
        high_sched = high_out[0]["strength"]
        self.assertAlmostEqual(high_sched[0], 0.0)
        self.assertAlmostEqual(high_sched[-1], 1.0)

    def test_partial_range(self):
        steps = 20
        start = 5
        end = 15
        
        high_out, _ = self.scheduler.run(
            self.high, self.low, steps, end, start, "linear", False, False
        )
        curve = high_out[0]["strength"]
        
        # Before start (Default High->Low): High=1.0
        self.assertEqual(curve[0], 1.0)
        # After end: High=0.0
        self.assertEqual(curve[-1], 0.0)
    
    def test_strategies(self):
        steps = 10
        for strategy in ["linear", "sigmoid", "step"]:
            high_out, _ = self.scheduler.run(
                self.high, self.low, steps, steps, 0, strategy, False, False
            )
            self.assertEqual(len(high_out[0]["strength"]), steps)

    def test_console_output(self):
        self.scheduler.run(self.high, self.low, 10, 10, 0, "linear", False, False)

if __name__ == '__main__':
    unittest.main()

