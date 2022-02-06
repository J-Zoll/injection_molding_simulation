import os.path
import unittest
from dataset import InjectionMoldingDataset


class TestInjectionMoldingDataset (unittest.TestCase):

    def test_processes(self):
        """Test whether processing runs without raising an Exception"""
        CONNECTION_RANGE = .003
        TIME_STEP_SIZE = 3
        DATA_ROOT_DIR = os.path.abspath("/Users/jonas/Documents/Bachelorarbeit/injection_molding_simulation/data")
        dataset = InjectionMoldingDataset(
            DATA_ROOT_DIR,
            CONNECTION_RANGE,
            TIME_STEP_SIZE,
            skip_processing=False
        )
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
