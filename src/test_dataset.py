import unittest
from dataset import InjectionMoldingDataset
from config import Config


class TestInjectionMoldingDataset (unittest.TestCase):

    def test_processes(self):
        """Test whether processing runs without raising an Exception"""
        CONNECTION_RANGE = .003
        TIME_STEP_SIZE = 3
        dataset = InjectionMoldingDataset(
            Config.DATA_DIR,
            CONNECTION_RANGE,
            TIME_STEP_SIZE,
            skip_processing=False
        )
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
