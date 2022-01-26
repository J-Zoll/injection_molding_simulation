import unittest
import preprocessing

class Test_calculate_edges(unittest.TestCase):
    def test_calculates_edges(self):
        node_positions = [
            [0,0,0],
            [1,0,0],
            [0,1,0]
        ]
        connection_range=1.0
        target_edges = list({
            (0, 1),
            (0, 2)
        })
        edges = preprocessing.calculate_edges(
            node_positions,
            connection_range
        )
        self.assertEqual(edges, target_edges)


class Test_encode_fill_state(unittest.TestCase):
    def test_encodes_fill_state(self):
        fill_states = [False, True]
        target_enc_fill_states = [[0.0, 1.0], [1.0, 0.0]]
        enc_fill_states = preprocessing.encode_fill_state(fill_states)
        self.assertEqual(enc_fill_states, target_enc_fill_states)


class Test_calculate_fill_states(unittest.TestCase):
    def test_calculates_fill_states(self):
        step_size = .2
        fill_times = [.0, .1, .2, .3]
        target_fill_states = [
            [True, False, False, False],
            [True, True, True, False],
            [True, True, True, True]
        ]
        fill_states = preprocessing.calculate_fill_states(
            step_size,
            fill_times
        )
        self.assertEqual(fill_states, target_fill_states)


if __name__ == "__main__":
    unittest.main()