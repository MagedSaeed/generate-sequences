import unittest

import torch

from generate_sequences.utils import sort_list_with_positions


class TestSortListWithPositions(unittest.TestCase):

    def test_sort_tensors_by_length(self):
        # Create a list of tensors to sort
        tensor_list = [
            torch.tensor([1, 2, 1]),
            torch.tensor([5]),
            torch.tensor([4, 6, 5, 1, 3]),
        ]

        # Sort the tensor list using the custom key function
        sorted_tensors, positions = sort_list_with_positions(tensor_list)

        # Check if the tensors are sorted correctly
        self.assertEqual(
            list(map(lambda tensor: [element for element in tensor], sorted_tensors)),
            [
                torch.tensor([4, 6, 5, 1, 3]).tolist(),
                torch.tensor([1, 2, 1]).tolist(),
                torch.tensor([5]).tolist(),
            ],
        )

        # Check the new positions after sorting
        self.assertEqual(positions, [1, 2, 0])
