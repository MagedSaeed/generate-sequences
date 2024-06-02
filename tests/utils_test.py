import unittest

import torch

from generate_sequences.utils import pad_tensors_list, sort_list_with_positions


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


class TestPadTensorsList(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_pad_tensors_list_left(self):
        tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
        pad_with = 0
        padded_tensors = pad_tensors_list(
            tensors, pad_with, padding_side="left", device=self.device
        )
        expected_output = torch.tensor([[1, 2, 3], [0, 4, 5], [0, 0, 6]]).to(self.device)
        self.assertTrue(
            torch.equal(padded_tensors, expected_output),
            f"Left padding failed. Expected: {expected_output}, but got: {padded_tensors}",
        )

    def test_pad_tensors_list_right(self):
        tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
        pad_with = 0
        padded_tensors = pad_tensors_list(
            tensors, pad_with, padding_side="right", device=self.device
        )
        expected_output = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]]).to(self.device)
        self.assertTrue(
            torch.equal(padded_tensors, expected_output),
            f"Right padding failed. Expected: {expected_output}, but got: {padded_tensors}",
        )

    def test_pad_tensors_list_different_pad_value(self):
        tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
        pad_with = -1
        padded_tensors = pad_tensors_list(
            tensors, pad_with, padding_side="left", device=self.device
        )
        expected_output = torch.tensor([[1, 2, 3], [-1, 4, 5], [-1, -1, 6]]).to(self.device)
        self.assertTrue(
            torch.equal(padded_tensors, expected_output),
            f"Padding with different pad_with value failed. Expected: {expected_output}, but got: {padded_tensors}",
        )

    def test_pad_tensors_list_max_length(self):
        tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
        pad_with = -1
        padded_tensors = pad_tensors_list(
            tensors, pad_with, padding_side="left", max_length=6, device=self.device
        )
        expected_output = torch.tensor(
            [[-1, -1, -1, 1, 2, 3], [-1, -1, -1, -1, 4, 5], [-1, -1, -1, -1, -1, 6]]
        ).to(self.device)
        self.assertTrue(
            torch.equal(padded_tensors, expected_output),
            f"Padding with specified max_length failed. Expected: {expected_output}, but got: {padded_tensors}",
        )
