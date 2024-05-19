from typing import Any, Callable, List, Tuple, Union

import torch


def sort_list_with_positions(
    elements: List[Union[torch.Tensor, str]],
    key: Callable[[Union[torch.Tensor, str]], Any] = len,
    reverse: bool = True,
) -> Tuple[List[Union[torch.Tensor, str]], List[int]]:
    # Pair each element with its original index
    indexed_lst = list(enumerate(elements))

    # Sort the list based on the provided key or default sorting
    sorted_lst = sorted(
        indexed_lst,
        key=lambda indexed_element: key(indexed_element[1]),
        reverse=reverse,
    )

    # Extract the sorted elements
    sorted_items = [item for _, item in sorted_lst]

    # Create a list of new positions for each original index, starting from 0
    new_positions = [0] * len(elements)
    for new_idx, (orig_idx, _) in enumerate(sorted_lst):
        new_positions[orig_idx] = new_idx

    return sorted_items, new_positions
