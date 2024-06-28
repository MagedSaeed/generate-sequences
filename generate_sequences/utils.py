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


def pad_tensors_list(
    tensors,
    pad_with,
    padding_side="left",
    max_length=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    if max_length is None:
        max_length = max([prompt.shape[-1] for prompt in tensors])
    if padding_side == "left":
        tensors = [
            torch.cat(
                [
                    torch.full(
                        (max_length - decoder_input.shape[-1],),
                        pad_with,
                        device=device,
                        dtype=torch.long,
                    ),
                    torch.tensor(decoder_input, device=device, dtype=torch.long),
                ],
                dim=-1,
            )
            for decoder_input in tensors
        ]
    elif padding_side == "right":
        tensors = [
            torch.cat(
                [
                    torch.tensor(decoder_input, device=device, dtype=torch.long),
                    torch.full(
                        (max_length - decoder_input.shape[-1],),
                        pad_with,
                        device=device,
                        dtype=torch.long,
                    ),
                ],
                dim=-1,
            )
            for decoder_input in tensors
        ]
    else:
        raise ValueError(f"Unknown padding side: {padding_side}. Should be 'left' or 'right'.")
    tensors = torch.stack(tensors)
    return tensors
