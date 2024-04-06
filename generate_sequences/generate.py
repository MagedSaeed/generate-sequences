from typing import Callable, Iterator, List, Union

import torch
from tqdm.auto import tqdm


class BaseGenerator:
    def __init__(
        self,
        decoder_start_token_id: int,
        eos_token_id: int,
        generate_fn: Callable[
            [Union[List[torch.Tensor], List[str]], torch.Tensor],
            torch.Tensor,
        ],
        max_length: int = 1_024,
        batch_size: int = 1,
        device: str = "cuda",
        use_tqdm: bool = True,
    ) -> None:
        self.device = device
        self.use_tqdm = use_tqdm
        self.max_length = max_length
        self.batch_size = batch_size
        self.generate_fn = generate_fn
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def get_batches(
        self, inputs: Union[List[torch.Tensor], List[str]]
    ) -> Iterator[List[str]]:
        for i in tqdm(
            range(0, len(inputs), self.batch_size),
            disable=not self.use_tqdm,
            desc="Generating Sequences",
            total=len(inputs) // self.batch_size,
        ):
            yield inputs[i : i + self.batch_size]

    def get_next_tokens(
        self,
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class GreedyGenerator(BaseGenerator):
    def get_next_tokens(self, outputs: torch.Tensor) -> torch.Tensor:
        next_tokens = torch.argmax(outputs, dim=-1)
        return next_tokens

    @torch.no_grad()
    def generate(
        self, inputs: Union[List[torch.Tensor], List[str]]
    ) -> List[torch.Tensor]:
        outputs = []
        for batch_inputs in self.get_batches(inputs):
            batch_size = len(batch_inputs)
            decoder_inputs = torch.full(
                (batch_size, self.max_length),
                self.eos_token_id,  # Pre-fill with EOS; only overwrite if generating
                dtype=torch.long,
                device=self.device,
            )
            decoder_inputs[:, 0] = self.decoder_start_token_id
            finished_mask = torch.zeros(
                batch_size, dtype=torch.bool, device=self.device
            )

            for step in range(1, self.max_length):
                if finished_mask.all():
                    break  # Stop if all sequences are finished
                batch_outputs = self.generate_fn(batch_inputs, decoder_inputs[:, :step])
                batch_outputs = batch_outputs[
                    :, -1, :
                ]  # Get last tokens' outputs for the batch
                next_tokens = self.get_next_tokens(batch_outputs)
                not_finished = ~finished_mask
                decoder_inputs[not_finished, step] = next_tokens[not_finished]
                finished_mask |= (
                    next_tokens == self.eos_token_id
                )  # Update finished sequences
            outputs.extend(decoder_inputs)
        return outputs
