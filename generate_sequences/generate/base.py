from typing import Callable, Iterable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from generate_sequences.utils import pad_tensors_list, sort_list_with_positions


class BaseGenerator:
    def __init__(
        self,
        decoder_start_token_id: int,
        eos_token_id: int,
        generation_forward: Callable[
            [
                Union[torch.Tensor, List[torch.Tensor], List[str], None],
                Union[torch.Tensor, List[torch.Tensor]],
            ],
            torch.Tensor,
        ],
        max_length: int = 1_024,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 1.0,
        use_tqdm: bool = True,
        top_k_sampling: int = 0,
        top_p_sampling: float = 0.0,
        multinomial_sampling: bool = False,
        sort_inputs_by_size: bool = False,
        return_logits: bool = False,  # New parameter
    ) -> None:
        self.device = device
        self.use_tqdm = use_tqdm
        self.max_length = max_length
        self.batch_size = batch_size
        self.generation_forward = generation_forward
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.temperature = temperature
        self.top_k_sampling = top_k_sampling
        self.top_p_sampling = top_p_sampling
        self.multinomial_sampling = multinomial_sampling
        self.sort_inputs_by_size = sort_inputs_by_size
        self.return_logits = return_logits  # New attribute

    def get_batches(
        self, inputs: Union[torch.Tensor, List[torch.Tensor], List[str]]
    ) -> Iterator[Union[torch.Tensor, List[torch.Tensor], List[str]]]:
        batched_inputs = inputs
        if self.sort_inputs_by_size:
            sorted_inputs, inputs_positions = sort_list_with_positions(inputs)
            self._inputs_original_positions = inputs_positions
            batched_inputs = sorted_inputs

        for i in tqdm(
            range(0, len(batched_inputs), self.batch_size),
            disable=not self.use_tqdm,
            desc="Generating Sequences",
            total=len(batched_inputs) // self.batch_size,
        ):
            yield batched_inputs[i : i + self.batch_size]

    def restore_outputs_order(self, outputs):
        if not self.sort_inputs_by_size:
            return outputs
        ordered_outputs = []
        for position in self._inputs_original_positions:
            ordered_outputs.append(outputs[position])
        return ordered_outputs

    def sample_next_tokens(self, logits, num_tokens=1, min_tokens_to_keep=2):
        logits = logits / self.temperature
        if self.top_k_sampling > 0:
            top_logits, _ = torch.topk(
                logits,
                min(self.top_k_sampling, logits.size(-1)),  # in case top_k_sampling > vocab
                dim=-1,
            )
            logits[logits < top_logits[:, [-1]]] = -float("Inf")
        if self.top_p_sampling > 0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p_sampling
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1,
                sorted_indices,
                sorted_indices_to_remove,
            )
            logits[indices_to_remove] = -float("Inf")
            # the above scatter is equivalent to something like:
            # for i in range(logits.size(0)):
            #     indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
            #     logits[i, indices_to_remove] = -float("Inf")
        logits = F.log_softmax(logits, dim=-1)
        if self.multinomial_sampling:
            next_tokens = torch.multinomial(
                torch.exp(logits),
                num_samples=num_tokens,
            )
            tokens_logits = logits.gather(-1, next_tokens)
        else:
            tokens_logits, next_tokens = torch.topk(logits, num_tokens)
        return tokens_logits, next_tokens

    @torch.no_grad()
    def generate(
        self,
        encoder_inputs: Union[torch.Tensor, Iterable, None] = None,
        decoder_inputs: Union[
            torch.Tensor,
            Iterable[torch.Tensor],
            Iterable[Iterable[int]],
            None,
        ] = None,
        pad_decoder_inputs: Optional[int] = None,
        decoder_inputs_padding_side: Optional[str] = "left",
    ) -> Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        # assert either encoder_inputs or decoder_inputs is not None
        assert (
            encoder_inputs is not None or decoder_inputs is not None
        ), "Either encoder_inputs or decoder_inputs must be provided"
        # raise user warning if both encoder_inputs and decoder_inputs are provided
        if encoder_inputs is not None and decoder_inputs is not None:
            raise UserWarning(
                "Both encoder_inputs and decoder_inputs are provided. Generally, this is not a standard practice unless you know what you are doing!"
            )
            # assert decoder_inputs is 2d tensors or list of 1d tensors or integers
        if decoder_inputs is not None:
            if isinstance(decoder_inputs, torch.Tensor):
                assert decoder_inputs.dim() == 2, "decoder_inputs must be a 2D tensor"
            elif isinstance(decoder_inputs, list):
                assert all(
                    isinstance(item, (torch.Tensor, list)) for item in decoder_inputs
                ), "decoder_inputs must be a list of 1D tensors or a list of lists of integers"
                if isinstance(decoder_inputs[0], torch.Tensor):
                    assert all(
                        tensor.dim() == 1 for tensor in decoder_inputs
                    ), "All items in decoder_inputs list must be 1D tensors"
                elif isinstance(decoder_inputs[0], list):
                    assert all(
                        isinstance(item, int) for sublist in decoder_inputs for item in sublist
                    ), "All items in decoder_inputs lists must be integers"
            else:
                raise TypeError(
                    "decoder_inputs must be either a 2D tensor, a list of 1D tensors, or a list of lists of integers"
                )

        if pad_decoder_inputs is not None and isinstance(decoder_inputs, list):
            decoder_inputs = pad_tensors_list(
                decoder_inputs,
                device=self.device,
                pad_with=pad_decoder_inputs,
                padding_side=decoder_inputs_padding_side,
            )
        outputs: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]] = []
        if encoder_inputs:
            outputs = self._encoder_decoder_generate(encoder_inputs)
        else:
            outputs = self._decoder_only_generate(decoder_inputs)
        return self.restore_outputs_order(outputs)

    def _encoder_decoder_generate(self, encoder_inputs):
        raise NotImplementedError

    def _decoder_only_generate(self, decoder_inputs):
        raise NotImplementedError
