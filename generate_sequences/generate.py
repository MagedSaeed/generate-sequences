import heapq
from typing import Callable, Iterator, List, Optional, Union

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
        sort_encoder_inputs: bool = False,
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
        self.sort_encoder_inputs = sort_encoder_inputs

    def get_batches(
        self, inputs: Union[torch.Tensor, List[torch.Tensor], List[str]]
    ) -> Iterator[Union[torch.Tensor, List[torch.Tensor], List[str]]]:
        batched_inputs = inputs
        if self.sort_encoder_inputs:
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
        if not self.sort_encoder_inputs:
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
            logits = logits.gather(-1, next_tokens)
        else:
            logits, next_tokens = torch.topk(logits, num_tokens)
        return logits, next_tokens


class GreedyGenerator(BaseGenerator):
    def _encoder_decoder_generate(self, encoder_inputs):
        outputs = []
        for encoder_inputs_batch in self.get_batches(encoder_inputs):
            batch_size = len(encoder_inputs_batch)
            decoder_inputs = torch.full(
                (batch_size, self.max_length),
                self.eos_token_id,  # Pre-fill with EOS; only overwrite if generating
                dtype=torch.long,
                device=self.device,
            )
            decoder_inputs[:, 0] = self.decoder_start_token_id
            finished_sequences_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            for step in range(1, self.max_length):
                if finished_sequences_mask.all():
                    break  # Stop if all sequences are finished
                batch_outputs = self.generation_forward(
                    encoder_inputs_batch,
                    decoder_inputs[:, :step],
                )
                logits = batch_outputs[:, -1, :]
                _, next_tokens = self.sample_next_tokens(logits)
                next_tokens = next_tokens.squeeze(-1)
                unfinished_sequences_mask = ~finished_sequences_mask
                decoder_inputs[unfinished_sequences_mask, step] = next_tokens[
                    unfinished_sequences_mask
                ]
                finished_sequences_mask |= (
                    next_tokens.squeeze() == self.eos_token_id
                )  # Update finished sequences
            outputs += decoder_inputs
        return outputs

    def _decoder_only_generate(self, decoder_inputs: torch.Tensor):
        outputs = []
        for decoder_inputs_batch in self.get_batches(decoder_inputs):
            batch_size = len(decoder_inputs_batch)
            start_decoding_from = decoder_inputs_batch.shape[-1]  # type: ignore
            # extend the current batch of decoder inputs with eos until max_length to be of size [batch_size, max_length]
            decoder_inputs_batch = torch.cat(
                (
                    decoder_inputs_batch,
                    torch.full(
                        (
                            batch_size,
                            self.max_length - decoder_inputs_batch.size(1),  # type: ignore
                        ),
                        self.eos_token_id,
                        device=self.device,
                    ),
                ),
                dim=-1,
            )
            finished_sequences_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            for step in range(start_decoding_from, self.max_length):
                if finished_sequences_mask.all():
                    break  # Stop if all sequences are finished
                batch_outputs = self.generation_forward(None, decoder_inputs_batch[:, :step])
                logits = batch_outputs[:, -1, :]
                _, next_tokens = self.sample_next_tokens(logits)
                next_tokens = next_tokens.squeeze(-1)
                unfinished_sequences_mask = ~finished_sequences_mask
                decoder_inputs_batch[unfinished_sequences_mask, step] = next_tokens[
                    unfinished_sequences_mask
                ]
                finished_sequences_mask |= (
                    next_tokens.squeeze() == self.eos_token_id
                )  # Update finished sequences
            outputs += decoder_inputs_batch
        return outputs

    @torch.no_grad()
    def generate(
        self,
        encoder_inputs: Union[torch.Tensor, List[torch.Tensor], List[List[int]], None],
        decoder_inputs: Union[torch.Tensor, List[torch.Tensor], None] = None,
        pad_decoder_inputs: Optional[int] = None,
        decoder_inputs_padding_side: Optional[str] = "left",
    ) -> List[torch.Tensor]:
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
        outputs: List[torch.Tensor] = []
        if encoder_inputs:
            outputs = self._encoder_decoder_generate(encoder_inputs)
        else:
            outputs = self._decoder_only_generate(decoder_inputs)
        return self.restore_outputs_order(outputs)


class BeamNode:
    """Represents a node in a beam search. Stores token sequences and their associated score."""

    def __init__(self, tokens: List[int], score: float) -> None:
        self.tokens = tokens
        self.score = score


def default_beam_nodes_ordering_fn(
    node: BeamNode,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculates the adjusted score of a node for beam sorting. Applies length penalty to score."""
    tokens = node.tokens
    if eos_token_id in tokens:
        tokens = tokens[1 : tokens.index(eos_token_id) + 1]
    return node.score / (len(tokens) ** length_penalty)


class BeamSearchGenerator(BaseGenerator):
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
        sort_encoder_inputs: bool = False,
        beam_width: int = 4,
        length_penalty: float = 1.0,
        beam_nodes_ordering_function: Callable[
            [BeamNode, int, float], float
        ] = default_beam_nodes_ordering_fn,
    ) -> None:
        super().__init__(
            decoder_start_token_id,
            eos_token_id,
            generation_forward,
            max_length,
            batch_size,
            device,
            temperature,
            use_tqdm,
            top_k_sampling,
            top_p_sampling,
            multinomial_sampling,
            sort_encoder_inputs,
        )
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.beam_nodes_ordering_function = beam_nodes_ordering_function

    def get_top_nodes(self, nodes) -> List[BeamNode]:
        """Returns the top k nodes in the beam according to the ordering function."""
        return heapq.nlargest(
            self.beam_width,
            nodes,
            key=lambda node: self.beam_nodes_ordering_function(
                node,
                self.eos_token_id,
                self.length_penalty,
            ),
        )

    @torch.no_grad
    def generate(self, encoder_inputs: Union[List[torch.Tensor], List[str]]) -> List[torch.Tensor]:
        outputs = []
        for batch in self.get_batches(encoder_inputs):
            batch_nodes = [
                [
                    BeamNode(
                        tokens=[self.decoder_start_token_id],
                        score=0.0,
                    )
                ]
                for _ in range(len(batch))
            ]
            batch_best_nodes = batch_nodes
            for step in range(self.max_length):
                next_nodes: List[List[BeamNode]] = [[] for _ in range(len(batch))]
                batch_best_nodes = [
                    self.get_top_nodes(sample_nodes) for sample_nodes in batch_best_nodes
                ]
                # break when all best nodes ends with eos
                if all(
                    batch_best_nodes[sample_index][i].tokens[-1] == self.eos_token_id
                    for sample_index in range(len(batch))
                    for i in range(len(batch_best_nodes[sample_index]))
                ):
                    break
                # beam width, taking the case where k < len(best_beams_nodes[0]), i.e. in the first step
                beam_width = 1 if step == 0 else self.beam_width
                for k in range(beam_width):
                    decoder_input_ids = torch.LongTensor(
                        [sample_best_nodes[k].tokens for sample_best_nodes in batch_best_nodes]
                    ).to(self.device)
                    batch_outputs = self.generation_forward(batch, decoder_input_ids)
                    logits = batch_outputs[:, -1, :]
                    logits, next_tokens = self.sample_next_tokens(
                        logits, num_tokens=self.beam_width
                    )
                    for sample_index in range(len(batch)):
                        if batch_best_nodes[sample_index][k].tokens[-1] == self.eos_token_id:
                            next_nodes[sample_index] += [
                                BeamNode(
                                    tokens=batch_best_nodes[sample_index][k].tokens
                                    + [self.eos_token_id],
                                    score=0,
                                )
                            ] * self.beam_width
                        else:
                            next_nodes[sample_index] += [
                                BeamNode(
                                    tokens=batch_best_nodes[sample_index][k].tokens
                                    + [next_tokens[sample_index][i].item()],
                                    score=batch_best_nodes[sample_index][k].score
                                    + logits[sample_index][i].item(),
                                )
                                for i in range(self.beam_width)
                            ]
                batch_best_nodes = next_nodes  # Update beams for the next time step

            batch_predictions = []
            for sample_nodes in batch_best_nodes:
                best_node = max(
                    sample_nodes,
                    key=lambda node: self.beam_nodes_ordering_function(
                        node,
                        self.eos_token_id,
                        self.length_penalty,
                    ),
                )
                batch_predictions.append(best_node.tokens)
            outputs += batch_predictions
        return self.restore_outputs_order(outputs)
