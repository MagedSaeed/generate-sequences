import heapq
from typing import Callable, List, Tuple, Union

import torch

from generate_sequences.generate.base import BaseGenerator


class BeamNode:
    """Represents a node in a beam search. Stores token sequences and their associated score."""

    def __init__(
        self, tokens: List[int], score: float, logits: List[Tuple[int, float]] = None  # type: ignore
    ) -> None:
        self.tokens = tokens
        self.score = score
        self.logits = logits or []


def default_beam_nodes_ordering_fn(
    node: BeamNode,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculates the adjusted score of a node for beam sorting. Applies length penalty to score."""
    tokens = node.tokens
    if eos_token_id in tokens:
        # get last index of eos_token_id
        last_eos_index = len(tokens) - tokens[::-1].index(eos_token_id)
        tokens = tokens[: last_eos_index + 1]
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
        sort_inputs_by_size: bool = False,
        beam_width: int = 4,
        length_penalty: float = 1.0,
        beam_nodes_ordering_function: Callable[
            [BeamNode, int, float], float
        ] = default_beam_nodes_ordering_fn,
        return_logits: bool = False,
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
            sort_inputs_by_size,
            return_logits,
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

    def _encoder_decoder_generate(
        self,
        encoder_inputs: Union[List[torch.Tensor], List[str]],
    ) -> List[Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[int, float]]]]]:
        outputs = []
        for encoder_inputs_batch in self.get_batches(encoder_inputs):
            batch_nodes = [
                [
                    BeamNode(
                        tokens=[self.decoder_start_token_id],
                        score=0.0,
                        logits=[(self.decoder_start_token_id, 0.0)],
                    )
                ]
                for _ in range(len(encoder_inputs_batch))
            ]
            batch_best_nodes = batch_nodes
            for step in range(self.max_length):
                next_nodes: List[List[BeamNode]] = [[] for _ in range(len(encoder_inputs_batch))]
                batch_best_nodes = [
                    self.get_top_nodes(sample_nodes) for sample_nodes in batch_best_nodes
                ]
                # break when all best nodes ends with eos
                if all(
                    batch_best_nodes[sample_index][i].tokens[-1] == self.eos_token_id
                    for sample_index in range(len(encoder_inputs_batch))
                    for i in range(len(batch_best_nodes[sample_index]))
                ):
                    break
                # beam width, taking the case where k < len(best_beams_nodes[0]), i.e. in the first step
                beam_width = 1 if step == 0 else self.beam_width
                for k in range(beam_width):
                    decoder_input_ids = torch.LongTensor(
                        [sample_best_nodes[k].tokens for sample_best_nodes in batch_best_nodes]
                    ).to(self.device)
                    batch_outputs = self.generation_forward(encoder_inputs_batch, decoder_input_ids)
                    logits = batch_outputs[:, -1, :]
                    logits, next_tokens = self.sample_next_tokens(
                        logits,
                        num_tokens=self.beam_width,
                    )
                    for sample_index in range(len(encoder_inputs_batch)):
                        if batch_best_nodes[sample_index][k].tokens[-1] == self.eos_token_id:
                            next_nodes[sample_index] += [
                                BeamNode(
                                    tokens=batch_best_nodes[sample_index][k].tokens
                                    + [self.eos_token_id],
                                    score=0,
                                    logits=batch_best_nodes[sample_index][k].logits
                                    + [(self.eos_token_id, 0.0)],
                                )
                            ] * self.beam_width
                        else:
                            next_nodes[sample_index] += [
                                BeamNode(
                                    tokens=batch_best_nodes[sample_index][k].tokens
                                    + [next_tokens[sample_index][i].item()],
                                    score=batch_best_nodes[sample_index][k].score
                                    + logits[sample_index][i].item(),
                                    logits=batch_best_nodes[sample_index][k].logits
                                    + [
                                        (
                                            next_tokens[sample_index][i].item(),
                                            logits[sample_index][i].item(),
                                        )
                                    ],
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
                if self.return_logits:
                    batch_predictions.append((torch.tensor(best_node.tokens), best_node.logits))
                else:
                    batch_predictions.append(torch.tensor(best_node.tokens))
            outputs += batch_predictions
        return outputs

    def _decoder_only_generate(
        self,
        decoder_inputs: torch.Tensor,
    ) -> List[Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[int, float]]]]]:
        outputs = []
        for decoder_inputs_batch in self.get_batches(decoder_inputs):
            batch_nodes = [
                [
                    BeamNode(
                        tokens=[token.item() for token in sample],  # type: ignore
                        score=0.0,
                        logits=[(token.item(), 0.0) for token in sample],  # type: ignore
                    )
                ]
                for sample in decoder_inputs_batch
            ]
            batch_best_nodes = batch_nodes
            for step in range(self.max_length - decoder_inputs_batch.size(1)):  # type: ignore
                next_nodes: List[List[BeamNode]] = [[] for _ in range(len(decoder_inputs_batch))]
                batch_best_nodes = [
                    self.get_top_nodes(sample_nodes) for sample_nodes in batch_best_nodes
                ]
                # break when all best nodes ends with eos
                if all(
                    batch_best_nodes[sample_index][i].tokens[-1] == self.eos_token_id
                    for sample_index in range(len(decoder_inputs_batch))
                    for i in range(len(batch_best_nodes[sample_index]))
                ):
                    break
                # beam width, taking the case where k < len(best_beams_nodes[0]), i.e. in the first step
                beam_width = 1 if step == 0 else self.beam_width
                for k in range(beam_width):
                    decoder_input_ids = torch.LongTensor(
                        [sample_best_nodes[k].tokens for sample_best_nodes in batch_best_nodes]
                    ).to(self.device)
                    batch_outputs = self.generation_forward(None, decoder_input_ids)
                    logits = batch_outputs[:, -1, :]
                    next_token_logits, next_tokens = self.sample_next_tokens(
                        logits,
                        num_tokens=self.beam_width,
                    )
                    for sample_index in range(len(decoder_inputs_batch)):
                        if batch_best_nodes[sample_index][k].tokens[-1] == self.eos_token_id:
                            next_nodes[sample_index] += [
                                BeamNode(
                                    tokens=batch_best_nodes[sample_index][k].tokens
                                    + [self.eos_token_id],
                                    score=0,
                                    logits=batch_best_nodes[sample_index][k].logits
                                    + [(self.eos_token_id, 0.0)],
                                )
                            ] * self.beam_width
                        else:
                            next_nodes[sample_index] += [
                                BeamNode(
                                    tokens=batch_best_nodes[sample_index][k].tokens
                                    + [next_tokens[sample_index][i].item()],
                                    score=batch_best_nodes[sample_index][k].score
                                    + next_token_logits[sample_index][i].item(),
                                    logits=batch_best_nodes[sample_index][k].logits
                                    + [
                                        (
                                            next_tokens[sample_index][i].item(),
                                            next_token_logits[sample_index][i].item(),
                                        )
                                    ],
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
                if self.return_logits:
                    batch_predictions.append((torch.tensor(best_node.tokens), best_node.logits))
                else:
                    batch_predictions.append(torch.tensor(best_node.tokens))
            outputs += batch_predictions
        return outputs
