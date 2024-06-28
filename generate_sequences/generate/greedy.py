import torch

from generate_sequences.generate.base import BaseGenerator


class GreedyGenerator(BaseGenerator):
    def _encoder_decoder_generate(self, encoder_inputs):
        outputs = []
        for encoder_inputs_batch in self.get_batches(encoder_inputs):
            batch_size = len(encoder_inputs_batch)
            decoder_inputs_batch = torch.full(
                (batch_size, self.max_length),
                self.eos_token_id,  # Pre-fill with EOS; only overwrite if generating
                dtype=torch.long,
                device=self.device,
            )
            decoder_inputs_batch[:, 0] = self.decoder_start_token_id
            finished_sequences_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            for step in range(1, self.max_length):
                if finished_sequences_mask.all():
                    break  # Stop if all sequences are finished
                batch_outputs = self.generation_forward(
                    encoder_inputs_batch,
                    decoder_inputs_batch[:, :step],
                )
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
