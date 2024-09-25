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

            if self.return_logits:
                logits_list = [
                    [(token.item(), 0.0) for token in seq] for seq in decoder_inputs_batch
                ]

            for step in range(1, self.max_length):
                if finished_sequences_mask.all():
                    break  # Stop if all sequences are finished
                batch_outputs = self.generation_forward(
                    encoder_inputs_batch,
                    decoder_inputs_batch[:, :step],
                )
                logits = batch_outputs[:, -1, :]
                next_token_logits, next_tokens = self.sample_next_tokens(logits)
                next_tokens = next_tokens.squeeze(-1)
                unfinished_sequences_mask = ~finished_sequences_mask
                decoder_inputs_batch[unfinished_sequences_mask, step] = next_tokens[
                    unfinished_sequences_mask
                ]

                # Update finished sequences
                newly_finished = next_tokens.squeeze() == self.eos_token_id
                finished_sequences_mask |= newly_finished

                if self.return_logits:
                    for i in range(batch_size):
                        if not finished_sequences_mask[i]:
                            logits_list[i][step] = (
                                next_tokens[i].item(),
                                next_token_logits[i].item(),
                            )

            if self.return_logits:
                outputs += list(zip(decoder_inputs_batch, logits_list))
            else:
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

            if self.return_logits:
                logits_list = [
                    [(token.item(), 0.0) for token in seq] for seq in decoder_inputs_batch
                ]

            for step in range(start_decoding_from, self.max_length):
                if finished_sequences_mask.all():
                    break  # Stop if all sequences are finished
                batch_outputs = self.generation_forward(None, decoder_inputs_batch[:, :step])
                logits = batch_outputs[:, -1, :]
                next_token_logits, next_tokens = self.sample_next_tokens(logits)
                next_tokens = next_tokens.squeeze(-1)
                unfinished_sequences_mask = ~finished_sequences_mask
                decoder_inputs_batch[unfinished_sequences_mask, step] = next_tokens[
                    unfinished_sequences_mask
                ]

                # Update finished sequences
                newly_finished = next_tokens.squeeze() == self.eos_token_id
                finished_sequences_mask |= newly_finished

                if self.return_logits:
                    for i in range(batch_size):
                        if not finished_sequences_mask[i]:
                            logits_list[i][step] = (
                                next_tokens[i].item(),
                                next_token_logits[i].item(),
                            )

            if self.return_logits:
                outputs += list(zip(decoder_inputs_batch, logits_list))
            else:
                outputs += decoder_inputs_batch
        return outputs
