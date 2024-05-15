import torch


class Sampler:
    def __init__(
        self,
        multinomial_sampling=False,
        top_k_sampling=0,
    ):
        self.multinomial_sampling = multinomial_sampling
        self.top_k_sampling = top_k_sampling
        if not self.multinomial_sampling and self.top_k_sampling > 0:
            raise ValueError("Top-k sampling requires multinomial sampling.")

    def sample(self, logits, max_tokens=1):
        if 0 < self.top_k_sampling < max_tokens:
            raise ValueError("Top-k sampling should be larger than or equal to max_tokens")
        sampled_indices = torch.topk(logits, k=len(logits[-1]), dim=-1)
        if self.top_k_sampling > 0:
            logits, sampled_indices = torch.topk(
                logits,
                self.top_k_sampling,
                dim=-1,
            )
        if self.multinomial_sampling:
            sampled_indices = torch.multinomial(
                logits,
                num_samples=max_tokens,
            ).squeeze()
        return logits, sampled_indices
