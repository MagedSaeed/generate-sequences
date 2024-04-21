import json
from typing import Dict

import datasets
import evaluate
import torch
from transformers import MarianMTModel, MarianTokenizer

from generate_sequences import BeamSearchGenerator, GreedyGenerator

DEVICE = "cpu"
BATCH_SIZE = 2
MAX_LENGTH = 50


model_name = "marefa-nlp/marefa-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(DEVICE)


bleu_scorer = evaluate.load("sacrebleu")


test_dataset = datasets.load_dataset("iwslt2017", "iwslt2017-ar-en", split="test")

source_language = "en"
target_language = "ar"


input_texts = [example[source_language] for example in test_dataset["translation"]][-10:]
targets = [example[target_language] for example in test_dataset["translation"]][-10:]


encoder_outputs: Dict[str, torch.Tensor] = {}


def hf_generate_fn(inputs, decoder_input_ids):
    global encoder_outputs
    tokenizer_results = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
    )
    if not encoder_outputs.get(json.dumps(inputs)):
        input_ids, attention_mask = (
            tokenizer_results["input_ids"],
            tokenizer_results["attention_mask"],
        )
        encoder_outputs[json.dumps(inputs)] = model.get_encoder()(
            input_ids,
            return_dict=True,
            attention_mask=attention_mask,
        )
    model_outputs = model(
        **tokenizer_results,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs[json.dumps(inputs)],
    )
    return model_outputs.logits


# Initialize GreedyGenerator
greedy_sequences_generator = GreedyGenerator(
    use_tqdm=True,
    device=model.device,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    generate_fn=hf_generate_fn,
    eos_token_id=model.generation_config.eos_token_id,
    decoder_start_token_id=model.generation_config.decoder_start_token_id,
)


# Initialize BeamSearchGenerator
beam_search_sequences_generator = BeamSearchGenerator(
    use_tqdm=True,
    device=model.device,
    max_length=MAX_LENGTH,
    length_penalty_alpha=0.6,
    generate_fn=hf_generate_fn,
    minimum_penalty_tokens_length=5,
    batch_size=BATCH_SIZE,
    eos_token_id=model.generation_config.eos_token_id,
    decoder_start_token_id=model.generation_config.decoder_start_token_id,
)


# Test generate method
def test_greedy_generate():
    generated_sequences = tokenizer.batch_decode(
        greedy_sequences_generator.generate(input_texts),
        skip_special_tokens=True,
    )
    assert (
        7
        < bleu_scorer.compute(
            predictions=generated_sequences,
            references=targets,
        )["score"]
        < 8
    )


def test_beam_search_generate():
    generated_sequences = tokenizer.batch_decode(
        beam_search_sequences_generator.generate(input_texts),
        skip_special_tokens=True,
    )
    assert (
        14
        < bleu_scorer.compute(
            predictions=generated_sequences,
            references=targets,
        )["score"]
        < 15
    )
