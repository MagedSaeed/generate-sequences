import json
from typing import Dict

import datasets
import evaluate
import torch
from transformers import MarianMTModel, MarianTokenizer

from generate_sequences import BeamSearchGenerator, GreedyGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


# Custom generation_forward function for the model above.
def generation_forward(inputs, decoder_input_ids):
    global encoder_outputs
    tokenizer_results = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)
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
    generation_forward=generation_forward,
    eos_token_id=model.generation_config.eos_token_id,
    decoder_start_token_id=model.generation_config.decoder_start_token_id,
)


# Initialize BeamSearchGenerator
beam_search_sequences_generator = BeamSearchGenerator(
    use_tqdm=True,
    length_penalty=0.6,
    device=model.device,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
    generation_forward=generation_forward,
    eos_token_id=model.generation_config.eos_token_id,
    decoder_start_token_id=model.generation_config.decoder_start_token_id,
)


# Test greeyd generation method
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


# Test greedy generation with multinomial sampling
# def test_greedy_generate_with_multinomail_sampling():
#     prev_state = greedy_sequences_generator.multinomial_sampling
#     greedy_sequences_generator.multinomial_sampling = True
#     generated_sequences = tokenizer.batch_decode(
#         greedy_sequences_generator.generate(input_texts),
#         skip_special_tokens=True,
#     )
#     greedy_sequences_generator.multinomial_sampling = prev_state
#     # we do not have control on the final results of the sequences with multinomial sampling
#     # each time they come with different varying outcomes
#     # we endup checking if it just get come bleu score
#     assert (
#         0
#         < bleu_scorer.compute(
#             predictions=generated_sequences,
#             references=targets,
#         )["score"]
#     )


# Test beam search generation
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


# Test beam search generation with temperature
def test_beam_search_generate_with_temperature():
    prev_temp = beam_search_sequences_generator.temperature
    beam_search_sequences_generator.temperature = 0.75
    generated_sequences = tokenizer.batch_decode(
        beam_search_sequences_generator.generate(input_texts),
        skip_special_tokens=True,
    )
    beam_search_sequences_generator.temperature = prev_temp
    assert (
        14
        < bleu_scorer.compute(
            predictions=generated_sequences,
            references=targets,
        )["score"]
        < 15
    )
