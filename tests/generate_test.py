import json
from typing import Dict

import datasets
import evaluate
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, MarianMTModel, MarianTokenizer

from generate_sequences import BeamSearchGenerator, GreedyGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
MAX_LENGTH = 50

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

model_name = "marefa-nlp/marefa-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(DEVICE)

bleu_scorer = evaluate.load("sacrebleu")

test_dataset = datasets.load_dataset(
    "iwslt2017", "iwslt2017-ar-en", split="test", trust_remote_code=True
)

source_language = "en"
target_language = "ar"

input_texts = [example[source_language] for example in test_dataset["translation"]][-10:]
targets = [example[target_language] for example in test_dataset["translation"]][-10:]

encoder_outputs: Dict[str, torch.Tensor] = {}


# Custom generation_forward function for the model above.
def encoder_decoder_generation_forward(inputs, decoder_input_ids):
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


# Custom generation_forward function for the GPT-2 model (decoder-only).
def decoder_only_generation_forward(inputs, decoder_input_ids):
    input_ids = gpt2_tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).input_ids.to(DEVICE)
    model_outputs = gpt2_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    return model_outputs.logits


# Initialize GreedyGenerator
greedy_generator = GreedyGenerator(
    use_tqdm=True,
    device=model.device,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    generation_forward=encoder_decoder_generation_forward,
    eos_token_id=model.generation_config.eos_token_id,
    decoder_start_token_id=model.generation_config.decoder_start_token_id,
)


# Initialize BeamSearchGenerator
beam_search_generator = BeamSearchGenerator(
    use_tqdm=True,
    length_penalty=0.6,
    device=model.device,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
    generation_forward=encoder_decoder_generation_forward,
    eos_token_id=model.generation_config.eos_token_id,
    decoder_start_token_id=model.generation_config.decoder_start_token_id,
)


# Test greedy generation method
def test_greedy_generate():
    generated_sequences = tokenizer.batch_decode(
        greedy_generator.generate(input_texts),
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


def test_greedy_generate_with_sorted_samples():
    greedy_generator.sort_inputs_by_size = True
    generated_sequences = tokenizer.batch_decode(
        greedy_generator.generate(input_texts),
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
def test_greedy_generate_with_sampling():
    prev_sampling = greedy_generator.multinomial_sampling
    prev_temp = greedy_generator.temperature
    greedy_generator.multinomial_sampling = True
    greedy_generator.temperature = 0.95
    greedy_generator.top_p_sampling = 0.9
    greedy_generator.top_k_sampling = 10
    generated_sequences = tokenizer.batch_decode(
        greedy_generator.generate(input_texts),
        skip_special_tokens=True,
    )
    greedy_generator.multinomial_sampling = prev_sampling
    greedy_generator.temperature = prev_temp
    # we do not have control on the final results of the sequences with multinomial sampling
    # each time they come with different varying outcomes
    # we end up checking if it just get come bleu score
    assert (
        0
        < bleu_scorer.compute(
            predictions=generated_sequences,
            references=targets,
        )["score"]
    )


# Test beam search generation
def test_beam_search_generate():
    generated_sequences = tokenizer.batch_decode(
        beam_search_generator.generate(input_texts),
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
def test_beam_search_generate_with_sampling():
    prev_temp = beam_search_generator.temperature
    prev_sampling = beam_search_generator.multinomial_sampling
    beam_search_generator.multinomial_sampling = True
    beam_search_generator.temperature = 0.95
    beam_search_generator.top_p_sampling = 0.9
    beam_search_generator.top_k_sampling = 10
    generated_sequences = tokenizer.batch_decode(
        beam_search_generator.generate(input_texts),
        skip_special_tokens=True,
    )
    beam_search_generator.temperature = prev_temp
    beam_search_generator.multinomial_sampling = prev_sampling
    assert (
        0
        < bleu_scorer.compute(
            predictions=generated_sequences,
            references=targets,
        )["score"]
    )


gpt2_greedy_generator = GreedyGenerator(
    use_tqdm=True,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    device=gpt2_model.device,
    generation_forward=lambda encoder_inputs, decoder_inputs: gpt2_model(
        input_ids=decoder_inputs
    ).logits,
    eos_token_id=gpt2_model.generation_config.eos_token_id,
    decoder_start_token_id=gpt2_model.generation_config.decoder_start_token_id,
)


def test_gpt2_greedy_generate():
    prompts = [
        "Once upon a time,",
        "another once, upon a time,",
        "Also, another once, upon a time,",
        "Finally, one more final once, upon a time,",
    ]
    input_ids = [
        gpt2_tokenizer(prompt, return_tensors="pt").to(DEVICE).input_ids.squeeze()
        for prompt in prompts
    ]
    results = gpt2_greedy_generator.generate(
        encoder_inputs=None,
        decoder_inputs=input_ids,
        pad_decoder_inputs=gpt2_tokenizer.bos_token_id,
    )
    assert len(results) == len(prompts)
    assert all(len(result) == MAX_LENGTH for result in results)
    # for debugging:
    # for result in results:
    #     print(gpt2_tokenizer.decode(result[:5], skip_special_tokens=True))


gpt2_beam_search_generator = BeamSearchGenerator(
    use_tqdm=True,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    device=gpt2_model.device,
    generation_forward=lambda encoder_inputs, decoder_inputs: gpt2_model(
        input_ids=decoder_inputs
    ).logits,
    eos_token_id=gpt2_model.generation_config.eos_token_id,
    decoder_start_token_id=gpt2_model.generation_config.decoder_start_token_id,
)


def test_gpt2_beam_search_generate():
    prompts = [
        "Once upon a time,",
        "another once, upon a time,",
        "Also, another once, upon a time,",
        "Finally, one more final once, upon a time,",
    ]
    input_ids = [
        gpt2_tokenizer(prompt, return_tensors="pt").to(DEVICE).input_ids.squeeze()
        for prompt in prompts
    ]
    results = gpt2_beam_search_generator.generate(
        encoder_inputs=None,
        decoder_inputs=input_ids,
        pad_decoder_inputs=gpt2_tokenizer.bos_token_id,
    )
    assert len(results) == len(prompts)
    assert all(len(result) == MAX_LENGTH for result in results)
    # for debugging:
    # print(gpt2_tokenizer.batch_decode(results, skip_special_tokens=True))


def test_greedy_generate_with_logits():
    greedy_generator.return_logits = True
    generated_sequences = greedy_generator.generate(input_texts)

    assert len(generated_sequences) == len(input_texts)
    for (seq, logits), input_text in zip(generated_sequences, input_texts):
        assert isinstance(seq, torch.Tensor)
        assert isinstance(logits, list)
        assert len(logits) > 0
        assert all(isinstance(token, int) and isinstance(logit, float) for token, logit in logits)
        assert len(seq) == len(logits)  # for the bos token

    # Check if the generated sequences (without logits) still have a reasonable BLEU score
    sequences_only = [seq for seq, _ in generated_sequences]
    decoded_sequences = tokenizer.batch_decode(sequences_only, skip_special_tokens=True)
    bleu_score = bleu_scorer.compute(predictions=decoded_sequences, references=targets)["score"]
    assert 0 < bleu_score < 100  # Ensure we get a valid BLEU score

    greedy_generator.return_logits = False  # Reset for other tests


def test_beam_search_generate_with_logits():
    beam_search_generator.return_logits = True
    generated_sequences = beam_search_generator.generate(input_texts)

    assert len(generated_sequences) == len(input_texts)
    for (seq, logits), input_text in zip(generated_sequences, input_texts):
        assert isinstance(seq, torch.Tensor)
        assert isinstance(logits, list)
        assert len(logits) > 0
        assert all(isinstance(token, int) and isinstance(logit, float) for token, logit in logits)
        assert len(logits) == len(seq), f"Expected {len(seq)} logits, got {len(logits)}"

        # In beam search, EOS tokens might have non-zero logits, so we don't check for that

    # Check if the generated sequences (without logits) still have a reasonable BLEU score
    sequences_only = [seq for seq, _ in generated_sequences]
    decoded_sequences = tokenizer.batch_decode(sequences_only, skip_special_tokens=True)
    bleu_score = bleu_scorer.compute(predictions=decoded_sequences, references=targets)["score"]
    assert 0 < bleu_score < 100  # Ensure we get a valid BLEU score

    beam_search_generator.return_logits = False  # Reset for other tests


def test_gpt2_greedy_generate_with_logits():
    gpt2_greedy_generator.return_logits = True
    prompts = [
        "Once upon a time,",
        "In a galaxy far, far away,",
        "It was a dark and stormy night,",
        "The quick brown fox",
    ]
    input_ids = [
        gpt2_tokenizer(prompt, return_tensors="pt").to(DEVICE).input_ids.squeeze()
        for prompt in prompts
    ]
    generated_sequences = gpt2_greedy_generator.generate(
        encoder_inputs=None,
        decoder_inputs=input_ids,
        pad_decoder_inputs=gpt2_tokenizer.eos_token_id,
    )

    assert len(generated_sequences) == len(prompts)
    for (seq, logits), input_ids in zip(generated_sequences, input_ids):
        assert isinstance(seq, torch.Tensor)
        assert isinstance(logits, list)
        assert len(logits) > 0
        assert all(isinstance(token, int) and isinstance(logit, float) for token, logit in logits)
        assert len(logits) == len(seq), f"Expected {len(seq)} logits, got {len(logits)}"

    # Check if the generated sequences are valid continuations
    sequences_only = [seq for seq, _ in generated_sequences]
    decoded_sequences = gpt2_tokenizer.batch_decode(sequences_only, skip_special_tokens=True)
    for prompt, continuation in zip(prompts, decoded_sequences):
        assert continuation.startswith(prompt)
        assert len(continuation) > len(prompt)

    gpt2_greedy_generator.return_logits = False  # Reset for other tests


# The beam search test for GPT-2 remains largely unchanged as EOS tokens might have non-zero logits in beam search
def test_gpt2_beam_search_generate_with_logits():
    gpt2_beam_search_generator.return_logits = True
    prompts = [
        "Once upon a time,",
        "In a galaxy far, far away,",
        "It was a dark and stormy night,",
        "The quick brown fox",
    ]
    input_ids = [
        gpt2_tokenizer(prompt, return_tensors="pt").to(DEVICE).input_ids.squeeze()
        for prompt in prompts
    ]
    generated_sequences = gpt2_beam_search_generator.generate(
        encoder_inputs=None,
        decoder_inputs=input_ids,
        pad_decoder_inputs=gpt2_tokenizer.eos_token_id,
    )

    assert len(generated_sequences) == len(prompts)
    for (seq, logits), input_ids in zip(generated_sequences, input_ids):
        assert isinstance(seq, torch.Tensor)
        assert isinstance(logits, list)
        assert len(logits) > 0
        assert all(isinstance(token, int) and isinstance(logit, float) for token, logit in logits)
        assert len(logits) == len(seq), f"Expected {len(seq)} logits, got {len(logits)}"

    # Check if the generated sequences are valid continuations
    sequences_only = [seq for seq, _ in generated_sequences]
    decoded_sequences = gpt2_tokenizer.batch_decode(sequences_only, skip_special_tokens=True)
    for prompt, continuation in zip(prompts, decoded_sequences):
        assert continuation.startswith(prompt)
        assert len(continuation) > len(prompt)
    gpt2_beam_search_generator.return_logits = False  # Reset for other tests
