import datasets
import evaluate
from transformers import MarianMTModel, MarianTokenizer

from generate_sequences.generate import GreedyGenerator

mname = "marefa-nlp/marefa-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(mname)
model = MarianMTModel.from_pretrained(mname)


bleu_scorer = evaluate.load("sacrebleu")

test_dataset = datasets.load_dataset("iwslt2017", "iwslt2017-ar-en", split="test")

source_language = "en"
target_language = "ar"

input_texts = [example[source_language] for example in test_dataset["translation"]][-10:]
targets = [example[target_language] for example in test_dataset["translation"]][-10:]

# Initialize GreedyGenerator
greedy_generator = GreedyGenerator(
    device="cpu",
    batch_size=2,
    max_length=50,
    generate_fn=lambda inputs, decoder_input_ids: model(
        **tokenizer(
            inputs,
            padding=True,
            return_tensors="pt",
        ),
        decoder_input_ids=decoder_input_ids,
    ).logits,
    eos_token_id=model.generation_config.eos_token_id,
    decoder_start_token_id=model.generation_config.decoder_start_token_id,
)


# Test generate method
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
