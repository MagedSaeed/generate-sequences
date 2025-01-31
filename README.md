# 🔢 Generate Sequences
A Python library for generating sequences with support for greedy search, beam search, and customizable configurations.

This package generates sequences from deep learning architectures developed with pytorch. It can generate in a greedy nature or using beam search. It can generate from both decoder-only, or encoder-decoder architectures.


- Introduction and Getting Started: <https://magedsaeed.github.io/generate-sequences/Getting%20Started>

- Examples: <https://magedsaeed.github.io/generate-sequences/examples/huggingface_encoder_decoder>

- ChangeLog: <https://magedsaeed.github.io/generate-sequences/CHANGELOG>


# 🚀 Introduction
Welcome to Generate Sequences, a robust Python package designed for generating high-quality sequences using popular decoding strategies. Whether you're working on text generation, machine translation, or any task requiring sequence decoding, this library provides a simple yet powerful interface to implement and customize your sequence generation pipelines.

# ✨ Features
Greedy Search: Quickly generate sequences by selecting the most probable token at each step.
Beam Search: Explore multiple hypotheses to generate the best possible sequence.
Custom Configurations: Fine-tune decoding parameters to suit your specific task.
Lightweight & Modular: Easy-to-read code designed with flexibility in mind.

# 📂 Directory Structure

```plaintext
generate-sequences/
├── docs/                     # Documentation & Examples
│   ├── examples/
│   │   ├── huggingface_decoder_only.ipynb
│   │   ├── huggingface_encoder_decoder.ipynb
│   ├── CHANGELOG.md
│   ├── Getting Started.ipynb
│   ├── index.md
│
├── generate_sequences/        # Core Library
│   ├── generate/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── beam_search.py
│   │   ├── greedy.py
│   ├── utils.py
│   ├── version.py
│   ├── py.typed
│
├── scripts/                   # Utility scripts
│   ├── prepare_changelog.py
│   ├── release_notes.py
│   ├── release.sh
│
├── tests/                     # Unit Tests
│   ├── __init__.py
│   ├── generate_test.py
│   ├── utils_test.py
│
├── .github/                   # GitHub Actions & Workflows
├── LICENSE                    # Project License
├── pyproject.toml             # Project dependencies & settings
├── README.md                  # Project documentation
├── RELEASE_PROCESS.md         # Release workflow guide
└── mkdocs.yml                 # Documentation configuration
```

# 🛠️ Installation

The package is installable through pip:

```bash
pip install -U generate-sequences
```

You can also install the package from the latest commit on git by cloning the repository, then install:

```bash
git clone https://github.com/MagedSaeed/generate-sequences.git
cd generate-sequences
pip install -e .
```
# 💡 Usage

First, import the generators:

```python
from generate_sequences import GreedyGenerator, BeamSearchGenerator
```

then, you need to tell the package how it should generate from your model:

```python
encoder_outputs = {}


def generate(inputs, decoder_input_ids):
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
            input_ids.repeat_interleave(
                model.generation_config.num_beams,
                dim=0,
            ),
            return_dict=True,
            attention_mask=attention_mask,
        )
    model_outputs = model(
        **tokenizer_results,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs[json.dumps(inputs)],
    )
    return model_outputs.logits
```

Then, you can do greedy-search or beam-search generation as follows:

- Greedy Search

```python
greedy_sequences_generator = GreedyGenerator(
    use_tqdm=True,
    sort_inputs_by_size=True,
    device=model.device,
    generation_forward=generate,
    batch_size=model.generation_config.batch_size,
    max_length=model.generation_config.max_length,
    eos_token_id=model.generation_config.eos_token_id,
    decoder_start_token_id=model.generation_config.decoder_start_token_id,
)

prediction_ids = greedy_sequences_generator.generate(input_texts)
predictions = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
len(input_texts), len(predictions), len(targets)
```
- Beam Search

```python
beam_search_sequences_generator = BeamSearchGenerator(
    beam_width=4,
    use_tqdm=True,
    length_penalty=0.6,
    device=model.device,
    sort_inputs_by_size=True,
    generation_forward=generate,
    batch_size=model.generation_config.batch_size,
    max_length=model.generation_config.max_length,
    eos_token_id=model.generation_config.eos_token_id,
    decoder_start_token_id=model.generation_config.decoder_start_token_id,
)

prediction_ids = beam_search_sequences_generator.generate(input_texts)
predictions = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
len(input_texts), len(predictions), len(targets)
```
Customizing Parameters
Both GreedySearch and BeamSearch allow you to customize decoding parameters, such as max_length and num_beams, to tailor the generation process to your needs.

# 🧪 Running Tests
Ensure everything works as expected by running the unit tests:

```bash
pytest tests/
```

# 📖 Documentation
Getting Started: Check out docs/Getting Started.ipynb for a step-by-step guide.
Examples: Notebooks demonstrating integration with Hugging Face models are available in `docs/examples/`

# 🤝 Contributing
Contributions are welcome! To contribute:

- Fork the repository.
- Create a new branch (`git checkout -b feature-name`).
- Commit your changes (`git commit -m "Add feature"`).
- Push to the branch (`git push origin feature-name`).
- Open a pull request.
Please ensure your code follows the existing style and includes appropriate tests.

# 🛡️ License
This project is licensed under the MIT License. See the LICENSE file for details.

# 🌟 Acknowledgments
Special thanks to the open-source community for inspiring this project. 🙌
