# 🔢 Generate Sequences

`generate-sequences` is a Python library for generating sequences from deep learning architectures with support for greedy search, beam search, and customizable configurations.

This package generates sequences from architectures developed with PyTorch. It can generate in a greedy manner or using beam search. It can generate from both decoder-only and encoder-decoder architectures.

- **Introduction and Getting Started**: This README.md, also: <https://magedsaeed.github.io/generate-sequences/Getting%20Started>
- **Examples**: <https://magedsaeed.github.io/generate-sequences/examples/huggingface_encoder_decoder>
- **ChangeLog**: <https://magedsaeed.github.io/generate-sequences/CHANGELOG>

# 🚀 Introduction
Welcome to Generate Sequences, a Python package designed for generating sequences using popular decoding strategies. Whether the system is a text generation, language modeling, machine translation, or any task requiring sequence decoding, this library provides a simple interface to implement and customize the sequence generation pipelines.

# 🎯 Motivation
Hugging Face’s `model.generate` method is a great tool for generating sequences from LLMs. However, in order to use this method, the developed model needs to be a Hugging Face model adhering to specific constraints depending on the model architecture. This package generalizes that approach by introducing a `generation_forward` method, where you specify how the model outputs sequences (this is the part that differs from model to model). Other parts are standardized across both beam search and greedy search. Additionally, this package supports popular generation methods such as sampling, temperature, top-p sampling, and top-k sampling.

# ✨ Features
- **Greedy Search**: Generate sequences by selecting the most probable token at each step.
- **Beam Search**: Explore multiple hypotheses to generate the best possible sequence.
- **Custom Configurations**: Fine-tune decoding parameters to suit your specific task.
- **Lightweight & Modular**: Easy-to-read code designed with flexibility in mind.

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

You can also install the package from the latest commit on GitHub by cloning the repository and installing it manually:

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

Then, specify how the package should generate from your model given the encoder and decoder inputs and returns the model logits. This method takes as an argument the encoder `inputs` (can be None for decoder-only architecture) and the decoder `decoder_input_ids` at a given time step.

This method This can be as simple as:

```python
def generate(inputs, decoder_input_ids):
    tokenizer_results = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
    )
    model_outputs = model(**tokenizer_results)
    return model_outputs.logits
```

Or more advanced with caching mechanism in encoder-decoder architectures where the encoder outputs are cached:

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

Then, you can perform greedy search or beam search as follows:

- **Greedy Search**

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
```

- **Beam Search**

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
```

# 🧪 Running Tests
Ensure everything works as expected by running the unit tests:

```bash
pytest tests/
```

# 📖 Documentation
You can explore the full docs here: https://magedsaeed.github.io/generate-sequences. You can also navigate through the repository to: 
- **Getting Started**: Check out `docs/Getting Started.ipynb` for a step-by-step guide.
- **Examples**: Notebooks demonstrating integration with Hugging Face models are available in `docs/examples/`

# 📜 ChangeLog
You can find a detailed list of changes and updates in the [ChangeLog](https://magedsaeed.github.io/generate-sequences/CHANGELOG). This document keeps track of new features, bug fixes, and other improvements.

# 🤝 Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

Please ensure your code follows the existing style and includes appropriate tests.

# 🛡️ License
This project is licensed under the MIT License. See the LICENSE file for details.

# 🌟 Acknowledgments
Special thanks to the open-source community for inspiring this project 🙌. I want to give the credit here to Allen AI Python Package Template (https://github.com/allenai/python-package-template), which provides excellent functionality including GitHub Actions with automated testing and PyPI deployments.
