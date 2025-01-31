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
Here’s a quick example of how to use Generate Sequences:

- Greedy Search

```python
from src.generators.greedy_search import GreedySearch

generator = GreedySearch(model, tokenizer)
sequence = generator.generate(input_ids, max_length=50)
print(sequence)
```
- Beam Search

```python
from src.generators.beam_search import BeamSearch

generator = BeamSearch(model, tokenizer, num_beams=5)
sequence = generator.generate(input_ids, max_length=50)
print(sequence)
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