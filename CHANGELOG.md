# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [v0.0.9](https://github.com/MagedSaeed/generate-sequences/releases/tag/v0.0.9) - 2024-06-25

### Added

- Add decoder-only generation.
- Add tests for decoder-only generation.
- Add a notebook to showcase decoder-only generation.

### Changed
- Restructuring the package, generate now is a subpackage istead of being a file.
- Restructure the notebooks into a folder with different notebooks for each generation architecture.

### Fixed
- Update tests to fix huggingface warnings.
- Fixed issues related to mypy in pre-commit.

## [v0.0.8](https://github.com/MagedSaeed/generate-sequences/releases/tag/v0.0.8) - 2024-05-19

### Added

- Add top-p and top-k sampling.
- Add tests for these new feature.
- update hf_generate notebook

### Changed
- Minor fixes and renaming.

## [v0.0.7](https://github.com/MagedSaeed/generate-sequences/releases/tag/v0.0.7) - 2024-05-8

### Added

- Add the feature of sorting samples before generation.
- Add tests for this new feature.

### Changed
- Minor fixes and renaming.


## [v0.0.6](https://github.com/MagedSaeed/generate-sequences/releases/tag/v0.0.6) - 2024-05-3

### Added

- Improve the code quality for beam search.
- General config updates.

## [v0.0.5](https://github.com/MagedSaeed/generate-sequences/releases/tag/v0.0.5) - 2024-04-29

### Added

- rename `generate_fn` to more appropriate name: `generation_forward`

## [v0.0.4](https://github.com/MagedSaeed/generate-sequences/releases/tag/v0.0.4) - 2024-04-28

### Added

- Add multinomial sampling for both generation methods, greedy and beam search.
- Add tests for multinomial sampling.
- Inegrate some examples regarding multinomial sampling in hf_compre notebook.

## [v0.0.3](https://github.com/MagedSaeed/generate-sequences/releases/tag/v0.0.3) - 2024-04-23

### Added

- Add temperature parameter for beam search generation.
- Add tests for temperature parameter.
- Add documentation when needed.

### Changed
- `length_penalty_alpha` parameter of beam search has been changed to `length_penalty`.
- Update the hf_compare notebook to reflect the new changes.

### Removed
- Remove the function `sample_tokens_probs` and replace its code in the `generate` method for each algorithm.
- Remove `minimum_penalty_tokens_length` from beam search generation.

## [v0.0.2](https://github.com/MagedSaeed/generate-sequences/releases/tag/v0.0.2) - 2024-04-21

### Added

- Add the beam search generation.
- Enrich the example notebook.
- Tests.

### Changed

- The method that samples the tokens is renamed from `get_next_tokens` to `sample_tokens_probs`

### FIXED

### REMOVED

## [0.0.1](https://github.com/MagedSaeed/generate-sequences/releases/tag/v0.0.1) - 2024-04-18

### Added

- Generate using greedy search.
- Notebook to compare the results on a huggingface model.
- Tests.

### Removed

- Unused code used to initialize the project to pypi.


## [v0.0.0](https://github.com/MagedSaeed/generate-sequences/releases/tag/v0.0.0) - 2024-03-30

### Added

- v0.0.0 pushing the project to pypi


