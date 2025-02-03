# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-02-03

### OFFICAL RELEASE

## [Unreleased]

## [0.9.92] - 2024-12-31

### Fixed
- Now can use different model of the same class in LLM

## [0.9.91] - 2024-11-26

### Added
- Added Sonnet 3.5-v2 support in:
    - AWS Bedrock
    - Anthropic
- Added support for inference profiles models in AWS Bedrock, models:
    - Anthropic models
    - LLAMA
- Note: Now default model for Anthropic API will be directed to latest versioning
- Added Timeout support while calling LLM

## [0.9.9] - 2024-11-10

### Added
- Using converse api for aws bedrock to invoke LLM
- Lingua bug solved
- Have added filter, sparse and id funtionality in pinecone
- New LLMs Anthropic 3.5, LLAMA 3.1 onwards and Mistral series
- Libs updated

## [0.9.8] - 2024-08-8

### Added
- Now sse is optional while streaming, simply get exact chunks while streaming
- Can add custom metadata while streaming or normal llm call
- Have added fetch() funtionality in knowledge base

### Changed
- Made Request optional so that orichain can be more versatile rather than just be used in api

### Fixed
- Lingua custom language error solved
- GenAI Validation check bug related to prev_chunk solved

## [0.9.7] - 2024-07-31

### Added
- Added Azure OpenAI embedding support

### Changed
- Default Azure OpenAI endpoint 

### Fixed
- Solved bug related to generative validation function

## [0.9.6] - 2024-07-29

### Added
- Added language detection 

### Changed
- Changed /results endpoint and utils.results_processsing()

### Fixed
- Solved bug related to ChromaDB