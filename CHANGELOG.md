# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.0] - 2025-08-25

### Added
- Added Together AI LLMs and Embedding Models support.

### Changed
- Changed the sentence-transformers embeddings implementation to handle 'convert_to_tensor' and 'convert_to_numpy' arguments.

## [2.3.0] - 2025-08-24

### Added
- Added support for tool calling for all providers.
- Added OpenAI gpt-5 series and Claude opus-4.1 support.
- Added prompt caching support for Anthropic models.

### Changed
- Changed the default llm model to "gpt-5-mini"
- Updated docs

## [2.2.3] - 2025-07-17

### Changed
- Updated the documentation.

### Added
- Added normalize_embeddings argument in sentence-transformers embedding models' call. 

### Removed
- Removed request_id and uuid from Gemini & VertexAI chat formatters.

## [2.2.2] - 2025-07-15

### Changed
- Changed the docstrings of most of the classes to support reStructuredText (reST) format.
- Changed the provider name of sentence transformers embeddings from "SentenceTransformer" to "SentenceTransformers"

### Added
- Added Documentation for Orichain.

## [2.2.1] - 2025-07-14

### Changed
- Changed the provider name of anthropic bedrock from "AnthropicAWSBedrock" to "AnthropicBedrock"

## [2.2.0] - 2025-06-19

### Changed
- Changed default LLM model to "gpt-4.1-mini"

### Added
- Added Google Gemini and Vertex AI models support

## [2.1.1] - 2025-06-02

### Changed
- Changed default embedding model to "text-embedding-3-small"

## [2.1.0] - 2025-05-30

### Added
- Added a `provider` argument to the initialization of `LLM` and `EmbeddingModel`, which:
  1. Removes ambiguity about which provider is being used.
  2. Enables support for new LLMs and Embedding Models not yet added to Orichain.

### Fixed
- Now can use different model of the same class in EmbeddingModel
- Fixed SentenceTransformer embeddings "List object has not attribute tolist" error.

## [2.0.5] - 2025-05-29

### Added
- Meta Llama 4 Scout and Llama 4 Maverick added in AWSBedrock

## [2.0.4] - 2025-05-28

### Added
- New Claude Sonnet 4 and Opus 4 models support in AnthropicAWSBedrock and Anthropic

## [2.0.3] - 2025-05-05

### Added
- Default prompt caching support for Bedrock and AnthropicBedrock Models.

## [2.0.2] - 2025-04-21

### Added
- LLAMA 3.3 models in AWS Bedrock
### Fixed
- Handled additional library: lingua-language-detector installation

## [2.0.1] - 2025-04-16

### Added
- GPT 4.1 Models support
### Changed
- Renamed 'EmbeddingModels' and 'AsyncEmbeddingModels' classes to 'EmbeddingModel' and 'AsyncEmbeddingModel'

## [2.0.0] - 2025-03-27

### Added
- Async iterator for AWSBedrock Streaming
### Changed
- Using usage in streaming instead of tiktoken

## [2.0.dev1] - 2025-03-06

### Added
- Sync funtionality of each class type
- Function and class definitions
- New args in sentence-transformers models
### Changed
- Async generator of AWSBedrock
- Made lingua-language-detector optional, will only be installed if the user specifies this

## [1.0.9] - 2025-03-06

### Changed
- Added timeout and retry feature in embeddings

## [1.0.8] - 2025-03-06

### Changed
- Made sentence-transformers optional, will only be installed if the user specifies this

## [1.0.7] - 2025-02-28

### Added
- Claude Sonnet 3.7 support on Anthropic API and AWS Bedrock

## [1.0.6] - 2025-02-18

### Added
- Amazon Nova models support on Bedrock

## [1.0.4] - 2025-02-13

### Added
- Pinecone GRPC Support by default
- Upgraded all packages

## [1.0.0] - 2025-02-03

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

### Fixed
- Solved bug related to ChromaDB
