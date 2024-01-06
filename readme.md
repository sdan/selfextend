# SelfExtend Attention for Mistral

Implementation of the Self-Extend paper that uses group attention to extend context windows of LLMs without fine-tuning/pre-training.

## Overview

The SelfExtend mechanism modifies the standard attention mechanism in the Mistral model to improve its context capturing capabilities. This is achieved by extending the attention span of the model, allowing it to consider a broader context while making predictions. This enhancement is particularly useful in tasks involving long sequences of data.

## Features

- **Compatibility**: Designed to work with the Hugging Face Transformers library.
- **Extended Context**: Currently it can take Mistral 7b's 8k context to 16k.
- **Grouped Attention**: Utilize a novel attention mechanism that groups tokens to mitigate the positional O.O.D. issue

## Requirements

To use this implementation, the following prerequisites must be met:

- Python 3.10
- PyTorch
- Transformers Library

## Installation

Clone the repository to your local machine and copy the modeling files into `transformers/src/transformers/models/mistral`

When initializing the weights specify the self_extend attention mechanism as such:

`model = MistralForCausalLM.from_pretrained("hf_mistral-7B-v0.1", attn_implementation="self_extend")`