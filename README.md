
# Quantized Side Tuning: Fast and Memory-Efficient Tuning of Quantized Large Language Models

## Quick Start

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Finetuning large language models (LLMs) has been empirically effective on a variety of downstream tasks. Existing approaches to finetuning an LLM either focus on parameter-efficient finetuning, which only updates a small number of trainable parameters, or attempt to reduce the memory footprint during the training phase of the finetuning. Typically, the memory footprint during finetuning stems from three contributors: model weights, optimizer states, and intermediate activations. However, existing works still require considerable memory and none can simultaneously mitigate memory footprint for all three sources. In this paper, we present Quantized Side Tuing (QST), which enables memory-efficient and fast finetuning of LLMs by operating through a dual-stage process. First, QST quantizes an LLM's model weights into 4-bit to reduce the memory footprint of the LLM's original weights; QST also introduces a side network separated from the LLM, which utilizes the hidden states of the LLM to make task-specific predictions. Using a separate side network avoids performing backpropagation through the LLM, thus reducing the memory requirement of the intermediate activations. Furthermore, QST leverages several low-rank adaptors and gradient-free downsample modules to significantly reduce the trainable parameters, so as to save the memory footprint of the optimizer states. 
Experiments show that QST can reduce the total memory footprint by up to 2.3 times and speed up the finetuning process by up to 3 times while achieving competent performance compared with the state-of-the-art. When it comes to full finetuning, QST can reduce the total memory footprint up to 7 times.

## Features

- **Feature 1**: 4-bit Quantization.
- **Feature 2**: Side tuning.

## Installation

1. git clone the repo:
   ```
   git clone https://github.com/YouAreSpecialToMe/QST.git
2. install requirements
   ```
   cd QST
   pip install -r requirements.txt

### Prerequisites

What things you need to install the software and how to install them.

### Setup

A step by step series of examples that tell you how to get a development environment running.

```bash
# Give the example
