# HVO: Hypervolume Optimization for Multi-Objective Text Summarization based on GRPO

## Overview
Text summarization is a crucial task that requires the simultaneous optimization of multiple objectives, including consistency, coherence, relevance, and fluency, which presents considerable challenges. Although large language models (LLMs) have demonstrated remarkable performance, enhanced by reinforcement learning (RL), few studies have focused on optimizing the multi-objective problem of summarization through RL based on LLMs. In this paper, we introduce hypervolume optimization (HVO), a novel optimization strategy that dynamically adjusts the scores between groups during the reward process in RL by using the hypervolume method. This method guides the model’s optimization to progressively approximate the pareto front, thereby generating balanced summaries across multiple objectives.

## Key Features
- **Multi-Objective RL**: Balances multiple reward signals during model training to ensure high-quality summaries.
- **Hypervolume Maximization**: Utilizes the hypervolume metric to guide the model’s optimization toward the Pareto optimal frontier.
- **Enhanced Stability**: Incorporates a length constraint mechanism to address summary length collapse during training.

## Requirements
- Python 3.10
- PyTorch 
- UniEval for multi-dimensional evaluation
- Pre-trained language models (e.g., Qwen 2.5)

## Setup and Installation

Clone the repository:
```bash
git clone https://github.com/ai4business-LiAuto/HVO.git
cd HVO
```
Install required dependencies:

```bash
pip install -r requirements.txt
```

Usage
```
sh run.sh
```
