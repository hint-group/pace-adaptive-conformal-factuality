# PACE: Probabilistic Adaptive Conformal Estimation

PACE is a framework for **Conformal Prediction (CP)** that adapts strictly to distribution shifts in data streams using an adaptive gamma parameter. This repository contains the implementation of PACE and its evaluation on various benchmarks.

## Project Structure

The codebase is organized into three main experiment directories:

*   **`synthetic_exp/`**: Contains synthetic experiments designed to simulate controlled distribution shifts (e.g., smooth shifts, jump shifts) to test the robustness of the adaptive CP method.
*   **`MMLU/`**: Experiments on the **MMLU (Massive Multitask Language Understanding)** benchmark, evaluating the method's performance on large language model outputs.
*   **`WikiData/`**: Experiments on the **WikiData** benchmark, focusing on the factuality and correctness of generated text.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Synthetic Experiments
Navigate to the `synthetic_exp/` directory to run synthetic shift experiments.
```bash
cd synthetic_exp
# Run sensitivity analysis for smooth shifts
python run_sensitivity_smooth.py
# Run sensitivity analysis for jump shifts
python run_sensitivity_jump.py
```

### MMLU Experiments
Navigate to the `MMLU/` directory.
```bash
cd MMLU
# Run MMLU experiments
python mmlu.py
```

### FActScore Experiments
Navigate to the `FActScore/` directory.
```bash
cd FActScore
# Run sensitivity analysis
python sensitivity_gpt5.2_fdp.py
```

## Dependencies
See `requirements.txt` for the full list of python package dependencies. Main requirements include:
- `numpy`, `pandas`, `scipy`
- `torch`, `sentence-transformers`
- `pyserini`, `openai`
