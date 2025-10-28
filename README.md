# HAR & CoPA
Official implementation for the paper:

**[NeurIPS'25] Bootstrapping Hierarchical Autoregressive Formal Reasoner with Chain-of-Proxy-Autoformalization**

Qi Liu, Xinhao Zheng, Renqiu Xia, Qinxiang Cao, Junchi Yan*

School of Computer Science & School of Artificial Intelligence, Shanghai Jiao Tong University

Shanghai Innovation Institute

(∗Corresponding author)

[🤗Collection](https://huggingface.co/collections/purewhite42/har-and-copa)
[📃Paper](https://openreview.net/pdf?id=2Xn8h68mP3)

## Quick Start
1. **Install and Build Dependencies**: Follow [Python Dependencies](#python-dependencies) and [Lean 4 Dependencies](#lean-4-dependencies).
2. **Download Model**: Download [🤗purewhite42/HAR_CoPA_Cycle2](https://huggingface.co/purewhite42/HAR_CoPA_Cycle2).
3. **Reproduce Experiments**: Deploy the model and evaluate HAR-Cycle2:
```shell
# Deploy model (replace /path/to/HAR_CoPA_Cycle2 with the real path)
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/HAR_CoPA_Cycle2 \
    --port 37210 \
    --api-key neurips25_har_copa \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests

# Evaluate HAR-Cycle2 on FormalMath500 (replace /path/to/HAR_CoPA_Cycle2 and /path/to/mathlib4 with the real path)
ulimit -s unlimited;
python -m evaluator.h_ar \
    --log_root output/cycle2/har/HAR_CoPA_Cycle2 \
    --benchmark formal_math500 \
    --solver_agent sft_vanilla \
    --solver_base_url "http://0.0.0.0:37210/v1" \
    --solver_api_key "neurips25_har_copa" \
    --solver_model_name /path/to/HAR_CoPA_Cycle2 \
    --prover_agent sft_vanilla \
    --prover_base_url "http://0.0.0.0:37210/v1" \
    --prover_api_key "neurips25_har_copa" \
    --prover_model_name /path/to/HAR_CoPA_Cycle2  \
    --benchmark_root data/benchmark \
    --project_root /path/to/mathlib4 \
    --solver_max_search_trials 80 \
    --num_concurrency 16;
```
4. Collect the experiment results with `result_collection.ipynb`.

## Installation
### Python Dependencies
| **Name**                                                            | **Version**                          |
|-----------------------------------------------------------------|------------------------------------------|
| [XTuner](https://github.com/InternLM/xtuner)                    | 0.1.23                                   |
| [vLLM](https://github.com/vllm-project/vllm)                    | 0.7.3                                   |
| [PyTorch](https://pytorch.org/)                    | 2.5.1                                   |
| [Transformers](https://github.com/huggingface/transformers)                    | 4.52.4                                   |

### Lean 4 Dependencies
We recommend using [elan](https://github.com/leanprover/elan) to manage Lean versions.

| **Name**                                                            | **Version**                          |
|-----------------------------------------------------------------|------------------------------------------|
| [Lean 4](https://github.com/leanprover/lean4)                 | v4.15.0                                |
| [Pantograph](https://github.com/leanprover-community/repl)            | v0.2.25 |
| [Mathlib 4](https://github.com/leanprover-community/mathlib4) | v4.15.0 |

### Environment Setting
After installing Pantograph, link its executables to the `common/pantograph` directory.
```shell
ln -s /path/to/Pantograph/lean-toolchain common/pantograph/lean-toolchain
ln -s /path/to/Pantograph/.lake/build/bin/repl common/pantograph/pantograph-repl
```

The final structure will look like this:
```shell
.
├── common
│   ├── pantograph
│   │   ├── lean-toolchain -> /path/to/Pantograph/lean-toolchain
│   │   ├── pantograph-repl -> /path/to/Pantograph/.lake/build/bin/repl
```
Set the default Lean version to v4.15.0:
```shell
elan default leanprover/lean4:v4.15.0 
```

## Dataset
[🤗purewhite42/CoPA_Dataset](https://huggingface.co/datasets/purewhite42/CoPA_Dataset) CoPA produces training data available for the following tasks:
- _Statement Autoformalization_ (`statement_autoformalization`): Given an informal problem and its informal ground-truth answer, the model outputs a corresponding formal statement.
- _Solution Drafting_ (`solution_drafting`): Input an informal problem, its informal ground-truth answer, its informal ground-truth solution, and a corresponding formal statement, and the model outputs a formal solution draft.
- _Next-Proof-Step Prediction_ (`next_proof_step_prediction`): Input the current proof state; the model outputs the next proof step.
- _Next-Solution-Step Prediction_ (`next_solution_step_prediction`): Input an informal problem and a current solution state; the model outputs the next solution step.
- _Next-Solution-Step Drafting_ (`next_solution_step_drafting`): Given an informal problem and a current solution state, the model outputs the next formal solution step draft.
- _Whole-Solution Generation_ (`whole_solution_generation`): Given an informal problem and its initial solution state, the model outputs a whole formal solution.
- _Whole-Solution Drafting_ (`whole_solution_drafting`): Input an informal problem and its initial solution state; the model outputs a formal solution draft.

All `jsonl` SFT data are organized as single-turn conversations with the following fields:
| **Field**                  | **Description**                                            |
| ---------------------- | -------------------------------------------------------------- |
| `system`        | System Prompt                                             |
| `input`          | User Input                                               |
| `output`               | Model Output             |

Training recipes of each pipeline are:
- _CoPA_, _H-SA_: Statement Autoformalization, Solution Drafting, Next-Proof-Step Prediction;
- _BFS_, _AR_: Next-Solution-Step Prediction;
- _WG_: Whole-Solution Generation;
- _H-BFS_, _HAR_: Next-Proof-Step Prediction, Next-Solution-Step Drafting;
- _H-WG_: Next-Proof-Step Prediction, Whole-Solution Drafting.

## Usage
### Training
Our models are supervised fine-tuned (SFT) from `Qwen/Qwen2.5-Math-7B` with hyperparameters detailed in `config/xtuner_train_config.py`.

If using CUDA devices, please tune line 183 of `config/xtuner_train_config.py` to `backend='nccl'`.

Please use 8 XPUs (GPU/NPU/TPU/...) to maintain the global batch size `batch_size`\*`accumulative_counts`\*`NPROC_PER_NODE`=512. Otherwise, please adjust `accumulative_counts` accordingly.

```shell
export SFT_TASK_NAME="your_experiment_config"
NPROC_PER_NODE=8 xtuner train ./config/${SFT_TASK_NAME}.py --deepspeed deepspeed_zero2;
# XTuner saves training ckpts at `./work_dirs`
xtuner convert pth_to_hf ./config/${SFT_TASK_NAME}.py ./work_dirs/${SFT_TASK_NAME}/epoch_1.pth /path/to/save/converted/ckpts/${SFT_TASK_NAME}/;
```

### Inference
#### Model Deployment
Please download model checkpoints from [🤗NeurIPS25 HAR & CoPA](https://huggingface.co/collections/purewhite42/har-and-copa).
We recommend using [vLLM](https://github.com/vllm-project/vllm) to serve the models locally.
```shell
# NPU
export ASCEND_RT_VISIBLE_DEVICES=...;
# CUDA
export CUDA_VISIBLE_DEVICES=...;

python -m vllm.entrypoints.openai.api_server \
 --model /path/to/model \
    --port ... \ # Can be arbitrarily set (should avoid conflict), e.g., 37210 
    --api-key ... \ # Can be arbitrarily set, e.g., "neurips25_har_copa"
    --dtype bfloat16 \
 --trust-remote-code \
    --enable-prefix-caching \
 --disable-log-requests
```

Please collect the results of the following experiments with `result_collection.ipynb`.

#### HAR
```shell
ulimit -s unlimited;
python -m evaluator.h_ar \
    --log_root output/cycleX/har/model_name \   # {(cycle1, HAR_CoPA_Cycle1), (cycle2, HAR_CoPA_Cycle2)}
    --benchmark ... \    # {formal_math500, minif2f_solving, mathodessy, putnam_solving}
    --solver_agent sft_vanilla \
    --solver_base_url ... \ # URL of the HAR model API, e.g., "http://0.0.0.0:37210/v1"
    --solver_api_key ... \ # API Key of the HAR model API
    --solver_model_name ... \ # Model name of the HAR model
    --prover_agent sft_vanilla \
    --prover_base_url ... \ # URL of the next-proof-step prediction model API, e.g., "http://0.0.0.0:37210/v1"
    --prover_api_key ... \ # API Key of the next-proof-step prediction model API
    --prover_model_name ...  \ # Model name of the next-proof-step prediction model
    --benchmark_root ... \ # Usually "data/benchmark"
    --project_root ... \ # Path to the Pantograph working directory, e.g. ../mathlib4
    --solver_max_search_trials 80 \ # the maximum number of step generation attempts $K_S$
    --num_concurrency 16
```

#### H-BFS (Hierarchical Best-first Search)
```shell
ulimit -s unlimited;
python -m evaluator.h_bfs \
    --log_root output/cycleX/h_bfs/model_name \   # Also use HAR models, {(cycle1, HAR_CoPA_Cycle1), (cycle2, HAR_CoPA_Cycle2)}
    --benchmark ... \    # {formal_math500, minif2f_solving, mathodessy, putnam_solving}
    --solver_agent sft_vanilla \
    --solver_base_url ... \ # URL of the HAR model API, e.g., "http://0.0.0.0:37210/v1"
    --solver_api_key ... \ # API Key of the HAR model API
    --solver_model_name ... \ # Model name of the HAR model
    --prover_agent sft_vanilla \
    --prover_base_url ... \ # URL of the next-proof-step prediction model API, e.g., "http://0.0.0.0:37210/v1"
    --prover_api_key ... \ # API Key of the next-proof-step prediction model API
    --prover_model_name ...  \ # Model name of the next-proof-step prediction model
    --benchmark_root ... \ # Usually "data/benchmark"
    --project_root ... \ # Path to the Pantograph working directory, e.g. ../mathlib4
    --solver_max_search_trials 80 \ # the maximum number of step generation attempts $K_S$
    --num_concurrency 16
```

#### H-SA (Hierarchical Solution Autoformalization)
```shell
ulimit -s unlimited;
python -m evaluator.h_sa \
    --log_root output/cycleX/h_sa/model_name \ # Also use HAR models, {(cycle1, HAR_CoPA_Cycle1), (cycle2, HAR_CoPA_Cycle2)}
    --benchmark ... \    # {formal_math500, minif2f_solving, mathodessy, putnam_solving}
    --solver_base_url ... \ # URL of the HAR model API, e.g., "http://0.0.0.0:37210/v1"
    --solver_api_key ... \ # API Key of the HAR model API
    --solver_model_name ... \ # Model name of the HAR model
    --prover_agent sft_vanilla \
    --prover_base_url ... \ # URL of the next-proof-step prediction model API, e.g., "http://0.0.0.0:37210/v1"
    --prover_api_key ... \ # API Key of the next-proof-step prediction model API
    --prover_model_name ...  \ # Model name of the next-proof-step prediction model
    --benchmark_root ... \ # Usually "data/benchmark"
    --project_root ... \ # Path to the Pantograph working directory, e.g. ../mathlib4
    --try_num 8 \ # the maximum number of whole-solution generation attempts $K_W$
    --num_concurrency 16
```

#### H-WG (Hierarchical Whole-Generation)
```shell
ulimit -s unlimited;
python -m evaluator.h_wg \
    --log_root output/cycle1/h_wg/HWG_CoPA_Cycle1 \
    --benchmark ... \    # {formal_math500, minif2f_solving, mathodessy, putnam_solving}
    --solver_base_url ... \ # URL of the H-WG model API, e.g., "http://0.0.0.0:37210/v1"
    --solver_api_key ... \ # API Key of the H-WG model API
    --solver_model_name ... \ # Model name of the H-WG model
    --prover_agent sft_vanilla \
    --prover_base_url ... \ # URL of the next-proof-step prediction model API, e.g., "http://0.0.0.0:37210/v1"
    --prover_api_key ... \ # API Key of the next-proof-step prediction model API
    --prover_model_name ...  \ # Model name of the next-proof-step prediction model
    --benchmark_root ... \ # Usually "data/benchmark"
    --project_root ... \ # Path to the Pantograph working directory, e.g. ../mathlib4
    --try_num 8 \ # the maximum number of whole-solution generation attempts $K_W$
    --num_concurrency 16
```

#### AR (Autoregressive Reasoning)
```shell
ulimit -s unlimited;
python -m evaluator.ar \
    --log_root output/cycle1/ar/AR_CoPA_Cycle1 \
    --benchmark ... \    # {formal_math500, minif2f_solving, mathodessy, putnam_solving}
    --solver_base_url ... \ # URL of the AR model API, e.g., "http://0.0.0.0:37210/v1"
    --solver_api_key ... \ # API Key of the AR model API
    --solver_model_name ... \ # Model name of the AR model
    --benchmark_root ... \ # Usually "data/benchmark"
    --project_root ... \ # Path to the Pantograph working directory, e.g. ../mathlib4
    --solver_max_search_trials 80 \ # the maximum number of step generation attempts $K_S$
    --num_concurrency 16
```

#### BFS (Best-first Search)
```shell
ulimit -s unlimited;
python -m evaluator.bfs \
    --log_root output/cycle1/bfs/AR_CoPA_Cycle1 \
    --benchmark ... \    # {formal_math500, minif2f_solving, mathodessy, putnam_solving}
    --solver_base_url ... \ # URL of the BFS model API, e.g., "http://0.0.0.0:37210/v1"
    --solver_api_key ... \ # API Key of the BFS model API
    --solver_model_name ... \ # Model name of BFS model
    --benchmark_root ... \ # Usually "data/benchmark"
    --project_root ... \ # Path to the Pantograph working directory, e.g. ../mathlib4
    --solver_max_search_trials 80 \ # the maximum number of step generation attempts $K_S$
    --num_concurrency 16
```

#### WG (Whole-Solution Generation)
```shell
ulimit -s unlimited;
python -m evaluator.wg \
    --log_root output/cycle1/wg/WG_CoPA_Cycle1 \
    --benchmark ... \    # {formal_math500, minif2f_solving, mathodessy, putnam_solving}
    --solver_base_url ... \ # URL of the WG model API, e.g., "http://0.0.0.0:37210/v1"
    --solver_api_key ... \ # API Key of the WG model API
    --solver_model_name ... \ # Model name of WG model
    --benchmark_root ... \ # Usually "data/benchmark"
    --project_root ... \ # Path to the Pantograph working directory, e.g. ../mathlib4
    --try_num 8 \ # the maximum number of whole-solution generation attempts $K_W$
    --num_concurrency 16
```

## Citation
If you find our work useful in your research, please cite
```bibtex
@inproceedings{
liu2025bootstrapping,
title={Bootstrapping Hierarchical Autoregressive Formal Reasoner with Chain-of-Proxy-Autoformalization},
author={Liu, Qi and Zheng, Xinhao and Xia, Renqiu and Cao, Qinxiang and Yan, Junchi},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=2Xn8h68mP3}
}
```

## License
This project is released under the Apache 2.0 license. Please see the [LICENSE](https://github.com/Purewhite2019/har_copa_main/blob/main/LICENSE) file for more information.

## Contact
Feel free to discuss the paper/data/code with us through issues/emails!
- Qi Liu: purewhite@sjtu.edu.cn
