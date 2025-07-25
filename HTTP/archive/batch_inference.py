# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
experimental support for tensor-parallel inference with torchrun,
see https://github.com/vllm-project/vllm/issues/11400 for
the motivation and use case for this example.
run the script with `torchrun --nproc-per-node=2 torchrun_example.py`,
the argument 2 should match the `tensor_parallel_size` below.
see `tests/distributed/test_torchrun_example.py` for the unit test.
"""

import torch.distributed as dist

from vllm import LLM, SamplingParams


if __name__ == "__main__":

    # Create prompts, the same across all ranks
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create sampling parameters, the same across all ranks
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Use `distributed_executor_backend="external_launcher"` so that
    # this llm engine/instance only creates one worker.
    # it is important to set an explicit seed to make sure that
    # all ranks have the same random seed, so that sampling can be
    # deterministic across ranks.
    llm = LLM(
        model="meta-llama/Llama-3.1-8B",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        distributed_executor_backend="external_launcher",
        max_model_len=32768,
        seed=1,
    )

    outputs = llm.generate(prompts, sampling_params)

    # all ranks will have the same outputs
    if dist.get_rank() == 0:
        print("-" * 50)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n")
            print("-" * 50)
    """
Further tips:

1. to communicate control messages across all ranks, use the cpu group,
a PyTorch ProcessGroup with GLOO backend.

```python
from vllm.distributed.parallel_state import get_world_group
cpu_group = get_world_group().cpu_group
torch_rank = dist.get_rank(group=cpu_group)
if torch_rank == 0:
    # do something for rank 0, e.g. saving the results to disk.
```

2. to communicate data across all ranks, use the model's device group,
a PyTorch ProcessGroup with NCCL backend.
```python
from vllm.distributed.parallel_state import get_world_group
device_group = get_world_group().device_group
```

3. to access the model directly in every rank, use the following code:
```python
llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
```
"""
