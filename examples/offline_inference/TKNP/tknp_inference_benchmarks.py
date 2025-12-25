"""
Inference benchmarking script for vLLM with Token Parallelism (TKNP) support.

Example usage:
Token parallelism: 
torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 1 --enable-token-parallel --token-parallel-size 2 --batch-size 16 --seq-length 4096

Tensor parallelism:
torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 2 --batch-size 16 --seq-length 4096

Pipeline parallelism:
torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 1 --pipeline-parallel-size 2 --batch-size 16 --seq-length 4096

"""


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import argparse
import torch.distributed as dist

from vllm import LLM, SamplingParams
from prompt_generator import generate_benchmark_prompts

import torch
import random
import numpy as np

def parse_args():
    """Parse command line arguments for distributed vLLM inference."""
    parser = argparse.ArgumentParser(description="Distributed vLLM inference with torchrun")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of tensor parallel processes (default: 1)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of pipeline parallel processes (default: 1)")
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="Number of data parallel processes (default: 1)")
    parser.add_argument("--token-parallel-size", type=int, default=1,
                        help="Number of token parallel processes (default: 1)")
    parser.add_argument("--enable-token-parallel", action="store_true",
                        help="Enable token parallelism")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name (default: meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Maximum model length (default: 32768)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    # batch size and seq length for prompts
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for prompts (default: 8)")
    parser.add_argument("--seq-length", type=int, default=128,
                        help="Sequence length for prompts (default: 128)")
    parser.add_argument("--print-outputs", action="store_true",
                        help="Print generated outputs")
    parser.add_argument("--decode-tokens", type=int, default=100,
                        help="Number of tokens to decode during benchmarking (default: 100)")

    return parser.parse_args()


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def inference_benchmark(llm, prompts, args):
    # Warmup run to avoid cold start overhead
    warmup_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1)
    _ = llm.generate(prompts[:1], warmup_params)
    
    # prefill timing
    prefill_tokens = 1
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=prefill_tokens)
    
    # Create CUDA events for precise GPU timing
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()  # Ensure all previous operations complete
    prefill_start.record()
    
    outputs = llm.generate(prompts, sampling_params)
    
    prefill_end.record()
    torch.cuda.synchronize()  # Wait for completion
    
    prefill_time_ms = prefill_start.elapsed_time(prefill_end)

    # prefill + decode (benchmarking)
    decode_tokens = args.decode_tokens
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=decode_tokens)
    
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    decode_start.record()
    
    outputs = llm.generate(prompts, sampling_params)
    
    decode_end.record()
    torch.cuda.synchronize()
    
    total_time_ms = decode_start.elapsed_time(decode_end)
    decode_only_time_ms = total_time_ms - prefill_time_ms
    average_decode_latency = decode_only_time_ms / decode_tokens
    average_decode_system_throughput = (args.batch_size * decode_tokens) / (decode_only_time_ms / 1000)
    average_decode_throughput_per_user = 1/(average_decode_latency / 1000)
    
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Timing Results (Batch size: {args.batch_size}, Seq length: {args.seq_length})")
        print(f"{'='*60}")
        print(f"Prefill time: {prefill_time_ms:.2f} ms")
        print(f"Total time (prefill + {decode_tokens} decode): {total_time_ms:.2f} ms")
        print(f"Decode only time: {decode_only_time_ms:.2f} ms")
        print(f"System Decode Throughput: {average_decode_system_throughput:.2f} tokens/sec")
        print(f"Average decode latency: {average_decode_latency:.2f} ms")
        print(f"Average decode throughput per user: {average_decode_throughput_per_user:.2f} tokens/sec")

        print(f"{'='*60}\n")

    # all ranks will have the same outputs
    if dist.get_rank() == 0 and args.print_outputs:
        print("-" * 50)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt[:128]!r} ....\nGenerated text: {generated_text!r}\n")
            print("-" * 50)

def main():
    args = parse_args()

    # Use `distributed_executor_backend="external_launcher"` so that
    # this llm engine/instance only creates one worker.
    # it is important to set an explicit seed to make sure that
    # all ranks have the same random seed, so that sampling can be
    # deterministic across ranks.
    
    # Prepare LLM kwargs - only include token parallel args if enabled
    llm_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "distributed_executor_backend": "external_launcher",
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "enforce_eager": True,
        "enable_prefix_caching": False,  # Disable prefix caching for benchmarking
        "gpu_memory_utilization": 0.9,  # Max GPU memory utilization
        "max_num_batched_tokens": 32768,  # max number of tokens in a single forward pass
    }
    
    # Only add token parallel configs if token parallelism is enabled
    if args.enable_token_parallel:
        if args.token_parallel_size <= 1:
            raise ValueError("Token parallelism requires token_parallel_size > 1")
        llm_kwargs["enable_token_parallel"] = True
        llm_kwargs["token_parallel_size"] = args.token_parallel_size
    
    llm = LLM(**llm_kwargs)

    if dist.get_rank() == 0:
        if args.enable_token_parallel:
            print(f"LLM initialized with tensor_parallel_size={args.tensor_parallel_size}, pipeline_parallel_size={args.pipeline_parallel_size}, data_parallel_size={args.data_parallel_size}, token_parallel_size={args.token_parallel_size}, enable_token_parallel={args.enable_token_parallel}")
        else:
            print(f"LLM initialized with tensor_parallel_size={args.tensor_parallel_size}, pipeline_parallel_size={args.pipeline_parallel_size}, data_parallel_size={args.data_parallel_size}")
        
        # Generate benchmark prompts
        prompts = generate_benchmark_prompts(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        tokenizer=None,
        model_name=args.model,
        vocab_style="natural",
        seed=42
        )
    else:
        prompts = None
    
    # Broadcast prompts to all ranks
    prompts_list = [prompts]
    dist.broadcast_object_list(prompts_list, src=0)
    prompts = prompts_list[0]
    
    # print(f"Rank {dist.get_rank()} received {len(prompts)} prompts.")
    # print(f"Rank {dist.get_rank()} prompts: {prompts}")
    # assert False, "Debugging: Stop execution here to check prompt distribution."
    
    # Create sampling parameters, the same across all ranks
    
    inference_benchmark(llm, prompts, args)
            
    # destroy the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()


