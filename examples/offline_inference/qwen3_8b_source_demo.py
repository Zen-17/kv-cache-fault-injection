# SPDX-License-Identifier: Apache-2.0
"""Run a reproducible Qwen3-8B baseline from the editable vLLM source."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep the engine in this process. This avoids importing the editable source a
# second time on slow shared storage and makes Python KV-cache hooks observable.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/opt/data/data/models/Qwen3-8B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-8B from editable vLLM source and record tokens.")
    parser.add_argument("--model",
                        type=Path,
                        default=DEFAULT_MODEL,
                        help="Local Qwen3-8B model directory.")
    parser.add_argument("--prompt",
                        default="请用简洁的语言介绍一下人工智能。",
                        help="User prompt.")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=64,
                        help="Maximum number of newly generated tokens.")
    parser.add_argument("--max-model-len",
                        type=int,
                        default=2048,
                        help="Maximum context length.")
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.90,
                        help="Fraction of GPU memory available to vLLM.")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.0,
                        help="Sampling temperature; 0 gives a greedy baseline.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Model and sampling random seed.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode in the chat template.")
    parser.add_argument(
        "--allow-early-stop",
        action="store_true",
        help="Honor EOS; otherwise generate exactly --max-tokens tokens.")
    parser.add_argument("--output-json",
                        type=Path,
                        help="Optional path for the structured result.")
    return parser.parse_args()


def check_environment(model_path: Path, vllm_module: Any) -> Path:
    source_path = Path(vllm_module.__file__).resolve()
    try:
        source_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            "vLLM is not imported from this source checkout. "
            f"Expected a path under {REPO_ROOT}, got {source_path}. "
            "Activate vllm0.8.5 and reinstall the checkout in editable mode."
        ) from exc

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    return source_path


def get_gpu_name() -> str:
    """Query the GPU without initializing CUDA in the engine client process."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            text=True,
        )
        return output.splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return "unknown"


def save_result(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main() -> None:
    args = parse_args()

    # Delay heavy imports so --help remains fast on shared storage.
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    model_path = args.model.expanduser().resolve()
    source_path = check_environment(model_path, vllm)
    gpu_name = get_gpu_name()

    print(f"vLLM version: {vllm.__version__}")
    print(f"vLLM source:  {source_path}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"GPU:          {gpu_name}")
    print(f"Model:        {model_path}")

    load_start = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )
    model_load_seconds = time.perf_counter() - load_start

    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        seed=args.seed,
        ignore_eos=not args.allow_early_stop,
    )

    generation_start = time.perf_counter()
    request_output = llm.generate([formatted_prompt], sampling_params)[0]
    generation_seconds = time.perf_counter() - generation_start

    generated = request_output.outputs[0]
    token_ids = list(generated.token_ids)
    prompt_token_ids = list(request_output.prompt_token_ids or [])
    tokens_per_second = (len(token_ids) / generation_seconds
                         if generation_seconds else 0.0)

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "vllm_version": vllm.__version__,
            "vllm_source": str(source_path),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "gpu": gpu_name,
            "model": str(model_path),
        },
        "configuration": {
            "seed": args.seed,
            "dtype": "bfloat16",
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "enable_thinking": args.enable_thinking,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "ignore_eos": not args.allow_early_stop,
        },
        "input": {
            "prompt": args.prompt,
            "formatted_prompt": formatted_prompt,
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_count": len(prompt_token_ids),
        },
        "output": {
            "text": generated.text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "finish_reason": generated.finish_reason,
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "generation_seconds": generation_seconds,
            "output_tokens_per_second": tokens_per_second,
        },
    }

    print("\n=== Qwen3-8B output ===")
    print(generated.text)
    print("\n=== Token statistics ===")
    print(f"Prompt tokens:    {len(prompt_token_ids)}")
    print(f"Generated tokens: {len(token_ids)}")
    print(f"Token IDs:        {token_ids}")
    print(f"Generation time:  {generation_seconds:.3f} s")
    print(f"Throughput:       {tokens_per_second:.2f} token/s")

    if args.output_json:
        output_path = args.output_json.expanduser().resolve()
        save_result(output_path, record)
        print(f"Result JSON:      {output_path}")

    print("QWEN3_8B_SOURCE_DEMO=OK")


if __name__ == "__main__":
    main()
