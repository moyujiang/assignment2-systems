import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM

import argparse
import timeit
import json
import statistics
import einsum
from typing import List, Tuple
from dataclasses import dataclass, asdict

MODEL_SPECS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_heads": 12, "num_layers": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_heads": 16, "num_layers": 24},
    "large": {"d_model": 1280, "d_ff": 5120, "num_heads": 20, "num_layers": 36},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_heads": 25, "num_layers": 48},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_heads": 32, "num_layers": 32},
}

@dataclass
class BenchmarkResult:
    mode: str
    device: str
    model_size: str
    batch_size: int
    context_length: int
    warmup_steps: int
    measure_steps: int
    mean_seconds: float
    std_seconds: float
    all_times_seconds: List[float]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS336 A2 basic forward/backward benchmark")
    parser.add_argument("--size", choices=list(MODEL_SPECS.keys()), default="small")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="forward_backward")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()

def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()

def build_model(args: argparse.Namespace, device: torch.device) -> BasicsTransformerLM:
    specs = MODEL_SPECS[args.size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=specs["d_model"],
        d_ff=specs["d_ff"],
        num_heads=specs["num_heads"],
        num_layers=specs["num_layers"],
        rope_theta=args.rope_theta,
    )
    return model.to(device)

def make_batch(args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device, dtype=torch.long)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device, dtype=torch.long)
    return x, y

def run_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    if mode == "forward":
        with torch.no_grad():
            _ = model(x)
        return
    
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()

    if mode == "train_step":
        if optimizer is None:
            raise ValueError("Optimizer must be provided for train_step mode")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.zero_grad(set_to_none=True)

def benchmark(args: argparse.Namespace) -> BenchmarkResult:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine")
    
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(args, device)
    x, y = make_batch(args, device)

    if args.mode == "forward":
        model.eval()
    else:
        model.train()

    synchronize_if_cuda(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if args.mode == "train_step" else None

    for _ in range(args.warmup_steps):
        run_step(model, x, y, args.mode, optimizer)
        synchronize_if_cuda(device)
    
    times: List[float] = []
    for _ in range(args.measure_steps):
        t0 = timeit.default_timer()
        run_step(model, x, y, args.mode, optimizer)
        synchronize_if_cuda(device)
        t1 = timeit.default_timer()
        times.append(t1 - t0)
    
    return BenchmarkResult(
        mode=args.mode,
        device=str(device),
        model_size=args.size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mean_seconds=statistics.mean(times),
        std_seconds=statistics.pstdev(times) if len(times) > 1 else 0.0,
        all_times_seconds=times,
    )

def main() -> None:
    args = parse_args()
    result = benchmark(args)
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(
            f"[{result.mode}] size={result.model_size} device={result.device} "
            f"bs={result.batch_size} ctx={result.context_length} "
            f"mean={result.mean_seconds:.6f}s std={result.std_seconds:.6f}s"
        )

if __name__ == "__main__":
    main()