#!/usr/bin/env python3
"""
Script to identify sink dimensions for Qwen-2.5-VL model.

This script analyzes hidden states and attention patterns to find dimensions
that are consistently activated in "attention sink" tokens - tokens that receive
disproportionately high attention across multiple heads.

The algorithm:
1. Run inference on multiple examples with hooks to capture hidden states and attention
2. For each layer, identify tokens receiving highest attention (potential sinks)
3. Apply RMS normalization to hidden states of these high-attention tokens
4. Track which dimensions consistently have high RMS values for sink tokens
5. Rank dimensions by their correlation with sink behavior

Usage:
    python find_sink_dimensions_qwen.py \
        --model /path/to/Qwen2.5-VL-3B-Instruct \
        --data_root /path/to/data \
        --dataset blink \
        --num_samples 50 \
        --top_dims 10 \
        --output sink_dimensions.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# -----------------------------
# Data loading utilities (simplified from main eval script)
# -----------------------------

def load_records_from_file(path: str) -> List[Dict[str, Any]]:
    """Load examples from jsonl file."""
    path = str(path)
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    raise ValueError(f"Only .jsonl files supported, got: {path}")


def _maybe_open_image(v: Any, base_dir: str) -> Optional[Image.Image]:
    """Load image from path or dict."""
    if v is None:
        return None

    if isinstance(v, Image.Image):
        return v

    if isinstance(v, dict):
        if "bytes" in v and v["bytes"] is not None:
            import io
            try:
                return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
            except Exception:
                return None
        if "path" in v and v["path"]:
            v = v["path"]

    if isinstance(v, (str, Path)):
        p = Path(v)
        if not p.is_absolute():
            p1 = Path(base_dir) / p
            if p1.exists():
                p = p1
            else:
                p2 = Path(base_dir) / "spatial_reasoning" / p
                if p2.exists():
                    p = p2
        if not p.exists():
            return None
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return None

    return None


def load_examples(dataset: str, data_root: str, limit: int = -1) -> List[Dict[str, Any]]:
    """Load examples from dataset."""
    DATASET_PATHS = {
        "blink": "spatial_reasoning/blink/blink_validation.jsonl",
        "sat2": "spatial_reasoning/sat2_test/sat2_test.jsonl",
    }

    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset}")

    ann_file = os.path.join(data_root, DATASET_PATHS[dataset])
    records = load_records_from_file(ann_file)

    if limit > 0:
        records = records[:limit]

    # Parse into examples
    examples = []
    for rec in records:
        prompt = rec['conversations'][0]['value']
        images = []

        # Load images
        for key in ("image_1", "image_2", "image_3", "image_4", "image", "img"):
            if key in rec:
                img = _maybe_open_image(rec.get(key), data_root)
                if img is not None:
                    images.append(img)

        if prompt.strip() and images:
            examples.append({"prompt": prompt, "images": images})

    return examples


# -----------------------------
# Hook system for capturing hidden states and attention
# -----------------------------

class AttentionCapture:
    """Captures attention weights and hidden states during forward pass."""

    def __init__(self, model: nn.Module, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.hidden_states = {}
        self.attention_weights = {}
        self.hooks = []
        self.image_token_positions = None

    def register_hooks(self):
        """Register forward hooks on all decoder layers."""
        # For Qwen2.5-VL, the structure is:
        # model.model.layers[i] for decoder layers

        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]

            # Hook to capture hidden states (input to layer)
            def make_hidden_hook(idx):
                def hook(module, input, output):
                    # input[0] is the hidden states
                    hidden = input[0].detach()
                    self.hidden_states[idx] = hidden
                return hook

            h1 = layer.register_forward_hook(make_hidden_hook(layer_idx))
            self.hooks.append(h1)

            # Hook to capture attention weights (after softmax)
            attn_layer = layer.self_attn

            def make_attn_hook(idx):
                def hook(module, input, output):
                    # For Qwen2.5-VL with output_attentions=True
                    # output is (attn_output, attn_weights) or just attn_output
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            self.attention_weights[idx] = attn_weights.detach()
                return hook

            h2 = attn_layer.register_forward_hook(make_attn_hook(layer_idx))
            self.hooks.append(h2)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        """Clear captured data."""
        self.hidden_states = {}
        self.attention_weights = {}
        self.image_token_positions = None


# -----------------------------
# Sink dimension analysis
# -----------------------------

def rms_normalize(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Apply RMS normalization to hidden states.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]

    Returns:
        Normalized hidden states with same shape
    """
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    normalized = hidden_states * torch.rsqrt(variance + 1e-6)
    return torch.abs(normalized)


def identify_sink_tokens(
    attention_weights: torch.Tensor,
    image_start: int,
    image_len: int,
    top_k_percent: float = 0.1
) -> torch.Tensor:
    """
    Identify potential sink tokens based on attention patterns.

    A token is considered a sink if it receives high attention across many heads.
    We focus on visual tokens only.

    Args:
        attention_weights: [batch, num_heads, seq_len, seq_len]
        image_start: Starting position of image tokens
        image_len: Number of image tokens
        top_k_percent: Consider top K% of tokens by attention as sinks

    Returns:
        sink_indices: Tensor of token indices (relative to full sequence)
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape

    # Focus on visual tokens only
    vis_end = image_start + image_len

    # Sum attention received by each token across all query positions and heads
    # attention_weights[:, :, :, i] gives attention TO token i
    attention_received = attention_weights[:, :, :, image_start:vis_end].sum(dim=(1, 2))  # [batch, vis_len]

    # Identify top K% tokens
    k = max(1, int(image_len * top_k_percent))
    _, top_indices = torch.topk(attention_received, k=k, dim=-1)  # [batch, k]

    # Convert to absolute positions
    sink_indices = top_indices + image_start  # [batch, k]

    return sink_indices


def analyze_sink_dimensions(
    hidden_states: torch.Tensor,
    sink_indices: torch.Tensor,
    top_n: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analyze which dimensions have high RMS values for sink tokens.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        sink_indices: [batch, num_sinks] - indices of sink tokens
        top_n: Number of top dimensions to return

    Returns:
        top_dims: [top_n] - dimension indices with highest RMS values
        rms_values: [top_n] - corresponding RMS values
    """
    batch_size = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[2]

    # Apply RMS normalization
    rms_hidden = rms_normalize(hidden_states)  # [batch, seq_len, hidden_dim]

    # Extract sink token hidden states
    # For each batch, gather sink tokens
    all_sink_hidden = []
    for b in range(batch_size):
        sink_inds = sink_indices[b]  # [num_sinks]
        sink_hidden = rms_hidden[b, sink_inds, :]  # [num_sinks, hidden_dim]
        all_sink_hidden.append(sink_hidden)

    all_sink_hidden = torch.cat(all_sink_hidden, dim=0)  # [total_sinks, hidden_dim]

    # For each dimension, compute max value across all sink tokens
    max_values, _ = torch.max(all_sink_hidden, dim=0)  # [hidden_dim]

    # Get top N dimensions
    top_values, top_dims = torch.topk(max_values, k=min(top_n, hidden_dim))

    return top_dims, top_values


class DimensionAnalyzer:
    """Accumulates statistics across multiple examples to find sink dimensions."""

    def __init__(self, num_layers: int, hidden_dim: int):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # For each layer, track dimension scores
        self.dimension_scores = defaultdict(lambda: np.zeros(hidden_dim, dtype=np.float32))
        self.dimension_counts = defaultdict(lambda: np.zeros(hidden_dim, dtype=np.int32))

    def add_example(
        self,
        layer_idx: int,
        top_dims: torch.Tensor,
        top_values: torch.Tensor
    ):
        """Add dimension statistics from one example."""
        dims_np = top_dims.cpu().numpy()
        values_np = top_values.cpu().numpy()

        for dim, val in zip(dims_np, values_np):
            self.dimension_scores[layer_idx][dim] += val
            self.dimension_counts[layer_idx][dim] += 1

    def get_top_dimensions(self, layer_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get top K dimensions for a specific layer."""
        scores = self.dimension_scores[layer_idx]
        counts = self.dimension_counts[layer_idx]

        # Average score for each dimension
        avg_scores = np.divide(
            scores,
            counts,
            out=np.zeros_like(scores),
            where=counts > 0
        )

        # Get top K
        top_indices = np.argsort(avg_scores)[::-1][:top_k]
        results = [(int(idx), float(avg_scores[idx])) for idx in top_indices]

        return results

    def get_consistent_dimensions(self, min_layers: int = 3, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get dimensions that are consistently in top ranks across multiple layers.

        Args:
            min_layers: Minimum number of layers where dimension should appear
            top_k: Number of dimensions to return

        Returns:
            List of (dimension_idx, consistency_score) tuples
        """
        # For each dimension, count how many layers it appears in top 50
        dimension_layer_counts = defaultdict(int)
        dimension_total_scores = defaultdict(float)

        for layer_idx in range(self.num_layers):
            top_50 = self.get_top_dimensions(layer_idx, top_k=50)
            for dim, score in top_50:
                dimension_layer_counts[dim] += 1
                dimension_total_scores[dim] += score

        # Filter by minimum layer count
        consistent_dims = [
            (dim, dimension_total_scores[dim] / dimension_layer_counts[dim])
            for dim, count in dimension_layer_counts.items()
            if count >= min_layers
        ]

        # Sort by average score
        consistent_dims.sort(key=lambda x: x[1], reverse=True)

        return consistent_dims[:top_k]


# -----------------------------
# Main analysis pipeline
# -----------------------------

def analyze_model(
    model: nn.Module,
    processor: Any,
    examples: List[Dict[str, Any]],
    num_layers: int,
    hidden_dim: int,
    top_k_percent: float = 0.1,
    device: str = "cuda"
) -> DimensionAnalyzer:
    """
    Run analysis on multiple examples to identify sink dimensions.

    Args:
        model: The VLM model
        processor: The model processor
        examples: List of examples with prompts and images
        num_layers: Number of decoder layers
        hidden_dim: Hidden dimension size
        top_k_percent: Percentage of tokens to consider as sinks
        device: Device to run on

    Returns:
        DimensionAnalyzer with accumulated statistics
    """
    analyzer = DimensionAnalyzer(num_layers, hidden_dim)
    capture = AttentionCapture(model, num_layers)

    # Register hooks
    capture.register_hooks()

    try:
        for ex_idx, example in enumerate(tqdm(examples, desc="Analyzing examples")):
            try:
                # Prepare inputs
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img}
                        for img in example["images"]
                    ] + [{"type": "text", "text": example["prompt"]}]
                }]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Try to use qwen_vl_utils if available
                try:
                    from qwen_vl_utils import process_vision_info
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                except ImportError:
                    inputs = processor(
                        text=[text],
                        images=example["images"],
                        padding=True,
                        return_tensors="pt",
                    )

                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Find image token positions
                image_token_id = model.config.image_token_id
                image_positions = (inputs["input_ids"] == image_token_id).nonzero(as_tuple=True)

                if len(image_positions[1]) == 0:
                    print(f"Warning: No image tokens found in example {ex_idx}")
                    continue

                image_start = image_positions[1][0].item()
                image_len = len(image_positions[1])

                # Clear previous captures
                capture.clear()
                capture.image_token_positions = (image_start, image_len)

                # Forward pass (prefill only, no generation)
                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                # Analyze each layer
                for layer_idx in range(num_layers):
                    # Skip first 2 layers (as in original paper)
                    if layer_idx < 2:
                        continue

                    # Get hidden states and attention for this layer
                    if layer_idx not in capture.hidden_states:
                        continue
                    if layer_idx not in capture.attention_weights:
                        continue

                    hidden = capture.hidden_states[layer_idx]
                    attn = capture.attention_weights[layer_idx]

                    # Identify sink tokens
                    sink_indices = identify_sink_tokens(
                        attn, image_start, image_len, top_k_percent
                    )

                    # Analyze dimensions
                    top_dims, top_values = analyze_sink_dimensions(
                        hidden, sink_indices, top_n=100
                    )

                    # Add to analyzer
                    analyzer.add_example(layer_idx, top_dims, top_values)

            except Exception as e:
                print(f"Error processing example {ex_idx}: {e}")
                continue

    finally:
        # Clean up hooks
        capture.remove_hooks()

    return analyzer


def main():
    parser = argparse.ArgumentParser(description="Find sink dimensions for Qwen-2.5-VL")
    parser.add_argument("--model", required=True, help="Path to Qwen2.5-VL model")
    parser.add_argument("--data_root", required=True, help="Root directory for datasets")
    parser.add_argument("--dataset", default="blink", choices=["blink", "sat2"],
                        help="Dataset to use for analysis")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to analyze")
    parser.add_argument("--top_dims", type=int, default=10,
                        help="Number of top dimensions to report")
    parser.add_argument("--top_k_percent", type=float, default=0.1,
                        help="Percentage of tokens to consider as sinks")
    parser.add_argument("--output", default="sink_dimensions_qwen.json",
                        help="Output file for results")
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()

    print("=" * 80)
    print("Sink Dimension Finder for Qwen-2.5-VL")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    print(f"Model has {num_layers} layers with hidden dimension {hidden_dim}")
    print()

    # Load examples
    print(f"Loading examples from {args.dataset}...")
    examples = load_examples(args.dataset, args.data_root, limit=args.num_samples)
    print(f"Loaded {len(examples)} examples")
    print()

    # Run analysis
    print("Running analysis...")
    analyzer = analyze_model(
        model=model,
        processor=processor,
        examples=examples,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        top_k_percent=args.top_k_percent,
        device=args.device,
    )
    print()

    # Report results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Per-layer top dimensions
    print("Top dimensions per layer:")
    print("-" * 80)
    per_layer_results = {}
    for layer_idx in range(2, num_layers):  # Skip first 2 layers
        top_dims = analyzer.get_top_dimensions(layer_idx, top_k=args.top_dims)
        if top_dims:
            dims_str = ", ".join([f"{dim}({score:.2f})" for dim, score in top_dims[:5]])
            print(f"Layer {layer_idx:2d}: {dims_str}")
            per_layer_results[f"layer_{layer_idx}"] = [
                {"dim": dim, "score": score} for dim, score in top_dims
            ]
    print()

    # Consistent dimensions across layers
    print("Most consistent dimensions across layers:")
    print("-" * 80)
    consistent = analyzer.get_consistent_dimensions(
        min_layers=max(3, num_layers // 4),
        top_k=args.top_dims
    )

    if consistent:
        print(f"Dimensions appearing in at least {max(3, num_layers // 4)} layers:")
        for dim, score in consistent:
            print(f"  Dimension {dim:4d}: avg_score={score:.4f}")
        print()
        print("RECOMMENDED SINK DIMENSIONS:")
        recommended = [dim for dim, _ in consistent[:2]]
        print(f"  {recommended}")
    else:
        print("No consistently high dimensions found across layers.")
        print("Try increasing --num_samples or adjusting --top_k_percent")
        recommended = []

    # Save results
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "num_samples": len(examples),
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "top_k_percent": args.top_k_percent,
        "recommended_dimensions": recommended,
        "consistent_dimensions": [
            {"dim": dim, "avg_score": score} for dim, score in consistent
        ],
        "per_layer_top_dimensions": per_layer_results,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {args.output}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
