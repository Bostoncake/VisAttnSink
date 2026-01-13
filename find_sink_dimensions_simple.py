#!/usr/bin/env python3
"""
Simplified script to find sink dimensions for Qwen-2.5-VL.

This version integrates with your existing evaluation framework and provides
a quick analysis without requiring hooks.

Strategy:
1. Run model on examples and save hidden states
2. Identify visual tokens with high attention aggregation
3. Find dimensions with consistently high RMS-normalized values for those tokens

Usage:
    python find_sink_dimensions_simple.py \
        --model /home/xiongyizhe/data/Qwen2.5-VL-3B-Instruct \
        --data_root /mnt/data1/xiongyizhe/data/spatial \
        --dataset blink \
        --num_samples 20
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_examples(data_root, dataset, limit):
    """Load examples from jsonl file."""
    paths = {
        "blink": "spatial_reasoning/blink/blink_validation.jsonl",
        "sat2": "spatial_reasoning/sat2_test/sat2_test.jsonl",
    }

    ann_file = os.path.join(data_root, paths[dataset])
    examples = []

    with open(ann_file, "r") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break

            rec = json.loads(line.strip())
            prompt = rec['conversations'][0]['value']

            # Load images
            images = []
            for key in ("image_1", "image_2", "image_3", "image_4", "image"):
                if key in rec:
                    img_path = rec[key]
                    if isinstance(img_path, str):
                        # Try different path combinations
                        for base in [data_root, os.path.join(data_root, "spatial_reasoning")]:
                            full_path = os.path.join(base, img_path)
                            if os.path.exists(full_path):
                                try:
                                    img = Image.open(full_path).convert("RGB")
                                    images.append(img)
                                    break
                                except:
                                    pass

            if prompt.strip() and images:
                examples.append({"prompt": prompt, "images": images})

    return examples


def rms_normalize(tensor):
    """Apply RMS normalization."""
    variance = tensor.pow(2).mean(-1, keepdim=True)
    return tensor * torch.rsqrt(variance + 1e-6)


def analyze_hidden_states(hidden_states, attentions, image_positions, top_percent=0.1):
    """
    Analyze hidden states to find dimensions associated with high-attention tokens.

    Args:
        hidden_states: Tuple of tensors [batch, seq_len, hidden_dim] for each layer
        attentions: Tuple of tensors [batch, num_heads, seq_len, seq_len] for each layer
        image_positions: Tuple of (start_idx, length) for image tokens
        top_percent: Percentage of tokens to consider as sinks

    Returns:
        Dictionary mapping layer_idx -> list of (dimension, score) tuples
    """
    image_start, image_len = image_positions
    image_end = image_start + image_len

    layer_results = {}

    for layer_idx in range(2, len(hidden_states)):  # Skip first 2 layers
        # Get hidden states and attention for this layer
        hs = hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
        attn = attentions[layer_idx]   # [batch, num_heads, seq_len, seq_len]

        # Focus on visual tokens only
        vis_hidden = hs[:, image_start:image_end, :]  # [batch, vis_len, hidden_dim]

        # Calculate total attention received by each visual token
        # Sum across query positions and heads
        attn_to_vis = attn[:, :, :, image_start:image_end]  # [batch, heads, seq_len, vis_len]
        total_attn = attn_to_vis.sum(dim=(1, 2))  # [batch, vis_len]

        # Identify top K% tokens as potential sinks
        batch_size, vis_len, hidden_dim = vis_hidden.shape
        k = max(1, int(vis_len * top_percent))

        # Get top-k tokens per batch
        _, top_indices = torch.topk(total_attn, k=k, dim=-1)  # [batch, k]

        # Extract hidden states of sink tokens
        sink_hidden = []
        for b in range(batch_size):
            sink_h = vis_hidden[b, top_indices[b], :]  # [k, hidden_dim]
            sink_hidden.append(sink_h)

        if sink_hidden:
            sink_hidden = torch.cat(sink_hidden, dim=0)  # [batch*k, hidden_dim]

            # Apply RMS normalization
            rms_hidden = torch.abs(rms_normalize(sink_hidden))  # [batch*k, hidden_dim]

            # For each dimension, get max value across all sink tokens
            dim_scores = rms_hidden.max(dim=0)[0]  # [hidden_dim]

            # Get top 100 dimensions
            top_values, top_dims = torch.topk(dim_scores, k=min(100, hidden_dim))

            layer_results[layer_idx] = list(zip(
                top_dims.cpu().numpy().tolist(),
                top_values.cpu().numpy().tolist()
            ))

    return layer_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", default="blink", choices=["blink", "sat2"])
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--top_percent", type=float, default=0.1)
    parser.add_argument("--output", default="sink_dims_qwen.json")
    args = parser.parse_args()

    print("="*80)
    print("Finding Sink Dimensions for Qwen-2.5-VL")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
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

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    print(f"Model: {num_layers} layers, {hidden_dim} hidden dimensions")
    print()

    # Load examples
    print("Loading examples...")
    examples = load_examples(args.data_root, args.dataset, args.num_samples)
    print(f"Loaded {len(examples)} examples")
    print()

    # Accumulate dimension scores across examples
    dimension_accumulator = defaultdict(lambda: defaultdict(list))

    print("Analyzing examples...")
    for ex_idx, example in enumerate(tqdm(examples)):
        try:
            # Prepare input
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": img} for img in example["images"]] +
                          [{"type": "text", "text": example["prompt"]}]
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                                 padding=True, return_tensors="pt")
            except:
                inputs = processor(text=[text], images=example["images"],
                                 padding=True, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Find image token positions
            image_token_id = model.config.image_token_id
            img_pos = (inputs["input_ids"] == image_token_id).nonzero(as_tuple=True)

            if len(img_pos[1]) == 0:
                continue

            image_start = img_pos[1][0].item()
            image_len = len(img_pos[1])

            # Forward pass with outputs
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # Analyze this example
            layer_dims = analyze_hidden_states(
                outputs.hidden_states,
                outputs.attentions,
                (image_start, image_len),
                args.top_percent
            )

            # Accumulate results
            for layer_idx, dim_scores in layer_dims.items():
                for dim, score in dim_scores:
                    dimension_accumulator[layer_idx][dim].append(score)

        except Exception as e:
            print(f"\nError on example {ex_idx}: {e}")
            continue

    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()

    # Aggregate results
    per_layer_top = {}
    dimension_frequency = defaultdict(int)
    dimension_avg_scores = defaultdict(list)

    print("Top dimensions per layer:")
    print("-"*80)

    for layer_idx in sorted(dimension_accumulator.keys()):
        # Average scores for each dimension
        dim_avg = {}
        for dim, scores in dimension_accumulator[layer_idx].items():
            avg_score = np.mean(scores)
            dim_avg[dim] = avg_score
            dimension_frequency[dim] += 1
            dimension_avg_scores[dim].append(avg_score)

        # Sort by average score
        sorted_dims = sorted(dim_avg.items(), key=lambda x: x[1], reverse=True)[:10]
        per_layer_top[f"layer_{layer_idx}"] = [
            {"dim": dim, "score": float(score)} for dim, score in sorted_dims
        ]

        dims_str = ", ".join([f"{dim}({score:.2f})" for dim, score in sorted_dims[:5]])
        print(f"Layer {layer_idx:2d}: {dims_str}")

    print()
    print("Most consistent dimensions across layers:")
    print("-"*80)

    # Find dimensions that appear frequently across layers
    min_layers = max(3, num_layers // 4)
    consistent_dims = []

    for dim, freq in dimension_frequency.items():
        if freq >= min_layers:
            avg_score = np.mean(dimension_avg_scores[dim])
            consistent_dims.append((dim, freq, avg_score))

    consistent_dims.sort(key=lambda x: (x[1], x[2]), reverse=True)

    if consistent_dims:
        print(f"Dimensions appearing in >= {min_layers} layers:")
        for dim, freq, avg_score in consistent_dims[:20]:
            print(f"  Dim {dim:4d}: appears in {freq:2d} layers, avg_score={avg_score:.4f}")

        print()
        print("RECOMMENDED SINK DIMENSIONS:")
        recommended = [dim for dim, _, _ in consistent_dims[:2]]
        print(f"  {recommended}")
        print()
        print(f"For your code, use: DIM_SINK['qwen2_5-vl'] = torch.tensor({recommended})")
    else:
        print("No consistently high dimensions found.")
        recommended = []

    # Save results
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "num_samples": len(examples),
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "recommended_dimensions": recommended,
        "consistent_dimensions": [
            {"dim": dim, "frequency": freq, "avg_score": float(avg_score)}
            for dim, freq, avg_score in consistent_dims[:20]
        ],
        "per_layer_top": per_layer_top,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
