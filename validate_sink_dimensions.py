#!/usr/bin/env python3
"""
Validate that found sink dimensions actually identify high-attention tokens.

This script checks if tokens identified using the sink dimensions actually
receive high attention, validating that the dimensions are meaningful.

Usage:
    python validate_sink_dimensions.py \
        --model /path/to/Qwen2.5-VL \
        --data_root /path/to/data \
        --dimensions 1234 5678 \
        --tau 20 \
        --num_samples 10
"""

import argparse
import json
import os

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

            images = []
            for key in ("image_1", "image_2", "image_3", "image_4", "image"):
                if key in rec:
                    img_path = rec[key]
                    if isinstance(img_path, str):
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


def detect_sinks_with_dimensions(hidden_states, dim_indices, tau):
    """
    Detect sink tokens using specified dimensions (DimProspector logic).

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        dim_indices: List of dimension indices to check
        tau: Threshold for sink detection

    Returns:
        sink_indices: Tensor of token indices
    """
    # Apply RMS normalization
    rms_norm_hs = torch.abs(rms_normalize(hidden_states))

    # Extract specific dimensions
    rms_values = torch.stack([rms_norm_hs[:, :, idx] for idx in dim_indices], dim=-1)

    # Find tokens exceeding threshold
    max_rms = torch.max(rms_values, dim=-1)[0]  # [batch, seq_len]
    sink_indices = torch.nonzero(max_rms > tau)[:, 1]  # Get token indices

    return sink_indices, max_rms


def calculate_attention_received(attention_weights, image_start, image_len):
    """
    Calculate total attention received by each visual token.

    Args:
        attention_weights: [batch, num_heads, seq_len, seq_len]
        image_start: Starting position of image tokens
        image_len: Number of image tokens

    Returns:
        attention_per_token: [batch, vis_len] - attention received by each visual token
    """
    image_end = image_start + image_len

    # Sum attention TO each visual token across all query positions and heads
    attn_to_vis = attention_weights[:, :, :, image_start:image_end]  # [batch, heads, seq_len, vis_len]
    total_attention = attn_to_vis.sum(dim=(1, 2))  # [batch, vis_len]

    return total_attention


def validate_dimensions(
    model, processor, examples, dim_indices, tau, device
):
    """
    Validate that dimensions successfully identify high-attention tokens.

    Returns validation metrics.
    """
    num_layers = model.config.num_hidden_layers

    # Track metrics
    metrics = {
        "precision": [],  # % of detected sinks that are in top 10% attention
        "recall": [],     # % of top 10% attention tokens that are detected as sinks
        "avg_attention_of_sinks": [],  # Average attention rank of detected sinks
    }

    print(f"Validating dimensions {dim_indices} with tau={tau}")
    print("-" * 80)

    for ex_idx, example in enumerate(tqdm(examples, desc="Validating")):
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

            # Forward pass
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # Validate for middle layers (skip first 2 and last 1)
            for layer_idx in range(2, num_layers - 1):
                hidden = outputs.hidden_states[layer_idx]
                attn = outputs.attentions[layer_idx]

                # Detect sinks using dimensions
                sink_indices, rms_scores = detect_sinks_with_dimensions(
                    hidden, dim_indices, tau
                )

                # Filter to visual tokens only
                vis_sink_mask = (sink_indices >= image_start) & (sink_indices < image_start + image_len)
                vis_sink_indices = sink_indices[vis_sink_mask] - image_start

                if len(vis_sink_indices) == 0:
                    continue

                # Calculate actual attention received
                attn_received = calculate_attention_received(attn, image_start, image_len)
                attn_received = attn_received[0]  # [vis_len]

                # Identify ground truth top 10% attention tokens
                k = max(1, int(image_len * 0.1))
                top_k_indices = torch.topk(attn_received, k=k)[1]

                # Calculate metrics
                detected_set = set(vis_sink_indices.cpu().numpy().tolist())
                ground_truth_set = set(top_k_indices.cpu().numpy().tolist())

                if len(detected_set) > 0:
                    true_positives = len(detected_set & ground_truth_set)
                    precision = true_positives / len(detected_set)
                    recall = true_positives / len(ground_truth_set) if len(ground_truth_set) > 0 else 0

                    metrics["precision"].append(precision)
                    metrics["recall"].append(recall)

                    # Calculate average attention percentile of detected sinks
                    if len(vis_sink_indices) > 0:
                        sink_attentions = attn_received[vis_sink_indices]
                        avg_attention = sink_attentions.mean().item()
                        metrics["avg_attention_of_sinks"].append(avg_attention)

        except Exception as e:
            print(f"\nError on example {ex_idx}: {e}")
            continue

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", default="blink", choices=["blink", "sat2"])
    parser.add_argument("--dimensions", nargs="+", type=int, required=True,
                       help="Dimension indices to validate (e.g., --dimensions 1234 5678)")
    parser.add_argument("--tau", type=float, default=20.0,
                       help="Threshold for sink detection")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    print("="*80)
    print("Validating Sink Dimensions")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Dimensions: {args.dimensions}")
    print(f"Tau: {args.tau}")
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

    # Load examples
    print("Loading examples...")
    examples = load_examples(args.data_root, args.dataset, args.num_samples)
    print(f"Loaded {len(examples)} examples")
    print()

    # Validate
    metrics = validate_dimensions(
        model, processor, examples,
        dim_indices=args.dimensions,
        tau=args.tau,
        device=device
    )

    # Report results
    print()
    print("="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print()

    if metrics["precision"]:
        avg_precision = np.mean(metrics["precision"])
        avg_recall = np.mean(metrics["recall"])
        avg_attn = np.mean(metrics["avg_attention_of_sinks"])

        print(f"Average Precision: {avg_precision:.3f}")
        print(f"  (What % of detected sinks are actually high-attention tokens)")
        print()
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"  (What % of high-attention tokens are detected as sinks)")
        print()
        print(f"Average Attention of Detected Sinks: {avg_attn:.4f}")
        print(f"  (Higher is better - detected sinks should receive high attention)")
        print()

        # Interpretation
        print("Interpretation:")
        print("-" * 80)
        if avg_precision > 0.5 and avg_recall > 0.3:
            print("✓ GOOD: Dimensions successfully identify high-attention tokens")
            print("  These dimensions are suitable for Visual Attention Sink")
        elif avg_precision > 0.3:
            print("△ OKAY: Dimensions capture some high-attention tokens")
            print("  Consider adjusting tau or trying different dimensions")
        else:
            print("✗ POOR: Dimensions do not reliably identify high-attention tokens")
            print("  These dimensions may not be suitable - try finding new ones")

        print()
        print("Recommendations:")
        print("-" * 80)
        if avg_precision < 0.5:
            print("- Try adjusting tau (current: {:.1f})".format(args.tau))
            print("  - Higher tau: fewer but more confident detections")
            print("  - Lower tau: more detections but less precise")
        if avg_recall < 0.3:
            print("- Consider finding additional dimensions")
            print("- Increase the number of dimensions from 2 to 3-4")
        if avg_precision > 0.5 and avg_recall > 0.3:
            print("✓ Dimensions validated successfully!")
            print("✓ You can use these in your Visual Attention Sink implementation")

    else:
        print("No metrics collected - check if examples processed correctly")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
