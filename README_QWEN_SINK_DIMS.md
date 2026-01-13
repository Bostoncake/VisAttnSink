# Finding Sink Dimensions for Qwen-2.5-VL

This package provides tools to identify "sink dimensions" for the Qwen-2.5-VL model, enabling you to implement the Visual Attention Sink method in your evaluation framework.

## Quick Start

### Step 1: Find Sink Dimensions (Recommended Method)

```bash
python find_sink_dimensions_simple.py \
    --model /home/xiongyizhe/data/Qwen2.5-VL-3B-Instruct \
    --data_root /mnt/data1/xiongyizhe/data/spatial \
    --dataset blink \
    --num_samples 30 \
    --output sink_dims_qwen.json
```

**Expected output:**
```
================================================================================
RESULTS
================================================================================

Top dimensions per layer:
--------------------------------------------------------------------------------
Layer  2: 1234(15.32), 5678(14.87), 2345(13.45), ...
Layer  3: 5678(16.21), 1234(15.99), 3456(13.21), ...
...

Most consistent dimensions across layers:
--------------------------------------------------------------------------------
Dimensions appearing in >= 7 layers:
  Dim 1234: appears in 12 layers, avg_score=15.234
  Dim 5678: appears in 11 layers, avg_score=14.987

RECOMMENDED SINK DIMENSIONS:
  [1234, 5678]

For your code, use: DIM_SINK['qwen2_5-vl'] = torch.tensor([1234, 5678])
```

### Step 2: Validate Dimensions

```bash
python validate_sink_dimensions.py \
    --model /home/xiongyizhe/data/Qwen2.5-VL-3B-Instruct \
    --data_root /mnt/data1/xiongyizhe/data/spatial \
    --dimensions 1234 5678 \
    --tau 20 \
    --num_samples 10
```

**Expected output:**
```
Average Precision: 0.623
  (What % of detected sinks are actually high-attention tokens)

Average Recall: 0.445
  (What % of high-attention tokens are detected as sinks)

Interpretation:
✓ GOOD: Dimensions successfully identify high-attention tokens
  These dimensions are suitable for Visual Attention Sink
```

### Step 3: Use in Your Implementation

See the implementation guide in the main analysis for how to integrate these dimensions into your Visual Attention Sink implementation.

## Files Provided

### 1. `find_sink_dimensions_simple.py` ⭐ **Recommended**
- **Purpose**: Find sink dimensions using a straightforward approach
- **Method**: Analyzes hidden states and attention patterns
- **Runtime**: ~2-5 minutes for 20-30 samples
- **Memory**: ~8-12GB GPU for 3B model

**When to use:** First time finding dimensions, quick analysis

### 2. `find_sink_dimensions_qwen.py` 🔬 **Advanced**
- **Purpose**: More detailed analysis with PyTorch hooks
- **Method**: Captures hidden states layer-by-layer using hooks
- **Runtime**: ~5-10 minutes for 50 samples
- **Memory**: More efficient (~6-10GB GPU for 3B model)

**When to use:** Need detailed per-layer analysis, memory-constrained

### 3. `validate_sink_dimensions.py` ✓ **Validation**
- **Purpose**: Verify that found dimensions actually work
- **Method**: Checks if detected sinks correlate with high attention
- **Runtime**: ~1-2 minutes for 10 samples

**When to use:** After finding dimensions, before implementing in production

### 4. `GUIDE_FINDING_SINK_DIMENSIONS.md` 📚 **Documentation**
- Comprehensive guide explaining the entire process
- Troubleshooting tips
- Background on sink dimensions
- Integration examples

## Workflow

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: Find Dimensions                                │
│  → Run find_sink_dimensions_simple.py                   │
│  → Get recommended dimensions: [1234, 5678]             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Step 2: Validate Dimensions                            │
│  → Run validate_sink_dimensions.py                      │
│  → Check precision & recall metrics                     │
│  → Precision > 0.5 & Recall > 0.3 → Good!               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Step 3: Integrate into Your Framework                  │
│  → Use dimensions in DimProspector                      │
│  → Implement HeadFork and VARProcessor                  │
│  → Test on your evaluation tasks                        │
└─────────────────────────────────────────────────────────┘
```

## Algorithm Overview

### What Are Sink Dimensions?

In the Visual Attention Sink paper, researchers discovered that attention sink tokens (tokens receiving excessive attention) have **consistently high activation values in specific dimensions** of their hidden states.

For example:
- LLaMA-v2-7b: dimensions **2533** and **1415** (out of 4096 total)
- LLaMA-v2-13b: dimensions **2100** and **4743** (out of 5120 total)

These dimensions act as **markers** to identify problematic tokens during inference.

### How We Find Them

```python
# Simplified algorithm
for each example:
    # 1. Run model and capture hidden states + attention
    outputs = model(..., output_attentions=True, output_hidden_states=True)

    # 2. Identify tokens with highest attention (potential sinks)
    high_attn_tokens = get_top_10_percent_by_attention(outputs.attentions)

    # 3. For those tokens, apply RMS normalization to hidden states
    rms_hidden = rms_normalize(outputs.hidden_states[layer])
    sink_hidden = rms_hidden[high_attn_tokens]

    # 4. Find which dimensions have highest values
    for dim in range(hidden_dim):
        dimension_scores[dim] += max(sink_hidden[:, dim])

# 5. Select top 2 dimensions that appear consistently across layers
recommended_dims = top_2_most_consistent_dimensions(dimension_scores)
```

## Key Parameters

### Finding Dimensions

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| `--num_samples` | 20 | Number of examples to analyze | 20-50 for stable results |
| `--top_percent` | 0.1 | % of tokens considered as sinks | 0.1 (top 10%) works well |
| `--dataset` | blink | Dataset to use | Try both blink and sat2 |

### Validating Dimensions

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| `--tau` | 20.0 | Threshold for sink detection | 15-25 typical range |
| `--num_samples` | 10 | Validation samples | 10 is enough for quick check |

## Expected Results

### Good Dimensions
- **Precision**: 0.5-0.7 (50-70% of detected sinks are high-attention)
- **Recall**: 0.3-0.5 (30-50% of high-attention tokens are detected)
- **Consistency**: Appear in 25%+ of layers
- **RMS Scores**: 12-20 range

### Poor Dimensions
- **Precision**: < 0.3
- **Recall**: < 0.2
- **Consistency**: < 20% of layers
- **RMS Scores**: < 8

If you get poor results:
1. Try more samples (50-100)
2. Test on different dataset
3. Adjust `--top_percent` (try 0.05, 0.15)
4. Check data loading (are images loading correctly?)

## Model-Specific Notes

### Qwen2.5-VL Architecture
- **3B model**: 36 layers, 3584 hidden dimensions
- **7B model**: 32 layers, 4096 hidden dimensions

Different model sizes **will have different sink dimensions**. Run the analysis separately for each model size you plan to use.

### Why Not Use LLaMA Dimensions?

Sink dimensions are **architecture-specific**:
- Different models have different hidden representations
- Different dimensions encode different features
- What works for LLaMA won't work for Qwen

You **must** find dimensions specific to Qwen-2.5-VL.

## Troubleshooting

### "No consistent dimensions found"
→ Increase `--num_samples` to 50+
→ Try both blink and sat2 datasets

### "Out of memory"
→ Use `find_sink_dimensions_qwen.py` (hook-based, more efficient)
→ Reduce `--num_samples`

### "Dimensions change between runs"
→ Too few samples - increase to 50+
→ Normal to have slight variation, but top 2-3 should be stable

### "Low validation precision/recall"
→ Adjust `--tau` (try 15, 20, 25)
→ Try finding dimensions again with different `--top_percent`

## Integration Example

Once you have dimensions (e.g., `[1234, 5678]`), integrate into your code:

```python
# constants.py or config
DIM_SINK = {
    "llama-v2-7b": torch.tensor([2533, 1415]),
    "qwen2_5-vl-3b": torch.tensor([1234, 5678]),  # Your found dimensions
}

# logic.py
class DimProspector:
    def __init__(self, model_name="qwen2_5-vl-3b", tau=20):
        self.dim_indices = DIM_SINK[model_name]
        self.tau = tau

    def detect_sinks(self, hidden_states):
        # Apply RMS normalization
        rms_hidden = self.rms_normalize(hidden_states)

        # Extract specific dimensions
        rms_values = torch.stack([
            rms_hidden[:, :, idx] for idx in self.dim_indices
        ], dim=-1)

        # Threshold
        max_rms = rms_values.max(dim=-1)[0]
        sink_indices = torch.nonzero(max_rms > self.tau)[:, 1]

        return sink_indices
```

See the main implementation guide for the complete Visual Attention Sink pipeline (HeadFork, VARProcessor, etc.).

## Citation

If you use these tools, please cite the original Visual Attention Sink paper:

```bibtex
@inproceedings{visual-attention-sink-2025,
  title={See What You Are Told: Visual Attention Sink in Large Multimodal Models},
  booktitle={ICLR},
  year={2025}
}
```

## Support

For issues or questions:
1. Check `GUIDE_FINDING_SINK_DIMENSIONS.md` for detailed explanations
2. Verify your data paths and model paths are correct
3. Try the validation script to diagnose dimension quality
4. Ensure you have the required packages (see environment setup below)

## Environment Requirements

```bash
pip install torch transformers pillow numpy tqdm qwen-vl-utils
```

Tested with:
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.37+
- CUDA 11.8+ (for GPU)

## Summary

This package provides a complete pipeline to:
1. ✓ Find sink dimensions for Qwen-2.5-VL
2. ✓ Validate they work correctly
3. ✓ Integrate into Visual Attention Sink implementation

Start with `find_sink_dimensions_simple.py`, validate with `validate_sink_dimensions.py`, then integrate into your framework using the examples provided.
