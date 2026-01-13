# Guide: Finding Sink Dimensions for Qwen-2.5-VL

This guide explains how to identify "sink dimensions" for the Qwen-2.5-VL model to implement the Visual Attention Sink method in your evaluation framework.

## Background

The Visual Attention Sink paper identified specific hidden dimensions that are highly activated in "attention sink tokens" - tokens that absorb excessive attention but don't contribute meaningful information. For LLaMA-v2 models:

- **LLaMA-v2-7b**: dimensions [2533, 1415]
- **LLaMA-v2-13b**: dimensions [2100, 4743]

These dimensions are model-specific and need to be identified empirically for Qwen-2.5-VL.

## What Are Sink Dimensions?

Sink dimensions are specific indices in the hidden state vector (out of 3584 dimensions for Qwen2.5-VL-3B or 4096 for 7B) that:

1. Have consistently **high RMS-normalized values** for tokens that receive disproportionate attention
2. Are **consistent across layers** (appear in multiple decoder layers)
3. Can be used as **indicators** to identify sink tokens during inference

## Method Overview

The algorithm to find sink dimensions:

```
For each example:
  1. Run forward pass, capture hidden states and attention weights
  2. Identify visual tokens with highest attention (top 10% → potential sinks)
  3. Apply RMS normalization to their hidden states
  4. For each dimension, record the max RMS value across sink tokens

Aggregate across all examples:
  5. Find dimensions with consistently high scores
  6. Select top 2 dimensions that appear frequently across layers
```

## Two Scripts Provided

### Option 1: `find_sink_dimensions_simple.py` (Recommended)

**Pros:**
- Simpler, easier to understand
- Uses model's built-in `output_hidden_states=True`
- No complex hooking mechanism
- Faster to run

**Cons:**
- Requires model to support `output_hidden_states` and `output_attentions`
- Uses more memory (stores all hidden states)

**Usage:**
```bash
python find_sink_dimensions_simple.py \
    --model /home/xiongyizhe/data/Qwen2.5-VL-3B-Instruct \
    --data_root /mnt/data1/xiongyizhe/data/spatial \
    --dataset blink \
    --num_samples 20 \
    --output sink_dims_qwen.json
```

### Option 2: `find_sink_dimensions_qwen.py` (Advanced)

**Pros:**
- More memory efficient (uses hooks)
- More detailed per-layer analysis
- Can work with models that don't return hidden states

**Cons:**
- More complex code
- Requires understanding of PyTorch hooks
- Slightly slower due to hook overhead

**Usage:**
```bash
python find_sink_dimensions_qwen.py \
    --model /home/xiongyizhe/data/Qwen2.5-VL-3B-Instruct \
    --data_root /mnt/data1/xiongyizhe/data/spatial \
    --dataset blink \
    --num_samples 50 \
    --top_dims 10 \
    --output sink_dimensions_qwen.json
```

## Step-by-Step Instructions

### Step 1: Run the Analysis

Choose one of the scripts above and run it. I recommend starting with the simple version:

```bash
cd /home/user/VisAttnSink

python find_sink_dimensions_simple.py \
    --model /home/xiongyizhe/data/Qwen2.5-VL-3B-Instruct \
    --data_root /mnt/data1/xiongyizhe/data/spatial \
    --dataset blink \
    --num_samples 30 \
    --top_percent 0.1 \
    --output sink_dims_qwen_3b.json
```

**Parameters:**
- `--num_samples`: Number of examples to analyze (20-50 is usually enough)
- `--top_percent`: What percentage of tokens to consider as sinks (0.1 = top 10%)
- `--dataset`: Use `blink` or `sat2` (both are spatial reasoning tasks)

### Step 2: Examine the Results

The script will output:

```
================================================================================
RESULTS
================================================================================

Top dimensions per layer:
--------------------------------------------------------------------------------
Layer  2: 1234(15.32), 5678(14.87), ...
Layer  3: 5678(16.21), 1234(15.99), ...
...

Most consistent dimensions across layers:
--------------------------------------------------------------------------------
Dimensions appearing in >= 7 layers:
  Dim 1234: appears in 12 layers, avg_score=15.234
  Dim 5678: appears in 11 layers, avg_score=14.987
  ...

RECOMMENDED SINK DIMENSIONS:
  [1234, 5678]

For your code, use: DIM_SINK['qwen2_5-vl'] = torch.tensor([1234, 5678])
```

The JSON output file will contain:
```json
{
  "model": "/path/to/model",
  "recommended_dimensions": [1234, 5678],
  "consistent_dimensions": [
    {"dim": 1234, "frequency": 12, "avg_score": 15.234},
    {"dim": 5678, "frequency": 11, "avg_score": 14.987},
    ...
  ],
  "per_layer_top": {
    "layer_2": [{"dim": 1234, "score": 15.32}, ...],
    ...
  }
}
```

### Step 3: Verify the Dimensions

**Important**: The dimensions should be:
1. **Consistent**: Appear in at least 25-30% of layers
2. **High scores**: RMS values > 10 (typically 12-20)
3. **Stable**: Running the script multiple times should give similar results

If you get inconsistent results, try:
- Increasing `--num_samples` to 50-100
- Testing on both `blink` and `sat2` datasets
- Adjusting `--top_percent` (try 0.05, 0.1, 0.15)

### Step 4: Test with Different Model Sizes

If you're using different Qwen2.5-VL sizes, run the analysis for each:

```bash
# For 3B model
python find_sink_dimensions_simple.py \
    --model /path/to/Qwen2.5-VL-3B-Instruct \
    --output sink_dims_qwen_3b.json

# For 7B model
python find_sink_dimensions_simple.py \
    --model /path/to/Qwen2.5-VL-7B-Instruct \
    --output sink_dims_qwen_7b.json
```

Different model sizes will likely have different sink dimensions.

## Understanding the Output

### Per-Layer Results

Shows top dimensions for each layer individually:
```
Layer  2: 1234(15.32), 5678(14.87), 2345(13.45), ...
```
- Numbers before parentheses are dimension indices
- Numbers in parentheses are RMS scores

**What to look for:**
- Do the same dimensions appear across multiple layers?
- Are there 2-3 dimensions that consistently rank in top 5?

### Consistent Dimensions

Dimensions ranked by how often they appear in top 100 across layers:
```
Dim 1234: appears in 12 layers, avg_score=15.234
```

**What to look for:**
- Look for dimensions appearing in ≥25% of layers
- Higher average score is better
- Top 2-3 dimensions are usually sufficient

### Recommended Dimensions

The script automatically recommends the top 2 most consistent dimensions. These are what you'll use in your implementation.

## Using the Results in Your Code

Once you have the sink dimensions (e.g., [1234, 5678]), you'll use them in your Visual Attention Sink implementation:

### Example Integration

```python
# In your logic.py or configuration file:

DIM_SINK = {
    "llama-v2-7b": torch.tensor([2533, 1415]),
    "llama-v2-13b": torch.tensor([2100, 4743]),
    "qwen2_5-vl-3b": torch.tensor([1234, 5678]),  # Your found dimensions
}

class DimProspector:
    def __init__(self, model_name, tau=20):
        self.dim_indices = DIM_SINK[model_name]
        self.tau = tau

    def run_logic(self, hidden_states):
        """Identify sink tokens based on dimensions."""
        # Apply RMS normalization
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        rms_norm_hs = hidden_states * torch.rsqrt(variance + 1e-6)
        rms_norm_hs = torch.abs(rms_norm_hs)

        # Extract specific dimensions
        rms_values = torch.stack([
            rms_norm_hs[:, :, idx] for idx in self.dim_indices
        ], dim=-1)

        # Find tokens exceeding threshold
        max_rms = torch.max(rms_values, dim=-1)[0]
        sink_indices = torch.nonzero(max_rms > self.tau)[:, 1]

        return sink_indices
```

## Troubleshooting

### Issue: No consistent dimensions found

**Solutions:**
1. Increase `--num_samples` to 50-100
2. Try different `--top_percent` values (0.05, 0.10, 0.15)
3. Use a different dataset (try both blink and sat2)
4. Check if images are loading correctly (script should show "Loaded N examples")

### Issue: Dimensions change between runs

**Likely cause:** Too few samples

**Solution:** Increase to 50+ samples for stability

### Issue: Out of memory

**Solutions:**
1. Reduce `--num_samples`
2. Use the advanced script with hooks (more memory efficient)
3. Use a machine with more GPU memory
4. Set `torch_dtype=torch.float16` (already default)

### Issue: Script crashes on forward pass

**Solutions:**
1. Check that images are loading correctly
2. Verify model path is correct
3. Ensure you have transformers version ≥4.37
4. Try with `--num_samples 1` first to debug

## Expected Runtime

- **Simple script**: ~2-5 minutes for 20 samples on A100
- **Advanced script**: ~5-10 minutes for 50 samples on A100

Memory usage:
- 3B model: ~8-12GB GPU memory
- 7B model: ~16-24GB GPU memory

## Next Steps

After finding the sink dimensions:

1. **Implement DimProspector** - Use dimensions to detect sink tokens
2. **Implement HeadFork** - Identify heads needing modification
3. **Implement VARProcessor** - Redistribute attention weights
4. **Test on your evaluation** - Compare performance with/without Visual Attention Sink

See the main Visual Attention Sink implementation guide for details on implementing the full pipeline.

## Questions?

Common questions:

**Q: How many samples do I need?**
A: 20-30 is usually enough, but 50+ gives more stable results.

**Q: Should dimensions be the same across model sizes?**
A: No, different model sizes (3B vs 7B) will have different dimensions.

**Q: What if I get more than 2 dimensions?**
A: The paper uses 2, but you can try 3-4. More isn't always better.

**Q: How do I know if the dimensions are "good"?**
A: They should (1) appear in 25%+ of layers, (2) have high scores (>10), and (3) be stable across runs.

**Q: Can I use dimensions from LLaMA for Qwen?**
A: No! Dimensions are model-architecture specific. You must find them for each model family.
