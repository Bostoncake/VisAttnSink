# ============================================================================
# Visual Attention Redistribution (VAR) / Visual Attention Sink Implementation
# ============================================================================
"""
Implementation of the Visual Attention Sink method from:
"See What You Are Told: Visual Attention Sink in Large Multimodal Models" (ICLR 2025)

This implementation includes three main components:
1. DimProspector: Identifies sink tokens based on hidden state dimensions
2. HeadFork: Selects attention heads that need modification
3. VARProcessor: Redistributes attention weights away from sink tokens

The method operates in three stages:
Stage 1 (DimProspector): Detect sink tokens using RMS-normalized hidden states
Stage 2 (HeadFork): Identify heads with meaningful visual attention but not sink-dominated
Stage 3 (VARProcessor): Reduce sink attention and redistribute to non-sink visual tokens
"""

import torch
from collections import defaultdict
from typing import Dict, List, Optional
import copy


# Model-specific sink dimensions (found empirically)
# These need to be identified for each model using the find_sink_dimensions scripts
DIM_SINK = {
    "llama-v2-7b": torch.tensor([2533, 1415]),
    "llama-v2-13b": torch.tensor([2100, 4743]),
    # Add Qwen2.5-VL dimensions after running find_sink_dimensions_simple.py
    # "qwen2_5-vl-3b": torch.tensor([YOUR_DIMS_HERE]),
    # "qwen2_5-vl-7b": torch.tensor([YOUR_DIMS_HERE]),
}


class DefaultConfig:
    """Default configuration for VAR state management."""
    model_config = {
        "model_name": "qwen2_5-vl",
        "config": None,
        "num_hidden_layers": None,
        "num_attention_heads": None,
        "hidden_size": None,
    }

    state = {
        "logic_flag": {},
        "vis_sink_token_idx": defaultdict(list),
        "vis_non_sink_token_idx": defaultdict(list),
    }

    metadata = {
        "vis_len": 0,
        "image_token_positions": None,  # (start_idx, length)
        "output_token_count": -1,
        "current_decoder_layer": 0,
    }


class StashEngine:
    """Base class for state management across VAR components."""
    default_config = DefaultConfig()
    model_config = default_config.model_config
    state = default_config.state
    metadata = default_config.metadata

    @classmethod
    def activate(cls):
        cls.set_flag(True)

    @classmethod
    def export_model_config(cls, config):
        cls.model_config['config'] = config
        cls.model_config["num_hidden_layers"] = config.num_hidden_layers
        cls.model_config["num_attention_heads"] = config.num_attention_heads
        cls.model_config["hidden_size"] = config.hidden_size

    @classmethod
    def set_flag(cls, flag_value=True):
        cls.state["logic_flag"][cls.__name__] = flag_value

    @classmethod
    def _flag(cls, name=None):
        if name is not None:
            return cls.state["logic_flag"].get(name, False)
        return cls.state["logic_flag"].get(cls.__name__, False)

    @classmethod
    def clear(cls):
        _default_config = DefaultConfig()
        cls.model_config = _default_config.model_config
        cls.state = _default_config.state
        cls.metadata = _default_config.metadata


class MetadataStation(StashEngine):
    """Tracks metadata about image tokens and generation state."""

    @classmethod
    def set_image_positions(cls, start_idx: int, length: int):
        """Set the position and length of image tokens in the sequence."""
        cls.__base__.metadata["image_token_positions"] = (start_idx, length)
        cls.__base__.metadata["vis_len"] = length

    @classmethod
    def get_image_positions(cls):
        """Get image token positions as (start_idx, length)."""
        return cls.__base__.metadata.get("image_token_positions")

    @classmethod
    def count_output_token(cls):
        """Increment output token counter (for tracking prefill vs decode)."""
        cls.__base__.metadata["output_token_count"] += 1

    @classmethod
    def get_output_token_count(cls):
        """Get current output token count."""
        return cls.__base__.metadata["output_token_count"]

    @classmethod
    def set_current_layer(cls, layer_idx: int):
        """Set the current decoder layer index."""
        cls.__base__.metadata["current_decoder_layer"] = layer_idx


class DimProspector(StashEngine):
    """
    Stage 1: Identify sink tokens based on hidden state dimensions.

    Sink tokens are identified by checking if their RMS-normalized hidden states
    exceed a threshold in specific dimensions. These dimensions are model-specific
    and should be found using the find_sink_dimensions scripts.
    """

    # Configuration
    tau = 20  # Threshold for sink detection
    dim_sink = None  # Dimension indices to check
    sink_select_layers = None  # Which layers to run on (e.g., range(2, num_layers-1))

    # State: stores sink token indices per layer
    indices = defaultdict(list)

    @classmethod
    def activate(cls, tau=20, dim_indices=None, model_name="qwen2_5-vl"):
        """
        Activate DimProspector with specified parameters.

        Args:
            tau: Threshold for sink detection (higher = fewer sinks)
            dim_indices: List/tensor of dimension indices to check
            model_name: Model name for looking up default dimensions
        """
        super().activate()
        cls.tau = tau

        if dim_indices is not None:
            if isinstance(dim_indices, list):
                dim_indices = torch.tensor(dim_indices)
            cls.dim_sink = dim_indices
        elif model_name in DIM_SINK:
            cls.dim_sink = DIM_SINK[model_name]
        else:
            raise ValueError(
                f"No sink dimensions found for model '{model_name}'. "
                "Please run find_sink_dimensions_simple.py to identify them."
            )

    @classmethod
    def run_logic(cls, hidden_states: torch.Tensor, layer: int):
        """
        Detect sink tokens for a given layer.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            layer: Current layer index
        """
        if cls.dim_sink is None:
            return

        # Move dim_sink to same device as hidden_states
        if cls.dim_sink.device != hidden_states.device:
            cls.dim_sink = cls.dim_sink.to(hidden_states.device)

        # Apply RMS normalization
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        rms_norm_hs = hidden_states * torch.rsqrt(variance + 1e-6)
        rms_norm_hs = torch.abs(rms_norm_hs)

        # Extract specific dimensions
        rms_values = torch.stack([
            rms_norm_hs[:, :, idx] for idx in cls.dim_sink
        ], dim=-1)  # [batch, seq_len, num_dims]

        # Find tokens where max RMS value exceeds threshold
        max_rms_values = torch.max(rms_values, dim=-1)[0]  # [batch, seq_len]
        indices = torch.nonzero(max_rms_values > cls.tau)[:, 1]  # Get token indices

        # Store for this layer
        cls.__base__.indices[layer] = indices


class HeadFork(StashEngine):
    """
    Stage 2: Identify attention heads that need modification.

    A head is selected for modification if:
    1. It pays meaningful attention to visual tokens (summation >= summ)
    2. But sink tokens don't dominate (portion <= rho)
    """

    # Configuration
    rho = 0.5   # Max allowed portion of attention to sink tokens
    summ = 0.2  # Min total attention to visual region

    # State: stores coordinates of heads to modify per layer
    forked_head = defaultdict(list)

    @classmethod
    def activate(cls, rho=0.5, summ=0.2):
        """
        Activate HeadFork with specified parameters.

        Args:
            rho: Maximum allowed portion of attention to sink tokens
            summ: Minimum total attention to visual region
        """
        super().activate()
        cls.rho = rho
        cls.summ = summ

    @classmethod
    def run_logic(cls, attn_weights: torch.Tensor, layer_idx: int):
        """
        Identify heads needing modification for a given layer.

        Args:
            attn_weights: [batch, num_heads, seq_len, seq_len] - attention weights after softmax
            layer_idx: Current layer index
        """
        # Get image token positions
        img_pos = MetadataStation.get_image_positions()
        if img_pos is None:
            return

        image_start, image_len = img_pos

        # Get sink indices for this layer
        sink_inds = DimProspector.indices.get(layer_idx)
        if sink_inds is None or len(sink_inds) == 0:
            return

        # Filter to visual sink tokens only
        vis_sink_mask = (sink_inds >= image_start) & (sink_inds < image_start + image_len)
        vis_sink_inds = sink_inds[vis_sink_mask]

        if len(vis_sink_inds) == 0:
            return

        # Extract attention to image region [batch, heads, query, image_tokens]
        image_attn = attn_weights[:, :, :, image_start:image_start+image_len]

        # Calculate portion of attention going to visual sink tokens
        # vis_sink_inds are absolute indices, need to make them relative to image start
        vis_sink_inds_relative = vis_sink_inds - image_start

        sink_attn_sum = image_attn[:, :, :, vis_sink_inds_relative].sum(dim=-1)
        total_image_attn = image_attn.sum(dim=-1) + 1e-6

        portion = sink_attn_sum / total_image_attn  # [batch, heads, query]
        summation = image_attn.sum(dim=-1)  # [batch, heads, query]

        # Apply conditions
        portion_condition = portion <= cls.rho
        summation_condition = summation >= cls.summ

        # Get coordinates [batch, head, query] of heads meeting both conditions
        candidate_coords = torch.nonzero(portion_condition & summation_condition)

        cls.__base__.forked_head[layer_idx] = candidate_coords.clone()


class VARProcessor(StashEngine):
    """
    Stage 3: Redistribute attention away from sink tokens.

    For selected heads, this:
    1. Reduces attention to all sink tokens (visual + text) by factor p
    2. Calculates freed attention budget
    3. Redistributes budget to non-sink visual tokens proportionally
    """

    # Configuration
    p = 0.6  # Fraction of sink attention to keep (0.6 = keep 60%, redistribute 40%)
    except_last_layer = True  # Whether to skip last decoder layer

    @classmethod
    def activate(cls, p=0.6, except_last_layer=True):
        """
        Activate VARProcessor with specified parameters.

        Args:
            p: Fraction of sink attention to keep
            except_last_layer: Whether to skip modifying the last layer
        """
        super().activate()
        cls.p = p
        cls.except_last_layer = except_last_layer

    @classmethod
    def attn_redist(cls, attention_map: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Redistribute attention weights for a given layer.

        Args:
            attention_map: [batch, num_heads, seq_len, seq_len] - attention weights
            layer_idx: Current layer index

        Returns:
            Modified attention map with same shape
        """
        # Skip last layer if configured
        num_layers = StashEngine.model_config["num_hidden_layers"]
        if cls.except_last_layer and layer_idx == num_layers - 1:
            return attention_map

        # Get image positions and forked heads
        img_pos = MetadataStation.get_image_positions()
        if img_pos is None:
            return attention_map

        image_start, image_len = img_pos
        coord = HeadFork.forked_head.get(layer_idx)

        if coord is None or len(coord) == 0:
            return attention_map

        # Get sink indices for this layer
        indices = DimProspector.indices.get(layer_idx)
        if indices is None or len(indices) == 0:
            return attention_map

        # Clone attention map for modification
        modified_attn = attention_map.clone()

        # Get query length (1 during decode, N during prefill)
        bsz, num_heads, q_len, seq_len = attention_map.shape

        # Split sink indices into visual and text
        vis_sink_mask = (indices >= image_start) & (indices < image_start + image_len)
        vis_indices = indices[vis_sink_mask]
        text_indices = indices[~vis_sink_mask]

        # Process each unique (batch, head) pair
        for h in range(num_heads):
            # Get query positions for this head
            head_mask = coord[:, 1] == h
            if not head_mask.any():
                continue

            query_coord = coord[head_mask][:, 2]
            batch_coord = coord[head_mask][:, 0]
            head_coord = coord[head_mask][:, 1]

            # Handle decode case: during decode, q_len=1, so query position is always 0
            if q_len == 1:
                # Clamp all query coordinates to 0 (the only valid position during decode)
                query_coord = torch.zeros_like(query_coord)

            # Select attention map for these queries [num_queries, seq_len]
            selected_attn_map = modified_attn[batch_coord, head_coord, query_coord, :].clone()

            # Save original for calculating budget
            original_attn_map = selected_attn_map.clone()

            # Step 1: Reduce sink token attention by factor p
            if len(vis_indices) > 0:
                selected_attn_map[:, vis_indices] *= cls.p
            if len(text_indices) > 0:
                selected_attn_map[:, text_indices] *= cls.p

            # Step 2: Calculate freed budget
            budget = 0.0
            if len(vis_indices) > 0:
                budget += original_attn_map[:, vis_indices].sum(dim=1) * (1 - cls.p)
            if len(text_indices) > 0:
                budget += original_attn_map[:, text_indices].sum(dim=1) * (1 - cls.p)

            # Step 3: Calculate redistribution ratios for non-sink visual tokens
            # Zero out sinks to get non-sink distribution
            vis_region = selected_attn_map[:, image_start:image_start+image_len].clone()

            if len(vis_indices) > 0:
                vis_sink_relative = vis_indices - image_start
                vis_region[:, vis_sink_relative] = 0

            # Calculate ratios
            vis_sum = vis_region.sum(dim=1, keepdim=True) + 1e-8
            ratios = vis_region / vis_sum  # [num_queries, image_len]

            # Step 4: Redistribute budget
            selected_attn_map[:, image_start:image_start+image_len] += budget.view(-1, 1) * ratios

            # Update the original attention map
            modified_attn[batch_coord, head_coord, query_coord, :] = selected_attn_map

        return modified_attn


class LogicEngine(StashEngine):
    """Coordinator for all VAR components."""

    @classmethod
    def activate(cls, tau=20, rho=0.5, summ=0.2, p=0.6, except_last_layer=True,
                 dim_indices=None, model_name="qwen2_5-vl"):
        """
        Activate all VAR components with specified parameters.

        Args:
            tau: Threshold for sink detection
            rho: Max portion of sink attention
            summ: Min total visual attention
            p: Fraction of sink attention to keep
            except_last_layer: Whether to skip last layer
            dim_indices: Dimension indices for sink detection
            model_name: Model name for default dimensions
        """
        super().activate()

        # Activate all components
        DimProspector.activate(tau=tau, dim_indices=dim_indices, model_name=model_name)
        HeadFork.activate(rho=rho, summ=summ)
        VARProcessor.activate(p=p, except_last_layer=except_last_layer)

        print(f"[VAR] Activated with tau={tau}, rho={rho}, summ={summ}, p={p}")

    @classmethod
    def clear(cls):
        """Clear all state from VAR components."""
        super().clear()
        DimProspector.clear()
        HeadFork.clear()
        VARProcessor.clear()
        MetadataStation.clear()
