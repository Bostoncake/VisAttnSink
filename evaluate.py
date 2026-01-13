# judge: exact match
# prompt: system prompt

#!/usr/bin/env python3
"""
Spatial eval (SAT-2, BLINK) WITHOUT vLLM and WITHOUT MCTS.

Design goals:
- Keep data loading/prompting/result parsing/correctness checking aligned with the repo's evaluation intent.
- Use per-example `prompt` and `answer` fields when available (BLINK provides these).
- Optionally read SYSTEM_PROMPT (and MODEL) from scripts/evaluation/eval_spatial.sh to stay consistent.

Usage examples:

1) Evaluate BLINK from local extracted DATA_ROOT using built-in HF Qwen2.5-VL runner:
    export DATA_ROOT=/path/to/data
    python scripts/evaluation/eval_spatial_no_vllm.py \
        --dataset blink --split val \
        --runner hf-qwen2_5-vl \
        --model gsarch/ViGoRL-7b-Spatial \
        --out_dir runs/no_vllm_eval

2) Evaluate SAT-2 similarly:
    python scripts/evaluation/eval_spatial_no_vllm.py \
        --dataset sat2 --split test \
        --runner hf-qwen2_5-vl \
        --model gsarch/ViGoRL-7b-Spatial

3) Use your own MLLM runner (plugin):
    # Provide a python callable: mypkg.my_runner:make_runner
    # which returns an object with: generate(prompt:str, images:list[PIL.Image], **gen_kwargs)->str
    python scripts/evaluation/eval_spatial_no_vllm.py \
        --dataset blink --split val \
        --runner python:mypkg.my_runner:make_runner
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import importlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Protocol, Union

# PIL is a hard dependency for image I/O
from PIL import Image


# -----------------------------
# Runner interface
# -----------------------------

class Runner(Protocol):
    def generate(self, prompt: str, images: List[Image.Image], **gen_kwargs: Any) -> str:
        ...


# -----------------------------
# Built-in HF runner: Qwen2.5-VL
# -----------------------------

class HFQwen25VLRunner:
    """
    Minimal Hugging Face runner for Qwen2.5-VL style chat VLMs.

    Notes:
    - This is intentionally lightweight and "eval-like" (temperature=0 by default).
    - It uses the model's chat template via the processor.
    """

    def __init__(
        self,
        model_id: str,
        model_cls,
        device: Optional[str] = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.model_id = model_id

        # device_map="auto" is the most common for evaluation-style usage.
        self.model = model_cls.from_pretrained(
            model_id,
            torch_dtype=(torch.float16 if dtype == "fp16" else "auto"),
            device_map="auto" if device is None else device,
            trust_remote_code=trust_remote_code,
        )
        if device is not None:
            self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

        # Optional helper used by Qwen examples; keep optional.
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
            self._process_vision_info = process_vision_info
        except Exception:
            self._process_vision_info = None

    def generate(self, prompt: str, images: List[Image.Image], **gen_kwargs: Any) -> str:
        torch = self.torch

        # generation kwargs
        system_prompt = gen_kwargs.pop("system_prompt", None)
        max_new_tokens = int(gen_kwargs.pop("max_new_tokens", 256))
        temperature = float(gen_kwargs.pop("temperature", 0.0))
        top_p = float(gen_kwargs.pop("top_p", 1.0))
        max_pixels = int(gen_kwargs.pop("max_pixels", 1587600))

        gen_method = str(gen_kwargs.pop("gen_method", "default"))
        gen_method_kwargs = gen_kwargs.pop("gen_method_kwargs", {})

        do_sample = temperature > 0.0

        # Build chat messages in Qwen VL format.
        # We provide images in order first, then the textual prompt.
        user_content: List[Dict[str, Any]] = []
        for img in images:
            user_content.append({"type": "image", "image": img, "max_pixels": max_pixels})
        user_content.append({"type": "text", "text": prompt})

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Fallback: pass images directly.
            inputs = self.processor(
                text=[text],
                images=images if images else None,
                padding=True,
                return_tensors="pt",
            )

        # Move tensors to the model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():       # note that this does not affect SLOT implementations
            if gen_method == "adapt_vis":

                import numpy as np

                first_round = self.model.generate(
                    **inputs,
                    max_new_tokens=1,       # only need the first token
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                uncertainty = np.round(float(max(torch.nn.functional.softmax(first_round['scores'][0], dim=-1)[0])), 2)

                if uncertainty < gen_method_kwargs["threshold"]:
                    # TODO: currently by default we use SDPA attention
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        top_p=top_p if do_sample else None,
                        adapt_vis_keys=torch.where(inputs['input_ids'] == self.model.config.image_token_id, 1, 0),
                        adapt_vis_weight=gen_method_kwargs["weight1"]
                    )
                else:
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        top_p=top_p if do_sample else None,
                        adapt_vis_keys=torch.where(inputs['input_ids'] == self.model.config.image_token_id, 1, 0),
                        adapt_vis_weight=gen_method_kwargs["weight2"]
                    )
            elif gen_method == "slot":
                os.environ["slot_times"] = str(gen_method_kwargs["slot_times"])
                os.environ["slot_lr"] = str(gen_method_kwargs["slot_lr"])
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                )
            else:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                )

        # Strip prompt tokens
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[:, prompt_len:]
        out = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return out.strip()

# -----------------------------
# Dataset loading
# -----------------------------

def load_records_from_file(path: str) -> List[Dict[str, Any]]:
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

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        # common wrappers
        for k in ("data", "annotations", "examples", "items"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # fallback: if dict-of-dicts
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
        raise ValueError(f"Unrecognized JSON structure in {path}")

    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")

    raise ValueError(f"Unsupported file type: {path}")


def _maybe_open_image(v: Any, base_dir: str) -> Optional[Image.Image]:
    if v is None:
        return None

    if isinstance(v, Image.Image):
        return v

    # HF datasets Image feature sometimes decodes to dict-like objects in some settings.
    if isinstance(v, dict):
        # Common patterns: {"path": "..."} or {"bytes": ...}
        if "bytes" in v and v["bytes"] is not None:
            try:
                return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
            except Exception:
                return None
        if "path" in v and v["path"]:
            v = v["path"]

    if isinstance(v, (str, Path)):
        p = Path(v)
        if not p.is_absolute():
            # Try relative to base_dir first
            p1 = Path(base_dir) / p
            if p1.exists():
                p = p1
            else:
                # Also try relative to spatial_reasoning root
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


@dataclass
class Example:
    ex_id: str
    prompt: str
    images: List[Image.Image]
    gold_answer: Any
    meta: Dict[str, Any]


def record_to_example(rec: Dict[str, Any], base_dir: str) -> Example:
    # Identify ID
    ex_id = str(
        str(rec.get("id"))
        or str(rec.get("idx"))
        or str(rec.get("__key__"))
        or str(rec.get("question_id"))
        or str(rec.get("uid"))
        or ""
    )

    prompt = rec['conversations'][0]['value']
    gold = rec['conversations'][1]['value']

    # Images:
    images: List[Image.Image] = []

    # BLINK-style multiple image fields:
    for key in ("image_1", "image_2", "image_3", "image_4"):
        if key in rec:
            img = _maybe_open_image(rec.get(key), base_dir)
            if img is not None:
                images.append(img)

    # Generic single-image keys:
    if not images:
        for key in ("image", "img", "image_path", "img_path", "path"):
            if key in rec:
                img = _maybe_open_image(rec.get(key), base_dir)
                if img is not None:
                    images.append(img)
                    break

    meta = {}
    for k in ("sub_task", "task", "split", "source"):
        if k in rec:
            meta[k] = rec[k]

    return Example(
        ex_id=ex_id,
        prompt=prompt,
        images=images,
        gold_answer=gold,
        meta=meta,
    )


# -----------------------------
# Answer parsing / scoring
# -----------------------------

_MC_LABEL_RE = re.compile(r"\(\s*([A-E])\s*\)")
_ANSWER_COLON_RE = re.compile(r"(?i)\banswer\b\s*[:\-]\s*([A-E])\b")
_TAG_RE = re.compile(r"(?is)<answer>\s*(.*?)\s*</answer>")

def normalize_mc_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    # Common gold in BLINK is "(A)".
    m = _MC_LABEL_RE.search(s)
    if m:
        return m.group(1).upper()
    # Or sometimes just "A"
    if len(s) == 1 and s.upper() in "ABCDE":
        return s.upper()
    # Or "A."
    if len(s) >= 1 and s[0].upper() in "ABCDE":
        return s[0].upper()
    return None

def extract_pred_label(text: str, choices: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract an MCQ label A-E from model output.

    Priority:
    1) <answer>...</answer>
    2) "(A)"-style
    3) "Answer: A"
    4) last standalone A-E token
    5) match choice text (weak heuristic)
    """
    if not text:
        return None

    # 1) answer tag
    m = _TAG_RE.search(text)
    if m:
        inner = m.group(1).strip()
        lab = normalize_mc_label(inner)
        if lab:
            return lab

    # 2) "(A)" style
    m = _MC_LABEL_RE.search(text)
    if m:
        return m.group(1).upper()

    # 3) "Answer: A"
    m = _ANSWER_COLON_RE.search(text)
    if m:
        return m.group(1).upper()

    # 4) last standalone label token
    toks = re.findall(r"\b([A-E])\b", text.upper())
    if toks:
        return toks[-1]

    # 5) weak text match to choices
    if choices:
        t = text.strip().lower()
        # If model outputs exact choice string, map to its index -> letter
        for i, ch in enumerate(choices):
            if isinstance(ch, str) and ch.strip().lower() == t:
                return "ABCDE"[i] if i < 5 else None

    return None

def normalize_freeform(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    # remove surrounding quotes/parens
    s = s.strip(" \t\r\n\"'`()[]{}")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def is_correct(pred_text: str, gold: Any, judge: str = "string_match") -> bool:

    def _remove_punctuation_spaces(ans: str) -> str:

        """
        Removes punctuation from the answer and leading and trailing spaces.
        
        INPUTS:
        - ans: The answer.
        
        OUTPUTS:
        - ans_filtered: The answer without punctuation.
        
        """

        #remove any spaces 
        ans = ans.strip()

        #remove any punctuation marks
        ans_filtered = ans.replace(".", "")
        ans_filtered = ans_filtered.replace("?", "")
        ans_filtered = ans_filtered.replace("!", "")
        ans_filtered = ans_filtered.replace(",", "")
        ans_filtered = ans_filtered.replace(";", "")
        ans_filtered = ans_filtered.replace(":", "")
        ans_filtered = ans_filtered.replace("'", "")
        ans_filtered = ans_filtered.replace('"', "")
        ans_filtered = ans_filtered.replace("(", "")
        ans_filtered = ans_filtered.replace(")", "")
        ans_filtered = ans_filtered.replace("[", "")
        ans_filtered = ans_filtered.replace("]", "")
        ans_filtered = ans_filtered.replace("{", "")
        ans_filtered = ans_filtered.replace("}", "")
        ans_filtered = ans_filtered.replace("<", "")
        ans_filtered = ans_filtered.replace(">", "")
        ans_filtered = ans_filtered.replace("/", "")
        ans_filtered = ans_filtered.replace("\\", "")
        ans_filtered = ans_filtered.replace("|", "")
        ans_filtered = ans_filtered.replace("=", "")
        ans_filtered = ans_filtered.replace("+", "")
        ans_filtered = ans_filtered.replace("-", "")
        ans_filtered = ans_filtered.replace("_", "")
        ans_filtered = ans_filtered.replace("*", "")
        ans_filtered = ans_filtered.replace("&", "")
        ans_filtered = ans_filtered.replace("^", "")
        ans_filtered = ans_filtered.replace("%", "")
        ans_filtered = ans_filtered.replace("$", "")
        ans_filtered = ans_filtered.replace("#", "")
        ans_filtered = ans_filtered.replace("@", "")
        ans_filtered = ans_filtered.replace("`", "")
        ans_filtered = ans_filtered.replace("~", "")
        ans_filtered = ans_filtered.replace(" ", "")
        ans_filtered = ans_filtered.strip()
        ans_filtered = ans_filtered.lower()

        return ans_filtered

    if judge == "string_match":
        gold = gold.lower()

        pred_text = _remove_punctuation_spaces(pred_text)
        gold = _remove_punctuation_spaces(gold)

        if gold == pred_text or gold in pred_text:
            # print(f"Judge response : 1.0")
            return 1.0
        else:
            # print(f"Judge response : 0.0")
            return 0.0
    else:
        raise NotImplementedError


# -----------------------------
# Runner loading
# -----------------------------

def build_runner(runner_spec: str, model_id: Optional[str], model_cls) -> Runner:
    """
    runner_spec:
      - "hf-qwen2_5-vl"
      - "python:module.submodule:callable"
    """
    if runner_spec == "hf-qwen2_5-vl":
        if not model_id:
            raise ValueError("--model is required for runner=hf-qwen2_5-vl")
        return HFQwen25VLRunner(model_id, model_cls)

    if runner_spec.startswith("python:"):
        # python:mypkg.my_mod:make_runner
        _, rest = runner_spec.split("python:", 1)
        mod_name, fn_name = rest.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        runner = fn()
        # minimal interface check
        if not hasattr(runner, "generate"):
            raise TypeError(f"Runner from {runner_spec} has no .generate()")
        return runner  # type: ignore[return-value]

    raise ValueError(f"Unknown runner: {runner_spec}")

DATASET_TYPE = {
    "sat2": "spatial_reasoning",
    "blink": "spatial_reasoning"
}

DATASET_FOLDER = {
    "sat2": "sat2_test",
    "blink": "blink"
}

DATASET_FILE = {
    "sat2": "sat2_test.jsonl",
    "blink": "blink_validation.jsonl"
}

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["blink", "sat2"])
    ap.add_argument("--data_root", default="/mnt/data1/xiongyizhe/data/spatial", help="defaults to env DATA_ROOT")
    ap.add_argument("--eval_sh", default="scripts/evaluation/eval_spatial.sh",
                    help="used ONLY to auto-load SYSTEM_PROMPT/MODEL if not provided")
    ap.add_argument("--system_prompt", default="You are a helpful assistant. With the image as context, pick the correct answer choice to answer the provided question. Only return the text of the correct answer choice.", help="override system prompt")
    # You are a helpful assistant. With the image as context, pick the correct answer choice to answer the provided question. Return the answer choice and the complete text of the correct answer choice.
    ap.add_argument("--model", default="/home/xiongyizhe/data/Qwen2.5-VL-3B-Instruct", help="HF model id/path for HF runners; optional for custom runners")
    ap.add_argument("--judge", required=True, choices=["string_match"], help="answer judge")

    ap.add_argument("--runner", default="hf-qwen2_5-vl",
                    help="hf-qwen2_5-vl | python:module:callable")

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--limit", type=int, default=-1, help="for quick debugging")
    ap.add_argument("--out_dir", default="./results/")
    ap.add_argument("--write_every", type=int, default=50)

    # for method specification
    ap.add_argument("--method", default='default', choices=["slot", "adapt_vis", "validate_sdpa", "var"], help="Specify method.")
    ap.add_argument("--slot_times", type=int, default=1)
    ap.add_argument("--slot_lr", type=float, default=0.001)
    # for AdaptVis
    ap.add_argument("--adaptvis_weight1", type=float, default=1.0)
    ap.add_argument("--adaptvis_weight2", type=float, default=1.0)
    ap.add_argument("--adaptvis_threshold", type=float, default=1.0)
    # for VAR
    ap.add_argument("--var_threshold", type=float, default=20)
    ap.add_argument("--var_attn_p", type=float, default=0.6)
    ap.add_argument("--var_head", type=float, default=0.8)

    args = ap.parse_args()

    if not args.data_root:
        print("ERROR: DATA_ROOT not set and --data_root not provided.", file=sys.stderr)
        sys.exit(2)

    data_root = os.path.abspath(args.data_root)
    spatial_root = os.path.join(data_root, DATASET_TYPE[args.dataset])
    if not os.path.isdir(spatial_root):
        print(f"ERROR: expected spatial_reasoning dir at: {spatial_root}", file=sys.stderr)
        sys.exit(2)

    # Load defaults from eval_spatial.sh if possible
    system_prompt = args.system_prompt
    model_id = args.model

    if system_prompt is None:
        system_prompt = ""  # stay explicit
    
    # import different qwen implementations for different methods
    try:
        if args.method == "slot":
            print("Use SLOT implementation.")
            from modeling_qwen2_5_vl_slot import Qwen2_5_VLForConditionalGeneration
            gen_method_kwargs= {
                "slot_times": args.slot_times,
                "slot_lr": args.slot_lr,
            }
        elif args.method == "adapt_vis":
            print("Use AdaptVis implementation.")
            from modeling_qwen2_5_vl_adapt_vis import Qwen2_5_VLForConditionalGeneration
            gen_method_kwargs= {
                "weight1": args.adaptvis_weight1,
                "weight2": args.adaptvis_weight2,
                "threshold": args.adaptvis_threshold,
            }
        elif args.method == "validate_sdpa":
            print("Use AdaptVis implementation.")
            from modeling_qwen2_5_vl_sdpa import Qwen2_5_VLForConditionalGeneration
            gen_method_kwargs = {}
        elif args.method == "var":
            print("Use Visualk Attention Attribution (Visual Attention Sink) implementation.")
            from modeling_qwen2_5_vl_var import Qwen2_5_VLForConditionalGeneration
            gen_method_kwargs = {
                "threshold": args.var_threshold,    # tao
                "attn_portion": args.var_attn_p,    # p
                "selected_heads": args.var_head,    # ro
            }
        else:
            print("Use default implementation.")
            from transformers import Qwen2_5_VLForConditionalGeneration
            gen_method_kwargs = {}
        model_cls = Qwen2_5_VLForConditionalGeneration
    except Exception:
        print("[Warning]: Import failed!")
        from transformers import AutoModelForVision2Seq  # type: ignore
        model_cls = AutoModelForVision2Seq
        gen_method_kwargs = {}

    runner = build_runner(args.runner, model_id, model_cls)

    ann_file = os.path.join(spatial_root, DATASET_FOLDER[args.dataset], DATASET_FILE[args.dataset])

    # Load & concatenate (some datasets may shard by subtask)
    records = load_records_from_file(ann_file)
    if args.limit > 0:
        records = records[: args.limit]

    # Convert to Examples
    examples: List[Example] = [record_to_example(r, base_dir=data_root) for r in records]
    # Filter empty prompts
    examples = [ex for ex in examples if ex.prompt.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    method_str = f"{args.method}"
    for key in gen_method_kwargs.keys():
        method_str += f"_{key}_{gen_method_kwargs[key]}"
    out_path = os.path.join(args.out_dir, f"{args.dataset}_predictions_{method_str}.jsonl")

    print(f"Loaded {len(examples)} examples")
    print(f"Writing predictions to: {out_path}")

    correct = 0
    total = 0
    per_task: Dict[str, List[int]] = {}

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(examples):

            if args.method == "slot":
                os.environ["slot_prompt_only"] = "True"

            gen_text = runner.generate(
                ex.prompt,
                ex.images,
                system_prompt=system_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                gen_method=args.method,
                gen_method_kwargs=gen_method_kwargs,
            )

            ok = is_correct(gen_text, ex.gold_answer, args.judge)
            total += 1
            correct += int(ok)

            task_name = str(ex.meta.get("sub_task") or ex.meta.get("task") or ex.meta.get("source") or "all")
            per_task.setdefault(task_name, []).append(int(ok))

            row = {
                "id": ex.ex_id,
                "dataset": args.dataset,
                "meta": ex.meta,
                "prompt": ex.prompt,
                "gold": ex.gold_answer,
                "pred_text": gen_text,
                "correct": ok,
                "num_images": len(ex.images),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % args.write_every == 0:
                fout.flush()
                elapsed = time.time() - t0
                acc = correct / total if total else 0.0
                print(f"[{i+1}/{len(examples)}] acc={acc:.4f} elapsed={elapsed:.1f}s")

        overall_row = {}

        overall_acc = correct / total if total else 0.0
        print(f"\nDONE. Overall accuracy: {overall_acc:.4f} ({correct}/{total})")
        overall_row["overall_acc"] = overall_acc
        overall_row["correct_num"] = correct
        overall_row["total_num"] = total

        # Per-task breakdown (useful for BLINK subtasks if present)
        if len(per_task) > 0:
            print("\nPer-sub_task accuracy:")
            for k in sorted(per_task.keys()):
                vals = per_task[k]
                print(f"  {k:24s}  {sum(vals)/len(vals):.4f} ({sum(vals)}/{len(vals)})")
                overall_row[k] = {
                    "task_acc": float(sum(vals)/len(vals)), 
                    "correct_num": sum(vals), 
                    "total_num": len(vals),
                }
        # import pdb; pdb.set_trace()
        fout.write(json.dumps(overall_row, ensure_ascii=False) + "\n")
        fout.flush()


if __name__ == "__main__":
    main()
