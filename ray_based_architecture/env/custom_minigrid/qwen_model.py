import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
from io import BytesIO


def simple_minigrid_to_messages(example):
    prompt = example.get("prompt")
    target = {
        "door_opened": example.get("door_opened"),
        "key_visible": example.get("key_visible"),
        "agent_in_green_square": example.get("agent_in_green_square"),
    }
    target = json.dumps(target, indent=4)

    example["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": example.get("image"),
                },
                {
                    "type": "text", 
                    "text": prompt
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]
    return example

def clean_messages(messages):
    """
    Removes None fields that HF datasets adds.
    """
    cleaned_messages = []
    for msg in messages:
        cleaned_msg = {
            "role": msg["role"],
            "content": []
        }
        for content in msg["content"]:
            content_dict = {}
            for key, value in content.items():
                if value is not None:
                    content_dict[key] = value
            cleaned_msg["content"].append(content_dict)
        cleaned_messages.append(cleaned_msg)
    return cleaned_messages

def _decode_hf_image_dict(img: Any):
    """Convert Hugging Face image storage dicts into PIL.Image."""
    if isinstance(img, dict) and "bytes" in img:
        print("Converting image")
        return Image.open(BytesIO(img["bytes"])).convert("RGB")
    return img


def _normalize_vision_inputs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for msg in messages:
        for content in msg.get("content", []):
            if content.get("type") == "image":
                content["image"] = _decode_hf_image_dict(content.get("image"))
    return messages

def extract_json_dict(text: str, allow_error: bool = False):
    """Extract JSON dict from text; optionally return None on parse failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                if allow_error:
                    return None
                return text.strip()
        if allow_error:
            return None
        return text.strip()

@torch.inference_mode()
def run_inference(model, processor, example: Dict[str, Any], max_new_tokens: int = 256):
    """
    Run single-example inference for Qwen3-VL + LoRA.
    Handles HF image dict format used by datasets.
    """
    if "messages" not in example:
        raise ValueError("Expected example to contain 'messages'.")

    messages = _normalize_vision_inputs(clean_messages(example["messages"]))
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        raise ValueError("No user message found in example['messages'].")

    text = processor.apply_chat_template(
        user_msgs, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        user_msgs,
        return_video_kwargs=True,
        return_video_metadata=True,
        image_patch_size=16,
    )
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs = list(video_inputs)
        if video_kwargs is None:
            video_kwargs = {}
        video_kwargs["video_metadata"] = list(video_metadatas)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        **(video_kwargs or {}),
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed_ids = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    decoded = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
    return extract_json_dict(decoded, allow_error=False)



def main():
    repo_root = Path("/data/private/rp218/reproducing/ClevrSkills")
    model_path = repo_root / "final_model_merged"


    # base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("Model loaded successfully.")

    seed = 42
    ds = simple_minigrid_load(seed=seed)
    ds = ds.map(simple_minigrid_to_messages, num_proc=4, desc="Building messages")
    example = ds["validation"][0]

    print("Prompt:")
    print(example["prompt"])
    print("\nGround truth booleans:")
    print(f"door_opened: {example['door_opened']}")
    print(f"key_visible: {example['key_visible']}")
    print(f"agent_in_green_square: {example['agent_in_green_square']}")

    pred = run_inference(model, processor, example, max_new_tokens=256)
    print("\nModel output:")
    print(pred)

    ground_truth = {
        "door_opened": bool(example["door_opened"]),
        "key_visible": bool(example["key_visible"]),
        "agent_in_green_square": bool(example["agent_in_green_square"]),
    }
    print("\nGround truth booleans:")
    print(ground_truth)

    if isinstance(pred, dict):
        print("\nMatch by key:")
        for key, value in ground_truth.items():
            print(f"{key}: pred={pred.get(key)} gt={value} match={pred.get(key) == value}")


if __name__ == "__main__":
    main()
