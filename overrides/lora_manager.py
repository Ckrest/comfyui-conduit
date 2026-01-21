"""
Override for comfyui-lora-manager's complex widget serialization.

The LoRA manager widget has complex frontend logic that:
- Formats strength values as (text:strength.toFixed(2))
- Deduplicates LoRAs by name
- Handles clipStrength separately from strength

Pattern detection can't capture all this, so we provide an explicit node override.
"""

from typing import Any, List
from conduit_overrides import register_node_override, MimicryContext, NodeOverrideResult


def transform_loras(value: Any) -> Any:
    """
    Transform the internal lora array format to the expected backend format
    by encoding strength values into the text field.
    """
    if not isinstance(value, list):
        return value

    result = []
    seen_names = set()

    for lora in value:
        if not isinstance(lora, dict):
            result.append(lora)
            continue

        name = lora.get("name", "")

        # Deduplicate by name (keep last occurrence)
        if name in seen_names:
            # Remove previous occurrence
            result = [l for l in result if l.get("name") != name]
        seen_names.add(name)

        # Format strength into text if present
        text = lora.get("text", name)
        strength = lora.get("strength")

        if strength is not None:
            try:
                strength_float = float(strength)
                formatted_text = f"({text}:{strength_float:.2f})"
            except (ValueError, TypeError):
                formatted_text = text
        else:
            formatted_text = text

        result.append({
            **lora,
            "text": formatted_text,
        })

    return result


def transform_trigger_words(value: Any) -> Any:
    """
    Transform trigger word tags to include strength in the text field.
    """
    if not isinstance(value, list):
        return value

    result = []
    for tag in value:
        if not isinstance(tag, dict):
            result.append(tag)
            continue

        text = tag.get("text", "")
        strength = tag.get("strength")

        if strength is not None:
            try:
                strength_float = float(strength)
                formatted_text = f"({text}:{strength_float:.2f})"
            except (ValueError, TypeError):
                formatted_text = text
        else:
            formatted_text = text

        result.append({
            **tag,
            "text": formatted_text,
        })

    return result


@register_node_override("LoraManager")
async def override_lora_manager(node_data: dict, ctx: MimicryContext) -> NodeOverrideResult:
    """
    Node-level override for LoraManager.

    Handles all LoraManager-specific input transformations:
    - loras: Deduplicate and format strength into text
    - trigger_words: Format strength into text

    Other inputs on this node get default mimicry behavior.
    """
    inputs = node_data.get("inputs", {})
    changes = {}
    handled_inputs = set()

    # Transform loras input
    if "loras" in inputs:
        original = inputs["loras"]
        transformed = transform_loras(original)
        if transformed != original:
            inputs["loras"] = transformed
            changes["loras"] = {"old": original, "new": transformed, "reason": "lora_manager_override"}
        handled_inputs.add("loras")

    # Transform trigger_words input
    if "trigger_words" in inputs:
        original = inputs["trigger_words"]
        transformed = transform_trigger_words(original)
        if transformed != original:
            inputs["trigger_words"] = transformed
            changes["trigger_words"] = {"old": original, "new": transformed, "reason": "lora_manager_override"}
        handled_inputs.add("trigger_words")

    return NodeOverrideResult(
        node_data=node_data,
        handled_inputs=handled_inputs,
        changes=changes,
    )


print("[Conduit Overrides] Loaded LoRA manager node override")
