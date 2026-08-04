"""
model_router.py

Builds the full list of chat-capable models supported by litellm, for use
in a searchable Streamlit dropdown. Also handles the "Other" free-text
fallback for any model not present in litellm's list.

Why this file exists (design note):
litellm.model_cost is a big dictionary of EVERY model litellm knows about
(chat, embeddings, image generation, audio, etc.) It has ~2985 entries.
We only care about "mode": "chat" entries, since our platform sends
chat-style prompts. Filtering that down here keeps evaluator.py and app.py
clean -- they just ask this file for "the list" and "the resolved model id".
"""

import litellm

OTHER_OPTION = "Other (type custom model)"


def get_all_chat_models():
    """
    Returns a sorted list of dicts:
    [{"display": "gpt-4o-mini  —  openai", "model_id": "gpt-4o-mini"}, ...]

    model_id is already the exact string litellm.completion() expects --
    no extra prefix-building needed, litellm's own keys are pre-formatted.
    """
    model_cost = litellm.model_cost
    models = []

    for model_id, info in model_cost.items():
        if model_id == "sample_spec":
            continue
        if info.get("mode") != "chat":
            continue

        provider = info.get("litellm_provider", "unknown")
        display = f"{model_id}  —  {provider}"
        models.append({"display": display, "model_id": model_id, "provider": provider})

    models.sort(key=lambda m: m["display"].lower())
    return models


def get_display_options():
    """
    Returns just the display strings for the dropdown, with the
    'Other' fallback appended at the end.
    """
    models = get_all_chat_models()
    options = [m["display"] for m in models]
    options.append(OTHER_OPTION)
    return options


def resolve_model_id(selected_display, custom_model_id=None):
    """
    Given what the user picked in the dropdown, return the exact model_id
    to pass into litellm.completion(model=...).

    If they picked "Other", we use whatever they typed in the custom text box.
    """
    if selected_display == OTHER_OPTION:
        if not custom_model_id or not custom_model_id.strip():
            raise ValueError("Custom model name is empty. Please type a model id.")
        return custom_model_id.strip()

    models = get_all_chat_models()
    for m in models:
        if m["display"] == selected_display:
            return m["model_id"]

    raise ValueError(f"Could not resolve model id for selection: {selected_display}")


if __name__ == "__main__":
    # quick manual check when running this file directly:
    # python layer1/model_router.py
    options = get_display_options()
    print(f"Total dropdown options: {len(options)}")
    print("First 5:", options[:5])
    print("Last option (should be fallback):", options[-1])

    test_id = resolve_model_id(options[0])
    print(f"Resolved '{options[0]}' -> '{test_id}'")