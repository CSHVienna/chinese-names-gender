"""Thin wrapper around the Anthropic Python SDK."""

import json

from anthropic import Anthropic

from libs.gender.llms.config import MODEL_ID, MAX_TOKENS, USER_PROMPT_TEMPLATE


def classify_batch(names: list[str], client: Anthropic, system_prompt: str) -> dict:
    """Send a batch of names to Claude and return the parsed JSON dict."""
    user_content = USER_PROMPT_TEMPLATE.format(names="\n".join(names))

    message = client.messages.create(
        model=MODEL_ID['claude'],
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )

    raw_text = "".join(
        block.text for block in message.content if block.type == "text"
    )
    clean = raw_text.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON response: ", e)
        print("Raw response:")
        print(raw_text)
        print("Cleaned response:")
        print(clean)
        data = None
        
    return raw_text, clean, data