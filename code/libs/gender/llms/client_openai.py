"""Thin wrapper around the OpenAI Python SDK.

Drop-in replacement for client.py — same function signature,
same return type (raw_text, clean, data).
"""

import json

from openai import OpenAI

from libs.gender.llms.config import MODEL_ID, MAX_TOKENS, USER_PROMPT_TEMPLATE


def classify_batch(names: list[str], client: OpenAI, system_prompt: str) -> tuple:
    """Send a batch of names to GPT and return the parsed JSON dict."""
    user_content = USER_PROMPT_TEMPLATE.format(names="\n".join(names))

    response = client.chat.completions.create(
        model=MODEL_ID['gpt'],
        max_completion_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw_text = response.choices[0].message.content or ""
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