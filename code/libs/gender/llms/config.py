"""Centralized configuration for the gender classification pipeline."""

MODEL_ID = {'claude': "claude-opus-4-8",
            'gpt'   : "gpt-5.4-nano"}

MAX_TOKENS = 4096
BATCH_SIZE = 100  # names per API call — tune to stay within output token limits

USER_PROMPT_TEMPLATE = """\
Classify the gender of the following Chinese names (Pinyin). Return one entry per name, \
preserving the exact input string as the key.

Names:
{names}
"""

CHUNK_SEPARATOR = "_chunk_"

INFERENCE_MODES = ["categorical", "numeric"]  # "categorical" for label output, "numeric" for 0-1 score output