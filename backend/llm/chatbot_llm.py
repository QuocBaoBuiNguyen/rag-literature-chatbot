import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()


# Maintain a rotating list of API keys. Prefer GEMINI_API_KEYS (comma-separated),
# fallback to single-key envs GOOGLE_API_KEY / GEMINI_API_KEY.
def _load_api_keys_from_env() -> List[str]:
    keys_env = ''
    keys = [k.strip() for k in keys_env.split(",") if k.strip()]
    return keys


_API_KEYS: List[str] = _load_api_keys_from_env()
_current_key_index: int = 0


def _configure_model_for_current_key() -> genai.GenerativeModel:  # type: ignore
    global _current_key_index
    api_key = _API_KEYS[_current_key_index]
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


_model = _configure_model_for_current_key()


def _rotate_key_and_reconfigure() -> None:
    global _current_key_index, _model
    _current_key_index = (_current_key_index + 1) % len(_API_KEYS)
    next_key_suffix = _API_KEYS[_current_key_index][-6:] if _API_KEYS[_current_key_index] else ""
    print(f"🔑 Rotating Gemini API key. Using key index #{_current_key_index} (…{next_key_suffix}).")
    _model = _configure_model_for_current_key()


def generate_answer(prompt: str) -> str:
    attempts_remaining = len(_API_KEYS)

    last_error: Exception | None = None
    while attempts_remaining > 0:
        try:
            response = _model.generate_content(prompt)
            # Some responses may be None or lack text in edge cases
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Empty response from model")
            return text.strip()
        except Exception as e:  # Broad catch to trigger key rotation on any generation failure
            last_error = e
            print(f"🔑 Current key: {_API_KEYS[_current_key_index]}")
            print(f"⚠️ Gemini generation failed with current key: {e}")
            attempts_remaining -= 1
            if attempts_remaining == 0:
                break
            _rotate_key_and_reconfigure()

    # If we exhausted all keys, re-raise the last error
    raise RuntimeError(f"Gemini generation failed after rotating all API keys: {last_error}")
