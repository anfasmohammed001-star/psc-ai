"""
AI connector with dual-backend support:
  Primary:  Google Gemini API (via google-genai SDK) — 1500 req/day free
  Fallback: OpenRouter API (requests) — free-tier with multiple models
Includes exponential backoff retries, base64 image encoding, and response caching.
"""
import base64
import logging
import time
import random
import requests
from typing import Dict, Any, Optional, List
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
VISION_EXTRACTION_PROMPT = """
You are a highly accurate OCR and content extraction engine specialized in Kerala PSC exam papers.
Analyze this page image and extract every exam question.

You MUST format your output precisely using the delimiters below:
[QUESTION_START]
**Question:** <question text in Malayalam or English as printed>
* A) <option A>
* B) <option B>
* C) <option C>
* D) <option D>
[QUESTION_END]

If a question spans multiple columns or continues, merge it correctly.
Do not extract instructions, headers, or footers.
Only output formatted questions matching the template structure.
"""

MARKDOWN_EXTRACTION_PROMPT = """
You are a highly accurate content extraction engine specialized in Kerala PSC exam papers.
Analyze the following document markdown content and extract every exam question.

You MUST format your output precisely using the delimiters below:
[QUESTION_START]
**Question:** <question text in Malayalam or English as printed>
* A) <option A>
* B) <option B>
* C) <option C>
* D) <option D>
[QUESTION_END]

If a question spans multiple sections, merge it correctly.
Do not extract instructions, headers, or footers.
Only output formatted questions matching the template structure.
"""


QUESTION_GENERATION_PROMPT = """
You are an expert curriculum writer and Kerala PSC exam developer.
Your task is to take this ORIGINAL question and create a brand-new PARALLEL practice question.

Guidelines:
1. The new question must test the EXACT same historical, regional, or scientific concept/topic.
2. The difficulty tier (vocabulary depth, conceptual complexity) must be identical.
3. Keep the exact same language (e.g., if the original is in Malayalam, the parallel must be in Malayalam).
4. Provide 4 options (A, B, C, D) that are plausible, but only ONE must be factually correct.
5. Provide the correct option clearly.

Format your output exactly as:
[QUESTION_START]
**Question:** <parallel question text>
* A) <option A>
* B) <option B>
* C) <option C>
* D) <option D>
* Correct Option: <A, B, C, or D>
[QUESTION_END]
"""

# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------
class GeminiClient:
    def __init__(self, api_key: str):
        from config import Config
        self.api_key = api_key                          # OpenRouter key
        self.model_name = Config.OPENROUTER_MODEL
        self.primary_model = Config.OPENROUTER_MODEL
        self.embed_model_name = Config.OPENROUTER_EMBED_MODEL
        self.base_url = "https://openrouter.ai/api/v1"
        self._daily_request_counter = 0
        self.disable_embeddings = False

        # ----------------------------------------------------------------
        # Google Gemini API (primary)
        # Free-tier: 1500 req/day but only ~15-20 req/min per model alias
        # We rotate through multiple model aliases to distribute load.
        # ----------------------------------------------------------------
        self._gemini_client = None
        self._gemini_api_key = Config.GEMINI_API_KEY
        # Ordered list of Gemini model aliases to try (separate per-model quotas)
        self._gemini_models = [
            "gemini-flash-latest",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ]
        if self._gemini_api_key:
            try:
                from google import genai as google_genai
                self._gemini_client = google_genai.Client(api_key=self._gemini_api_key)
                logger.info("Google Gemini API client initialised (primary backend).")
            except Exception as e:
                logger.warning(f"Failed to initialise Google Gemini client: {e}. Will use OpenRouter only.")
                self._gemini_client = None
        else:
            logger.info("GEMINI_API_KEY not set — using OpenRouter as the primary backend.")

        # ----------------------------------------------------------------
        # OpenRouter fallback lists
        # ----------------------------------------------------------------
        self.vision_fallbacks = [
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "moonshotai/kimi-k2.6:free",
            "nvidia/nemotron-nano-12b-v2-vl:free",
        ]
        if self.primary_model not in self.vision_fallbacks:
            self.vision_fallbacks.insert(0, self.primary_model)

        self.text_fallbacks = [
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "moonshotai/kimi-k2.6:free",
        ]
        if self.primary_model not in self.text_fallbacks:
            self.text_fallbacks.insert(0, self.primary_model)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jefna/psc-prep-ai",
            "X-Title": "Kerala PSC Exam Prep AI",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_quota_used(self) -> int:
        return self._daily_request_counter

    # ------------------------------------------------------------------
    # Gemini API helpers
    # ------------------------------------------------------------------
    def _gemini_call_with_retry(self, call_func, label: str) -> str:
        """
        Try each Gemini model alias in sequence.
        On 429 rate-limit, honour the retryDelay (capped at 65s) and retry
        the SAME model before moving to the next alias.
        Raises RuntimeError only when all aliases are exhausted.
        """
        import re
        from google.genai import errors as genai_errors

        last_error = None
        for model in self._gemini_models:
            max_model_retries = 2  # retry same model once after waiting
            for attempt in range(max_model_retries):
                try:
                    logger.info(f"Calling Google Gemini API ({model}) for {label}.")
                    return call_func(model)
                except genai_errors.ClientError as e:
                    status_code = getattr(e, 'status_code', None) or getattr(e, 'code', 0)
                    err_str = str(e)
                    if status_code == 429 or '429' in err_str:
                        # Extract retryDelay seconds from error message if present
                        delay_match = re.search(r'retryDelay.*?(\d+)s', err_str)
                        wait = min(int(delay_match.group(1)) if delay_match else 60, 65)
                        if attempt < max_model_retries - 1:
                            logger.warning(
                                f"Gemini {model} rate-limited (429). "
                                f"Waiting {wait}s before retry (attempt {attempt+1}/{max_model_retries})..."
                            )
                            time.sleep(wait)
                            continue  # retry same model
                        else:
                            logger.warning(
                                f"Gemini {model} still rate-limited after {max_model_retries} attempts. "
                                "Trying next model alias..."
                            )
                            last_error = e
                            break  # move to next model
                    else:
                        # Non-rate-limit error — skip this model immediately
                        logger.warning(f"Gemini {model} error for {label}: {err_str[:120]}. Trying next alias...")
                        last_error = e
                        break
                except Exception as e:
                    logger.warning(f"Gemini {model} unexpected error for {label}: {str(e)[:120]}. Trying next alias...")
                    last_error = e
                    break

        raise RuntimeError(f"All Gemini model aliases exhausted for {label}. Last error: {last_error}")

    def _gemini_vision(self, image_path: str) -> str:
        """Call Google Gemini with an image and return the text response."""
        from google.genai import types as genai_types

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        def call_func(model: str) -> str:
            response = self._gemini_client.models.generate_content(
                model=model,
                contents=[
                    genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    VISION_EXTRACTION_PROMPT,
                ],
            )
            return response.text

        return self._gemini_call_with_retry(call_func, "vision extraction")

    def _gemini_text(self, prompt: str) -> str:
        """Call Google Gemini for text generation and return the text response."""
        def call_func(model: str) -> str:
            response = self._gemini_client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text

        return self._gemini_call_with_retry(call_func, "text generation")

    # ------------------------------------------------------------------
    # OpenRouter retry wrapper
    # ------------------------------------------------------------------
    def _execute_with_retry(self, request_func, *args, **kwargs) -> requests.Response:
        """Implements exponential backoff with random jitter for robust rate limit handling."""
        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                self._daily_request_counter += 1
                response = request_func(*args, **kwargs)
            except Exception as conn_err:
                logger.error(f"HTTP Connection error: {str(conn_err)}")
                if attempt == max_retries - 1:
                    raise conn_err
                time.sleep((base_delay ** attempt) * random.uniform(0.5, 1.5))
                continue

            if response is not None:
                # Check for daily free-tier limit exhaustion
                if "free-models-per-day" in response.text:
                    raise RuntimeError(
                        "OpenRouter daily free-tier quota exceeded. "
                        "Please add credits or use a paid/different API key."
                    )

                is_retryable_error = False
                error_msg = ""
                try:
                    res_data = response.json()
                    if "error" in res_data:
                        err = res_data["error"]
                        err_code = err.get("code")
                        error_msg = err.get("message", "")
                        if (
                            err_code in [429, 500, 502, 503, 504]
                            or "aborted" in error_msg.lower()
                            or "timeout" in error_msg.lower()
                        ):
                            is_retryable_error = True
                        else:
                            raise ValueError(f"OpenRouter API error: {err}")
                except ValueError as ve:
                    raise ve
                except Exception:
                    pass

                if (
                    response.status_code == 429
                    or response.status_code in [502, 503, 504]
                    or is_retryable_error
                ):
                    jitter = random.uniform(0.5, 1.5)
                    delay = (base_delay ** attempt) * jitter
                    msg = (
                        f"Status {response.status_code}"
                        if not is_retryable_error
                        else f"JSON error '{error_msg}'"
                    )
                    logger.warning(
                        f"OpenRouter {msg}. Retrying in {delay:.2f}s… "
                        f"(Attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    response.raise_for_status()
                    return response

        raise RuntimeError("OpenRouter API request failed after maximum retry attempts.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_questions_from_image(self, image_path: str) -> str:
        """
        Extract questions from a page image.
        Tries Google Gemini API first, then falls back to OpenRouter free models.
        """
        # ── Primary: Google Gemini API ──────────────────────────────────
        if self._gemini_client is not None:
            try:
                return self._gemini_vision(image_path)
            except Exception as e:
                logger.warning(
                    f"Google Gemini vision failed ({e}). Falling back to OpenRouter."
                )

        # ── Fallback: OpenRouter free models ───────────────────────────
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read/encode image {image_path}: {str(e)}")
            raise

        last_error = None
        for model in self.vision_fallbacks:
            logger.info(f"Attempting vision extraction with OpenRouter model: {model}")
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 2000,
            }
            if "reasoning" in model:
                payload["reasoning"] = {"enabled": True}

            def make_vision_call(p=payload):
                return requests.post(
                    f"{self.base_url}/chat/completions",
                    json=p,
                    headers=self.headers,
                    timeout=60,
                )

            try:
                response = self._execute_with_retry(make_vision_call)
                res_data = response.json()
                if "choices" in res_data and res_data["choices"]:
                    content = res_data["choices"][0]["message"].get("content")
                    return content if content is not None else ""
                raise ValueError(
                    f"Unexpected response structure from OpenRouter: {res_data}"
                )
            except Exception as e:
                logger.warning(
                    f"Vision extraction failed for OpenRouter model {model}: {str(e)}. "
                    "Trying next fallback…"
                )
                last_error = e

        raise RuntimeError(
            f"Vision extraction failed on all backends. Last error: {str(last_error)}"
        )

    def extract_questions_from_markdown(self, markdown_content: str) -> str:
        """
        Extract exam questions from a markdown formatted document.
        Tries Google Gemini API first, then falls back to OpenRouter free models.
        """
        full_prompt = f"{MARKDOWN_EXTRACTION_PROMPT}\n\nHere is the document content in Markdown:\n{markdown_content}"
        
        # ── Primary: Google Gemini API ──────────────────────────────────
        if self._gemini_client is not None:
            try:
                return self._gemini_text(full_prompt)
            except Exception as e:
                logger.warning(
                    f"Google Gemini markdown extraction failed ({e}). Falling back to OpenRouter."
                )

        # ── Fallback: OpenRouter free models ───────────────────────────
        last_error = None
        for model in self.text_fallbacks:
            logger.info(f"Attempting markdown extraction with OpenRouter model: {model}")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 3000,
            }
            if "reasoning" in model:
                payload["reasoning"] = {"enabled": True}

            def make_text_call(p=payload):
                return requests.post(
                    f"{self.base_url}/chat/completions",
                    json=p,
                    headers=self.headers,
                    timeout=90,
                )

            try:
                response = self._execute_with_retry(make_text_call)
                res_data = response.json()
                if "choices" in res_data and res_data["choices"]:
                    content = res_data["choices"][0]["message"].get("content")
                    return content if content is not None else ""
                raise ValueError(
                    f"Unexpected response structure from OpenRouter: {res_data}"
                )
            except Exception as e:
                logger.warning(
                    f"Markdown extraction failed for OpenRouter model {model}: {str(e)}. "
                    "Trying next fallback…"
                )
                last_error = e

        raise RuntimeError(
            f"Markdown extraction failed on all backends. Last error: {str(last_error)}"
        )

    @lru_cache(maxsize=500)
    def generate_parallel_question(self, original_question_block: str) -> str:
        """
        Generate a parallel question.
        Tries Google Gemini API first, then falls back to OpenRouter free models.
        """
        full_prompt = (
            f"{QUESTION_GENERATION_PROMPT}\n\nOriginal Question:\n{original_question_block}"
        )

        # ── Primary: Google Gemini API ──────────────────────────────────
        if self._gemini_client is not None:
            try:
                return self._gemini_text(full_prompt)
            except Exception as e:
                logger.warning(
                    f"Google Gemini text generation failed ({e}). "
                    "Falling back to OpenRouter."
                )

        # ── Fallback: OpenRouter free models ───────────────────────────
        last_error = None
        for model in self.text_fallbacks:
            logger.info(
                f"Attempting parallel question generation with OpenRouter model: {model}"
            )
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 1000,
            }
            if "reasoning" in model:
                payload["reasoning"] = {"enabled": True}

            def make_text_call(p=payload):
                return requests.post(
                    f"{self.base_url}/chat/completions",
                    json=p,
                    headers=self.headers,
                    timeout=60,
                )

            try:
                response = self._execute_with_retry(make_text_call)
                res_data = response.json()
                if "choices" in res_data and res_data["choices"]:
                    content = res_data["choices"][0]["message"].get("content")
                    return content if content is not None else ""
                raise ValueError(
                    f"Unexpected response structure from OpenRouter: {res_data}"
                )
            except Exception as e:
                logger.warning(
                    f"Parallel question generation failed for OpenRouter model {model}: "
                    f"{str(e)}. Trying next fallback…"
                )
                last_error = e

        raise RuntimeError(
            f"Parallel question generation failed on all backends. "
            f"Last error: {str(last_error)}"
        )

    def get_embedding(self, text: str) -> List[float]:
        """Fetches vector embedding for semantic comparison using OpenRouter embeddings API."""
        if self.disable_embeddings:
            raise RuntimeError(
                "Embeddings API is disabled due to previous 402 Payment Required response."
            )

        payload = {"model": self.embed_model_name, "input": text}

        def make_call():
            return requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=self.headers,
                timeout=30,
            )

        try:
            response = self._execute_with_retry(make_call)
            res_data = response.json()
            if "data" in res_data and res_data["data"]:
                return res_data["data"][0]["embedding"]
            raise ValueError(
                f"Unexpected response structure from OpenRouter embeddings: {res_data}"
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 402:
                logger.warning(
                    "Embeddings API returned 402 Payment Required. "
                    "Disabling embeddings and falling back to Jaccard similarity."
                )
                self.disable_embeddings = True
            raise
        except Exception as e:
            logger.error(f"Error calling embeddings API: {str(e)}")
            raise
