"""Tokenizer and detokenizer for LLMs."""

import time
from typing import Any, AsyncGenerator, Dict, Union

import numpy as np
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,  # type: ignore
    PreTrainedTokenizerFast,
)

from .logging import get_logger
from .utils import async_caller_empty_batch_handler, maybe_download_model_from_s3

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast, Any]

logger = get_logger(__name__)


def get_cached_tokenizer(tokenizer: AnyTokenizer) -> AnyTokenizer:
    """Get tokenizer with cached properties.

    This will patch the tokenizer object in place.
    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown. This
    function caches these properties for faster access.

    Args:
        tokenizer: The tokenizer object.

    Returns:
        The patched tokenizer object.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    # For VLM, the text tokenizer is wrapped by a processor.
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
        # Some VLM's tokenizer has chat_template attribute (e.g. Qwen/Qwen2-VL-7B-Instruct),
        # however some other VLM's tokenizer does not have chat_template attribute (e.g.
        # mistral-community/pixtral-12b). Therefore, we cache the processor's chat_template.
        if chat_template is None:
            chat_template = getattr(tokenizer, "chat_template", None)

    tokenizer_all_special_ids = set(tokenizer.all_special_ids)
    tokenizer_all_special_tokens_extended = tokenizer.all_special_tokens_extended
    tokenizer_all_special_tokens = set(tokenizer.all_special_tokens)
    tokenizer_len = len(tokenizer)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore
        @property
        def all_special_ids(self):
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self):
            return tokenizer_all_special_tokens

        @property
        def all_special_tokens_extended(self):
            return tokenizer_all_special_tokens_extended

        @property
        def chat_template(self):
            return chat_template

        def __len__(self):
            return tokenizer_len

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    tokenizer.__class__ = CachedTokenizer
    return tokenizer


class ChatTemplateTokenizer:
    """Tokenizer with chat template applied.

    Args:
        model: The model name.
    """

    def __init__(self, model: str) -> None:
        self.model = maybe_download_model_from_s3(model)
        self.tokenizer = get_cached_tokenizer(AutoProcessor.from_pretrained(self.model))

    @async_caller_empty_batch_handler
    async def __call__(
        self, batch: Dict[str, np.ndarray]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Call the tokenizer to process a batch.
        This function first process inputs in the batch asynchronously to apply
        chat template because this step cannot be batched. Then it tokenizes all inputs at once.

        Args:
            batch: The batch.

        Yields:
            The output.
        """
        if "messages" not in batch:
            raise KeyError(f'"messages" not found in {batch.keys()=}')

        start_t = time.perf_counter()
        messages = batch["messages"].tolist()

        # Tokenize text prompts.
        full_prompts = []
        for conversation in messages:
            # add generation prompt only if the last message is from the user
            add_generation_prompt = conversation[-1]["role"] == "user"
            full_prompts.append(
                self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=not add_generation_prompt,
                )
            )
        tokens = self.tokenizer(full_prompts)["input_ids"]
        time_taken_tokenizer = time.perf_counter() - start_t

        ret = {
            **batch,
            "prompt": full_prompts,
            "tokenized_prompt": tokens,
            "num_text_tokens": [len(t) for t in tokens],
            "time_taken_tokenizer": [time_taken_tokenizer] * len(tokens),
        }

        yield ret


class Detokenizer:
    """Detokenizer for LLMs.

    Args:
        model: The model name.
    """

    def __init__(self, model: str) -> None:
        self.model = maybe_download_model_from_s3(model)
        self.tokenizer = get_cached_tokenizer(AutoTokenizer.from_pretrained(self.model))

    async def __call__(
        self, batch: Dict[str, np.ndarray]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Detokenize the batch.

        Args:
            batch: The batch data.

        Returns:
            The detokenized batch.
        """
        start_t = time.perf_counter()
        generated_tokens = batch["generated_tokens"]
        flattened = False
        # if the generated tokens are nested lists, flatten them
        if isinstance(generated_tokens[0][0], np.ndarray):
            # flatten the lists of lists for detokenization
            flattened = True
            generated_tokens = [
                token for tokens in generated_tokens for token in tokens
            ]  # flattens list
        generated_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        if flattened:
            # unflatten the list back to original structure
            curr_idx = 0
            generated_text_unflattened = []
            for sublist in batch["generated_tokens"]:
                sublist_len = len(sublist)
                generated_text_unflattened.append(
                    generated_text[curr_idx : curr_idx + sublist_len]
                )
                curr_idx += sublist_len
            generated_text = generated_text_unflattened
        time_taken_detokenizer = time.perf_counter() - start_t
        yield {
            **batch,
            "generated_text": generated_text,
            "time_taken_detokenizer": [time_taken_detokenizer] * len(generated_text),
        }
