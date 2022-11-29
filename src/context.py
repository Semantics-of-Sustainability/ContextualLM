import logging
from typing import Any, Dict, Generator, List, Optional

import torch
from transformers import RobertaModel, RobertaTokenizerFast
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from transformers.tokenization_utils_base import CharSpan, TokenSpan

from .text import Text


class Context(Text):
    def __init__(
        self,
        text: str,
        model: RobertaModel,
        tokenizer: RobertaTokenizerFast,
        metadata: Dict[str, Any],
        token: str,
        char_index: int,
    ) -> None:

        super().__init__(text, model, tokenizer, metadata)

        self._token = token
        self._char_index = char_index

    def __repr__(self) -> str:
        return str({"token": self.token, "text": self.text})

    @property
    def token(self) -> str:
        return self._token

    def token_embedding(self) -> Optional[torch.Tensor]:
        if self.has_word():
            if token_span := self._word_to_tokens():
                return self._aggregate_embeddings(token_span)
        else:
            logging.warning(
                f"Context does not contain word '{self._token}'. Consider calling 'has_word()' before token_embeddings(). Context metadata: {self._metadata}"
            )

        return None

    def has_word(self) -> bool:
        return self._word() == self._token

    def _word(self) -> str:
        char_span = self._word_to_chars()
        return self._text[char_span.start : char_span.end]

    def _word_to_chars(self) -> CharSpan:
        return self._encoding.word_to_chars(
            self._encoding.char_to_word(self._char_index)
        )

    def _word_to_tokens(self) -> TokenSpan:
        return self._encoding.word_to_tokens(
            self._encoding.char_to_word(self._char_index)
        )

    def _word_index(self) -> Optional[int]:
        return self._encoding.char_to_word(self._char_index)

    @classmethod
    def contexts(
        cls,
        text: str,
        token: str,
        context_characters: int,
        model: FeatureExtractionPipeline,
        tokenizer: RobertaTokenizerFast,
        metadata: Dict[str, Any],
    ) -> Generator["Context", None, None]:
        """Generate a Context object for each occurence of a token as a full word in a text.

        Args:
            - text: a text
            - token: a word/token
            - context_characters: the number of characters to extract from the text around the occurrence of the token
            - the model to use for generating embeddings and tokenization
            - metadata for the context

        Yields:
            A Context object for each occurrence of the token in which it constitutes a whole word.
        """

        match_index = text.find(token)
        while match_index >= 0:
            window_radius: int = int(context_characters / 2)
            start = max(0, match_index - window_radius)
            end = min(len(text), match_index + len(token) + window_radius)

            context = cls(
                text[int(start) : int(end)],
                model,
                tokenizer,
                metadata,
                token,
                match_index - start,
            )

            if context.has_word():
                yield context

            match_index = text.find(token, match_index + len(token))
