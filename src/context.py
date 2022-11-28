import logging
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from numpy.typing import ArrayLike
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from transformers.tokenization_utils_base import CharSpan, TokenSpan

from .sentence import Sentence


class Context(Sentence):
    def __init__(
        self,
        text: str,
        model: FeatureExtractionPipeline,
        metadata: Dict[str, Any],
        token: str,
        char_index: int,
    ) -> None:

        super().__init__(text, model, metadata)

        self._token = token
        self._char_index = char_index

    def __repr__(self) -> str:
        return str({"token": self.token, "text": self.text})

    @property
    def token(self) -> str:
        return self._token

    def token_embedding(self) -> Optional[ArrayLike]:
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

    def _word_to_tokens(self):
        return self._encoding.word_to_tokens(
            self._encoding.char_to_word(self._char_index)
        )

    def _word_index(self) -> Optional[int]:
        return self._encoding.char_to_word(self._char_index)

    def _find_token(self) -> Optional[TokenSpan]:
        if word_index := self._word_index():
            return self._encoding.word_to_tokens(word_index)
        else:
            logging.warning(
                f"Could not find token '{self.token}' at position {self._char_index}"
            )
            return None

    @classmethod
    def contexts(
        cls,
        text: str,
        token: str,
        context_characters: int,
        model: FeatureExtractionPipeline,
        metadata: Dict[str, Any],
    ) -> Generator["Sentence", None, None]:

        match_index = text.find(token)
        while match_index >= 0:
            window_radius: int = int(context_characters / 2)
            start = max(0, match_index - window_radius)
            end = min(len(text), match_index + len(token) + window_radius)

            context = cls(
                text[int(start) : int(end)], model, metadata, token, match_index - start
            )

            if context.has_word():
                yield context

            match_index = text.find(token, match_index + len(token))
