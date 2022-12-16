import os
from pathlib import Path
from typing import Generator, List

import pandas as pd
import spacy
import spacy.cli
from tqdm import tqdm

from .settings import ENCODING


class SentenceConverter:
    """Use SpaCy for sentence splitting"""

    def __init__(self, spacy_model, max_doc_length: int = 2000000) -> None:
        spacy_model.max_length = max_doc_length
        self._spacy = spacy_model

    def _process(self, text: str) -> Generator[str, None, None]:
        # TODO: prepend date
        doc = self._spacy(text)
        for sent in doc.sents:
            yield str(sent)

    def convert_csv(
        self,
        filepath_or_buffer: Path,
        text_columns: List[str],
        *,
        compression="gzip",
        encoding: str = ENCODING,
        parse_dates: List[str] = ["date"],
        sep=";",
        **kwargs
    ) -> Generator[str, None, None]:

        df: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=filepath_or_buffer,
            sep=sep,
            compression=compression,
            encoding=encoding,
            # error_bad_lines=False,
            parse_dates=parse_dates,
            # **kwargs
        ).dropna(subset=text_columns)

        for row in tqdm(
            df.itertuples(), total=len(df), unit="row", desc=str(filepath_or_buffer)
        ):
            for column in text_columns:
                yield from self._process(row._asdict()[column])
            yield os.linesep

    @classmethod
    def from_spacy(cls, model_name: str):
        try:
            model = spacy.load(model_name)
        except OSError:
            spacy.cli.download(model_name)  # TODO: test
            model = spacy.load(model_name)

        for component in model.pipe_names:
            model.remove_pipe(component)
        model.add_pipe("sentencizer")

        return cls(model)
