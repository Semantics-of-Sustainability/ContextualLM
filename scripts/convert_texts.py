import argparse
import os
import sys

from convert.sentence import SentenceConverter
from convert.settings import ENCODING

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert input files into one-sentence-per-line text files."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r"),
        nargs="+",
        default=sys.stdin,
        help="The input file(s). Default to stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("wt", encoding=ENCODING),
        default=sys.stdout,
        help="The output file. Defaults to stdout.",
    )
    parser.add_argument(
        "-m",
        "--spacy-model",
        type=str,
        required=True,
        help="The SpaCy model to use for sentence splitting (e.g. 'nl_core_news_sm').",
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=str,
        nargs="+",
        help="The column(s) to read text from in the input file(s).",
    )

    # TODO: arguments for compression etc.

    args = parser.parse_args()

    converter = SentenceConverter.from_spacy(args.spacy_model)

    for input_file in args.input:
        for line in converter.convert_csv(input_file.name, text_columns=args.columns):
            args.output.write(line)

    args.output.close()
