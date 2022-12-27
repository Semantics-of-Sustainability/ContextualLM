import argparse
from pathlib import Path
import sys

from tqdm import tqdm

from convert.sentence import SentenceConverter
from convert.settings import ENCODING

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert input files into one-sentence-per-line text files."
    )

    parser.add_argument(
        "-p", "--path", type=Path, required=True, help="Input directory."
    )
    parser.add_argument(
        "-g",
        "--glob",
        type=str,
        default="*",
        help="Glob for identifying input files in the input directory.",
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

    input_files = list(args.path.glob(args.glob))

    for input_file in tqdm(input_files, unit="file", desc="Reading"):
        for line in converter.convert_csv(input_file, text_columns=args.columns):
            args.output.write(line)

    args.output.close()
