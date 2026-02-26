import json
import csv
from pathlib import Path
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Script for extracting JSON data from the SAP challenge 2 to TSV.")
    parser.add_argument("input", nargs='+', help="Input JSON file.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file name")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    INFILES = args.input
    ROOT = Path.cwd()

    for INFILE in INFILES:
        OUTFILE = args.output if args.output else INFILE.split('.')[0] + ".tsv"

        headings = ["path", "transcript", "etiology",
                    "prompt_category", "intelligibility"]
        with open(OUTFILE, 'w') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            writer.writerow(headings)

        with open(INFILE, 'r') as infile:
            json_data = json.load(infile)

        etiology = json_data["Etiology"]

        with open(OUTFILE, 'a') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            for f in json_data["Files"]:
                path = str(ROOT / f["Filename"])

                transcript = f["Prompt"]["Transcript"].strip()
                # replacing \u00a0, a non-printable ascii char
                transcript = transcript.replace("\u00a0", " ")
                prompt_category = f["Prompt"]["Category Description"]

                # Removing spontaneous speech prompts in square brackets
                if prompt_category.strip() == "Spontaneous Speech Prompts":
                    transcript = ']'.join(transcript.split(']')[1:])

                intelligibility = 0
                for dim in f["Ratings"]:
                    if dim["Dimension Description"].strip().lower() in ["intelligibility", "intelligbility"] \
                            and dim["Level"]:
                        intelligibility = int(dim["Level"])

                writer.writerow([path, transcript, etiology,
                                prompt_category, intelligibility])


if __name__ == "__main__":
    main()
