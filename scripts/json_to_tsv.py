import json
import csv
from pathlib import Path
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Script for extracting JSON data from the SAP challenge 2 to TSV.")
    parser.add_argument("input", nargs='+', help="Input JSON file.")
    parser.add_argument("-o", "--output", nargs='+', default=None,
                        help="Output file names or directory")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    INFILES = args.input
    OUTFILES = args.output
    ROOT = Path.cwd()
    out_dir = ROOT

    if OUTFILES:
        if not (len(OUTFILES) == len(INFILES) or len(OUTFILES) == 1):
            raise ValueError(
                "Number of input files must match number of output files, or output must be a directory")
        if len(OUTFILES) == 1:
            output = OUTFILES[0]
            if output.endswith('/'):
                out_dir = ROOT / output
                out_dir.mkdir(parents=True, exist_ok=True)

    if not OUTFILES or len(OUTFILES) <= 1:
        OUTFILES = [out_dir / (f.split('.')[0] + ".tsv") for f in INFILES]

    for OUTFILE, INFILE in zip(OUTFILES, INFILES):
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
