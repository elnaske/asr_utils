import json
import csv
from pathlib import Path
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Script for extracting JSON data from the SAP challenge 2 to TSV.")
    parser.add_argument("input", nargs='+', help="Input JSON file.")
    parser.add_argument("-o", "--output", nargs='+', default=[],
                        help="Output file names or directory")
    return parser


def parse_args(parser):
    args = parser.parse_args()
    out_dir = Path.cwd()
    OUTFILES = args.output
    INFILES = args.input
    merge_files = False

    if OUTFILES:
        if not (len(OUTFILES) == len(INFILES) or len(OUTFILES) == 1):
            raise ValueError(
                "Number of input files must match number of output files, or output must be a directory")
        if len(OUTFILES) == 1:
            output = Path(OUTFILES[0])

            if output.parts[-1].endswith(".tsv"):
                merge_files = True
                output.parent.mkdir(parents=True, exist_ok=True)
                OUTFILES = [output] * len(INFILES)
            elif not '.' in output.parts[-1]:
                # output is not a file, therefore a directory
                out_dir = output
                output.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(
                    f"Unsupported file extension `.{output.parts[-1].split('.')[-1]}`")

    if len(OUTFILES) <= 1:
        OUTFILES = [out_dir / (Path(f).parts[-1].split('.')[0] + ".tsv") for f in INFILES]

    return INFILES, OUTFILES, merge_files


def write_header(headings, outfile):
    with open(outfile, 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow(headings)


def parse_json_and_append(infile, outfile):
    with open(infile, 'r') as f:
        json_data = json.load(f)
    
    etiology = json_data["Etiology"]

    # assuming JSON is in same dir as audio files
    root = Path(infile).parent.absolute()

    with open(outfile, 'a') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for f in json_data["Files"]:
            path = str(root / f["Filename"])

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


def main():
    parser = get_parser()
    INFILES, OUTFILES, merge_files = parse_args(parser)

    headings = ["path", "transcript", "etiology",
                "prompt_category", "intelligibility"]

    for i, (INFILE, OUTFILE) in enumerate(zip(INFILES, OUTFILES)):
        if i == 0 or not merge_files:
            write_header(headings, OUTFILE)


        parse_json_and_append(INFILE, OUTFILE)


if __name__ == "__main__":
    main()
