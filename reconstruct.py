import numpy as np
import os
import textwrap

def decompress_dict_parse(dict_path, parse_path, w, output_fasta):
    """
    Reconstructs the original sequence file from:
      - dict_path: path to the '.dict' file (phrases joined by \x03)
      - parse_path: path to the '.parse' file (little-endian uint32 indices)
      - w: the window size used during compression
      - output_fasta: path where the reconstructed FASTA file will be written

    The compressor uses:
      • '\x01'*w  to mark the very start and end of all sequences
      • '\x02'*w  to separate adjacent sequences

    This function:
      1. Reads and splits the dict into a list of phrases.
      2. Reads the parse indices and for each one appends the corresponding phrase.
      3. Strips off the global '\x01' padding prefix/suffix.
      4. Splits by the '\x02'*w separator to recover each original sequence.
      5. Writes them out in FASTA format with lines wrapped at 80 chars.
    """

    # 1) Load phrases
    with open(dict_path, 'rb') as f:
        dict_bytes = f.read()
    raw_phrases = dict_bytes.split(b'\x03')
    phrases = [p.decode('utf-8') for p in raw_phrases]
    print(phrases)

    # 2) Load parse indices
    #    dtype='<u4' = little-endian uint32
    parse_indices = np.fromfile(parse_path, dtype='<u4')
    print(parse_indices)

    # 3) Re-assemble the full padded concatenation
    full_padded = ''
    for idx in parse_indices:
        full_padded += phrases[idx][w:]
    # full_padded = ''.join(phrases[idx] for idx in parse_indices)
    print(full_padded)
    # 4) Strip off the leading '\x01'*w and trailing '\x01'*w if present
    start_marker = '\x01' * w
    end_marker   = '\x01' * w
    if full_padded.startswith(start_marker):
        full_padded = full_padded[w:]
    if full_padded.endswith(end_marker):
        full_padded = full_padded[:-w]

    # 5) Split back into individual sequences on '\x02'*w
    sep = '\x02' * w
    sequences = full_padded.split(sep)

    # 6) Write out as FASTA
    with open(output_fasta, 'w') as out:
        for i, seq in enumerate(sequences, start=1):
            header = f">seq{i}"
            out.write(header + "\n")
            # wrap at 80 chars per line
            for line in textwrap.wrap(seq, width=60):
                out.write(line + "\n")

    print(f"Reconstructed {len(sequences)} sequences → {output_fasta}")
    return sequences


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decompress FASTA from dict+parse files")
    parser.add_argument("-d", "--dict",   required=True,
                        help="path to .dict file")
    parser.add_argument("-p", "--parse",  required=True,
                        help="path to .parse file")
    parser.add_argument("-w", "--wsize",  type=int, default=5,
                        help="window size used during compression")
    parser.add_argument("-o", "--output", required=True,
                        help="output FASTA file to write")
    args = parser.parse_args()

    decompress_dict_parse(args.dict, args.parse, args.wsize, args.output)