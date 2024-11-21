import cupy as cp
from numba import cuda
import cudf
import time
import rmm
import gzip
import argparse
import os

pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(),initial_pool_size=75*2**30, maximum_pool_size=80*2**30)
rmm.mr.set_current_device_resource(pool)



#identify all trigger strings in the sequence
@cuda.jit
def trigger_string_finder(sequence, prime, w, p, result):
    idx = cuda.grid(1)
    start = idx
    if start + w <= sequence.size:
        hash_val = 0
        for i in range(w):
            hash_val = (hash_val * prime + sequence[start + i]) & 0xFFFFFFFF
        if hash_val % p == 0 or start == 0 or start+w == sequence.size:
            result[idx] = 1


# Will need a better method for reading fasta file probably. Maybe read one sequence at a time with yield in main.
# Written by Tyler Pencinger
def read_fasta(file_path, w, lineLimit=1000000):
        # Reads a FASTA file and return a list of sequences
        firstSeq = True
        lineCount = 0

        with open(file_path, 'r') as file:
            current_seq = []
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequence = ''.join(current_seq)
                        if firstSeq:
                            firstSeq = False
                            yield '\x01' * w + sequence + '\x02' * w
                        else:
                            yield '\x02' * w + sequence + '\x02' * w
                        current_seq = []
                        lineCount = 0
                else:
                    current_seq.append(line)
                    lineCount += 1
                    if lineCount > lineLimit:
                        sequence = ''.join(current_seq)
                        if firstSeq:
                            firstSeq = False
                            yield '\x01' * w + sequence + '\x02' * w
                        else:
                            yield '\x02' * w + sequence + '\x02' * w
                        current_seq = []
                        lineCount = 0
            if current_seq:
                sequence = ''.join(current_seq)
                if firstSeq:
                    firstSeq = False
                    yield '\x01' * w + sequence + '\x02' * w
                else:
                    yield '\x02' * w + sequence + '\x01' * w
                current_seq = []
                lineCount = 0
        
        # for i, seq in enumerate(sequences):
        #     if i == 0:
        #         sequences[i] = '\x01' * w + seq
        #     else:
        #         sequences[i] = '\x02' * w + seq
        
        # sequences[-1] += '\x01' * w

        # for i in range(len(sequences) - 1):
        #     sequences[i] += '\x02' * w
        
        # # print(sequences)
        
        # return sequences


# Add some user input, command line options, etc. ##################################################
# Add timer option

def gpuPFP(input, w, p):

    dict = cudf.DataFrame({'phrases': cudf.Series(dtype='str'), 'hashes': cudf.Series(dtype='int64')})
    parse = cudf.Series(dtype='int64')

    programStart = time.time()

    # sequences = read_fasta(r"/blue/boucher/share/MHC-haplotigs/HPRC/concatenated_assemblies_w_grch38_chm13.fa.gz", w)
    # sequences = read_fasta(r"sequences.fasta", w)
    # sequences = read_fasta(input, w)
    # sequences = read_fasta(r"Chr21.10.reheader_consensus.fasta", w)


    for sequence in read_fasta(input, w):
        num_windows = len(sequence) - w + 1

        # Turn the sequence into ints for finding trigger string (rolling hash)
        sequenceAsInt = cp.array(bytearray(sequence.encode('utf-8')), dtype=cp.uint8)
        # Turn sequence into byte array for parsing on GPU
        # sequenceAsChar = cp.array(bytearray(sequence.encode('utf-8')), dtype=cp.uint8)

        # Store results for trigger string finding. Cupy array prefilled with 0s based on num windows.
        result = cp.zeros(num_windows, dtype=cp.uint8)

        # Need to figure out how to best determine threads per block, etc. ###############################################
        threads_per_block = 128
        blocks_per_grid = (num_windows + threads_per_block - 1) // threads_per_block

        # Call to trigger string finder GPU function, maybe generate prime (31) randomly every time program is called.
        trigger_string_finder[blocks_per_grid, threads_per_block](sequenceAsInt, 31, w, p, result)

        # Find all positions at which there is a trigger string. The index in the result matches up to the start position of the trigger string in the seq.
        # Basically, if result[i] == 1, then seq[i:i+w] is the trigger string.
        triggerStrings = cp.where(result == 1)[0]

        # Need to determine some values so we can create an n x m array of the necessary size, where n is num trigger strings and m is longest phrase for the GPU since dynamic allocation doesnt work :(
        # Num of phrases is easy. Maximum length is the longest distance between trigger string starts + w.
        num_phrases = len(triggerStrings) - 1

        # Get the phrases using trigger string positions
        if num_phrases > 1:
            # max_len = max(triggerStrings[i+1] - triggerStrings[i] for i in range(num_phrases)) + w
            # Need to figure out how to best determine threads per block, etc. ###############################################
            threads_per_block = 128
            blocks_per_grid = (num_phrases + (threads_per_block - 1)) // threads_per_block

            # Call to phrase parser GPU function.
            phrases = []
            triggerStrings = triggerStrings.get()
            for i in range(num_phrases):
                phrases.append(sequence[triggerStrings[i]:triggerStrings[i+1]+w])
            phrases_series = cudf.Series(phrases)

        else:
            phrases_series = cudf.Series([sequence])

        # Hashes the phrases with murmurhash3 using cudf built-in function and outputs a series of hashes. Also create dict for curr seq
        curr_parse = phrases_series.hash_values()
        curr_dict = cudf.DataFrame({'phrases': phrases_series, 'hashes': curr_parse})
        
        # Combine the dict and parse from curr seq to the final, preserving order for the parse and removing duplicates from the dict.
        parse = cudf.concat([parse, curr_parse], ignore_index=True)

        dict = cudf.concat([dict, curr_dict], ignore_index=True)

        dict = dict.drop_duplicates(subset=['hashes']).reset_index(drop=True)

        # Memory Collection
        del result
        del phrases
        del triggerStrings
        del phrases_series
        del curr_parse
        del curr_dict
        # print("\n\n")

    # Final step is to sort the dictionary, create a mapping to replace hashes with indices in the parse, then drop hashes from dict.
    dict = dict.sort_values(by='phrases', ignore_index=True)
    mapping = cudf.Series(dict.index, index=dict['hashes'])
    dict = dict.drop(columns=['hashes'])
    parse = parse.map(mapping)


    endTime = time.time()
    print(f"Total program runtime:{endTime - programStart:.4f}")
    
    return dict, parse



def get_file_prefix(filepath):

    # Remove the directory
    filename = os.path.basename(filepath)
    
    # Split the name and extension
    name, ext = os.path.splitext(filename)
    
    # Handle double extensions (e.g., .fasta.gz)
    if ext in ['.gz', '.bz2', '.xz']:
        name, _ = os.path.splitext(name)
    
    # Prefix is the base name without extensions
    return name


def main():
    parser = argparse.ArgumentParser(description="Placeholder Description", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', help="path to input fasta file",type=str, required=True)
    parser.add_argument('-w', '--wsize', help="sliding window size", default=5, type=int)
    parser.add_argument('-p', '--mod', help="hash modulus", default=15, type=int)
    parser.add_argument('-o', '--output', help="output files prefix", type=str)

    args = parser.parse_args()

    dict, parse = gpuPFP(args.input, args.wsize, args.mod)

    if not args.output:
        output_prefix = get_file_prefix(args.input)
    else:
        output_prefix = args.output
        
    dict = dict['phrases']
    with open(output_prefix + '.dict', 'w', encoding='utf-8') as file:
        file.write(dict.to_pandas().str.cat(sep='\x03'))
    with open(output_prefix + '.parse', 'wb') as file:
        file.write(parse.to_numpy().tobytes())




if __name__ == '__main__':
    main()

# cudf.core.column.as_column(d_phrases.ravel(), dtype="str")
# print(f"Execution Time for keeping unique dict values: {endTime - startTime:.4f}")

# GPU hashing function; this function kept in case needed
# @cuda.jit(device=True)
# def fnv1a_hash(data):
#     hval = 0x811c9dc5  # 32-bit FNV offset basis
#     fnv_prime = 0x01000193  # 32-bit FNV prime
#     for byte in data:
#         hval = hval ^ byte
#         hval = hval * fnv_prime    
#     return hval


# # GPU string parser function. Not currently used.
# @cuda.jit
# def dictionary_parser(triggerStrings, w, sequence, phrases, offsets):
#     idx = cuda.grid(1)
#     if idx < triggerStrings.size:
#         substr_len = triggerStrings[idx+1] - triggerStrings[idx]
#         offsets[idx] = substr_len + w
#         for i in range(substr_len + w):
#             phrases[idx, i] = sequence[triggerStrings[idx] + i]
#         # hashVal = fnv1a_hash(phrases[idx, :substr_len])
#         # hashes[idx] = hashVal