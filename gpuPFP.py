import cupy as cp
from numba import cuda
import cudf
import time

#identify all trigger strings in the sequence
@cuda.jit
def trigger_string_finder(sequence, prime, w, p, result):
    idx = cuda.grid(1)
    start = idx
    allSpecial = True
    if start + w <= sequence.size:
        hash_val = 0
        for i in range(w):
            if sequence[start + i] > 5:
                allSpecial = False
            hash_val = (hash_val * prime + sequence[start + i]) & 0xFFFFFFFF
        if hash_val % p == 0 or allSpecial:
            result[idx] = 1


# GPU hashing function; this function kept in case needed
# @cuda.jit(device=True)
# def fnv1a_hash(data):
#     hval = 0x811c9dc5  # 32-bit FNV offset basis
#     fnv_prime = 0x01000193  # 32-bit FNV prime
#     for byte in data:
#         hval = hval ^ byte
#         hval = hval * fnv_prime    
#     return hval


# GPU string parser function. Not currently used.
@cuda.jit
def dictionary_parser(triggerStrings, w, sequence, phrases, offsets):
    idx = cuda.grid(1)
    if idx < triggerStrings.size:
        substr_len = triggerStrings[idx+1] - triggerStrings[idx]
        offsets[idx] = substr_len + w
        for i in range(substr_len + w):
            phrases[idx, i] = sequence[triggerStrings[idx] + i]
        # hashVal = fnv1a_hash(phrases[idx, :substr_len])
        # hashes[idx] = hashVal


# Will need a better method for reading fasta file probably. Maybe read one sequence at a time with yield in main.
# Written by Tyler Pencinger
def read_fasta(file_path, w, lineLimit=1000000):
        # Reads a FASTA file and return a list of sequences
        sequences = []
        lineCount = 0

        with open(file_path, 'r') as file:
            current_seq = []
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                        lineCount = 0
                else:
                    current_seq.append(line)
                    lineCount += 1
                    if lineCount > lineLimit:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                        lineCount = 0

            if current_seq:
                sequences.append(''.join(current_seq))
        
        for i, seq in enumerate(sequences):
            if i == 0:
                sequences[i] = '\x01' * w + seq
            else:
                sequences[i] = '\x02' * w + seq
        
        sequences[-1] += '\x01' * w

        for i in range(len(sequences) - 1):
            sequences[i] += '\x02' * w
        
        # print(sequences)
        
        return sequences


# Add some user input, command line options, etc. ##################################################
w = 10
p = 100


programStart = startTime = time.time()
# sequences = read_fasta(r"SARS.25k.fa", w)
# sequences = read_fasta(r"sequences.fasta", w)
sequences = read_fasta(r"/blue/boucher/eferro1/analysis/fasta/Chr21.10.reheader_consensus.fasta", w)
endTime = time.time()
print(f"Execution Time for reading file: {endTime - startTime:.4f}")


# Create necessary datastructures
dict = cudf.DataFrame({'phrases': cudf.Series(dtype='str'), 'hashes': cudf.Series(dtype='int64')})
parse = cudf.Series(dtype='int64')

print(f"Number of sequences to process: {len(sequences)}\n")

count = 0
for sequence in sequences:
    count += 1
    print(f"Sequence: {count}")
    startTime = time.time()
    # This has been accurate so far, double check
    num_windows = len(sequence) - w + 1

    # Turn the sequence into ints for finding trigger string (rolling hash)
    sequenceAsInt = cp.array(bytearray(sequence.encode('utf-8')), dtype=cp.uint8)
    # Turn sequence into byte array for parsing on GPU
    # sequenceAsChar = cp.array(bytearray(sequence.encode('utf-8')), dtype=cp.uint8)

    # Store results for trigger string finding. Cupy array prefilled with 0s based on num windows.
    result = cp.zeros(num_windows, dtype=cp.uint8)
    endTime = time.time()
    print(f"Execution Time for trigger string preprocess: {endTime - startTime:.4f}")

    # Need to figure out how to best determine threads per block, etc. ###############################################
    threads_per_block = 128
    blocks_per_grid = (num_windows + threads_per_block - 1) // threads_per_block

    # Call to trigger string finder GPU function, maybe generate prime (31) randomly every time program is called.
    startTime = endTime
    trigger_string_finder[blocks_per_grid, threads_per_block](sequenceAsInt, 31, w, p, result)
    endTime = time.time()
    print(f"Execution Time for trigger string GPU: {endTime - startTime:.4f}")

    # Find all positions at which there is a trigger string. The index in the result matches up to the start position of the trigger string in the seq.
    # Basically, if result[i] == 1, then seq[i:i+w] is the trigger string.
    startTime = endTime
    triggerStrings = cp.where(result == 1)[0]
    endTime = time.time()
    print(f"Execution Time for trigger string positions post processing: {endTime - startTime:.4f}")

    # Need to determine some values so we can create an n x m array of the necessary size, where n is num trigger strings and m is longest phrase for the GPU since dynamic allocation doesnt work :(
    # Num of phrases is easy. Maximum length is the longest distance between trigger string starts + w.
    num_phrases = len(triggerStrings) - 1

    if num_phrases > 1:
        # max_len = max(triggerStrings[i+1] - triggerStrings[i] for i in range(num_phrases)) + w
        # print(num_phrases)
        # print(max_len)

        # Create the cupy array for the GPU, uint8 works for the first bc theyre chars between 0-255,
        # d_phrases = cp.full((num_phrases, max_len), 255, dtype=cp.uint8)
        # d_phrases = cp.empty((num_phrases, max_len), dtype=cp.uint8)
        # offsets = cp.empty(num_phrases, dtype=cp.uint64)

        # Need to figure out how to best determine threads per block, etc. ###############################################
        threads_per_block = 128
        blocks_per_grid = (num_phrases + (threads_per_block - 1)) // threads_per_block

        # Call to phrase parser GPU function.
        startTime = endTime
        # dictionary_parser[blocks_per_grid, threads_per_block](triggerStrings.values, w, sequenceAsChar, d_phrases, offsets)
        phrases = []
        triggerStrings = triggerStrings.get()
        for i in range(num_phrases):
            phrases.append(sequence[triggerStrings[i]:triggerStrings[i+1]+w])
        phrases_series = cudf.Series(phrases)
        endTime = time.time()
        print(f"Execution Time for phrase parsing CPU: {endTime - startTime:.4f}")
        # print(d_phrases.get())

        # Because GPU needs strings as bytes, turn the phrases back into strings and cut off excess. Using 255 since its not an issue for fasta's, but may need to revisit this later.
        # startTime = time.time()
        # # phrases = ["".join(chr(x) for x in row if x != 0) for row in d_phrases.get()]
        # # phrases_series = cudf.Series(cudf.DataFrame.from_records(d_phrases).apply(lambda row: row.str.cat(sep=''), axis=1))
        # endTime = time.time()
        # print(f"Execution Time for turning phrases to strings: {endTime - startTime:.2f}")
        # print(phrases_series)
        # del d_phrases
    else:
        phrases_series = cudf.Series([sequence])

    # Hashes the phrases with murmurhash3 using cudf built-in function and outputs a series of hashes. Also create dict for curr seq
    startTime = endTime
    curr_parse = phrases_series.hash_values()
    curr_dict = cudf.DataFrame({'phrases': phrases_series, 'hashes': curr_parse})
    endTime = time.time()
    print(f"Execution Time for hashing phrases & creating dict: {endTime - startTime:.4f}")
    
    # Combine the dict and parse from curr seq to the final, preserving order for the parse and removing duplicates from the dict.
    startTime = endTime
    parse = cudf.concat([parse, curr_parse], ignore_index=True)
    endTime = time.time()
    print(f"Execution Time for concat parse: {endTime - startTime:.4f}")

    startTime = endTime
    dict = cudf.concat([dict, curr_dict], ignore_index=True)
    endTime = time.time()
    print(f"Execution Time for concat dict: {endTime - startTime:.4f}")

    startTime = endTime
    dict = dict.drop_duplicates(subset=['hashes']).reset_index(drop=True)
    endTime = time.time()
    print(f"Execution Time for keeping unique dict values: {endTime - startTime:.4f}")

    # Memory Collection
    del result
    del phrases
    del triggerStrings
    del phrases_series
    del curr_parse
    del curr_dict
    print("\n\n")

# Final step is to sort the dictionary, create a mapping to replace hashes with indices in the parse, then drop hashes from dict.
startTime = endTime
dict = dict.sort_values(by='phrases', ignore_index=True)
endTime = time.time()
print(f"Execution Time for sorting dict: {endTime - startTime:.4f}")

startTime = endTime
mapping = cudf.Series(dict.index, index=dict['hashes'])
endTime = time.time()
print(f"Execution Time for creating mapping: {endTime - startTime:.4f}")

dict = dict.drop(columns=['hashes'])

startTime = endTime
parse = parse.map(mapping)
endTime = time.time()
print(f"Execution Time for applying mapping: {endTime - startTime:.4f}")

# print(dict)
# print(parse)
startTime = time.time()
with open('dict.txt', 'w', encoding='utf-8') as file:
    file.write(dict.to_string())
with open('parse.txt', 'w', encoding='utf-8') as file:
    file.write(parse.to_string())
endTime = time.time()
print(f"Write time: {endTime - startTime:.4f}\n\n")

print(f"Total program runtime:{endTime - programStart:.4f}")

# cudf.core.column.as_column(d_phrases.ravel(), dtype="str")