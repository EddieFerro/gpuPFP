import cupy as cp
from numba import cuda
import cudf
import time
import rmm
import rmm.statistics
import shutil
import gzip
import argparse
import os
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.numba import RMMNumbaManager
import pylibcudf as plc

cuda.set_memory_manager(RMMNumbaManager)
cp.cuda.set_allocator(rmm_cupy_allocator)

free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
print(f"GPU memory (CuPy): {free_bytes/1e9:.2f} GB free / {total_bytes/1e9:.2f} GB total")
pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(),initial_pool_size=150*2**30, maximum_pool_size=190*2**30)
rmm.mr.set_current_device_resource(pool)
rmm.statistics.enable_statistics()


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
            result[start] = w

@cuda.jit
def sequence_expander(sequence, positions, shifts, w, output):
    tid = cuda.grid(1)
    if tid >= sequence.size:
        return
    pos = tid + shifts[tid]
    output[pos] = sequence[tid]

    if positions[tid]:
        for i in range(1,w):
            if tid + i < sequence.size:
                output[pos + i] = sequence[tid + i]
        if tid + w < sequence.size:
            output[pos + w] = sequence[tid]


# Will need a better method for reading fasta file probably. Maybe read one sequence at a time with yield in main.
# Written by Tyler Pencinger
def read_fasta(file_path, w, lineLimit=1000000):
        # Reads a FASTA file and return a list of sequences
        firstSeq = True
        byteCount = 0
        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        
        if ext in '.gz':
            file = gzip.open(file_path, 'rt')
        else:
            file = open(file_path, 'r')

        # with open(file_path, 'r') as file:
        buffer = bytearray()
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                continue
                # if current_seq:
                #     sequence = ''.join(current_seq)
                #     if firstSeq:
                #         firstSeq = False
                #         yield '\x01' * w + sequence + '\x02' * w
                #     else:
                #         yield '\x02' * w + sequence + '\x02' * w
                #     current_seq.clear()
                #     byteCount = 0
            buffer.extend(line.encode('utf-8'))
            byteCount += len(line)
            if byteCount/total_bytes > 0.02:
                start_pad = bytearray(b'\x01' * w) if firstSeq else bytearray(b'\x02' * w)
                end_pad = bytearray(b'\x02' * w)
                yield start_pad + buffer + end_pad
                buffer.clear()
                byteCount = 0
                firstSeq = False
        if buffer:
            start_pad = bytearray(b'\x01' * w) if firstSeq else bytearray(b'\x02' * w)
            end_pad = bytearray(b'\x01' * w)
            yield start_pad + buffer + end_pad
        file.close()        


def gpuPFP(input, w, p, tmp_path):
    tempFileCount = 0
    dictionary = cudf.DataFrame({'phrases': cudf.Series(dtype='str'), 'hashes': cudf.Series(dtype='int64')})
    parse = cudf.Series(dtype='int32')

    programStart = time.time()

    for sequence in read_fasta(input, w):
        seq_len = len(sequence)

        # Turn the sequence into ints for finding trigger string (rolling hash)
        # sequenceAsInt = cp.array(bytearray(sequence.encode('utf-8')), dtype=cp.uint8)
        sequenceAsInt = cp.array(sequence, dtype=cp.uint8)

        # Store results for trigger string finding. Cupy array prefilled with 0s based on num windows.
        result = cp.zeros(seq_len, dtype=cp.uint8)

        threads_per_block = 128
        blocks_per_grid = (seq_len + threads_per_block - 1) // threads_per_block

        # Call to trigger string finder GPU function, maybe generate prime (31) randomly every time program is called.
        trigger_string_finder[blocks_per_grid, threads_per_block](sequenceAsInt, 31, w, p, result)

        # Find all positions at which there is a trigger string. The index in the result matches up to the start position of the trigger string in the seq.
        # Basically, if result[i] == 1, then seq[i:i+w] is the trigger string.
        triggerStrings = cp.where(result == w)[0]
        # Need to determine some values so we can create an n x m array of the necessary size, where n is num trigger strings and m is longest phrase for the GPU since dynamic allocation doesnt work :(
        # Num of phrases is easy. Maximum length is the longest distance between trigger string starts + w.
        num_phrases = triggerStrings.size - 1

        # Get the phrases using trigger string positions
        # Blows up GPU memory currently
        if num_phrases > 1:
            # Reset so that beginning and end isnt duplicated by kernel
            result[0] = 0
            result[-w] = 0
            # Get the shift for each character based on expansion
            temp = cp.cumsum(result)
            shifts = temp - result
            # Set up output array
            output_len = seq_len + int(temp[-1])
            output = cp.empty(output_len, dtype=cp.uint8)
            # print(sequenceAsInt.get())
            # print(shifts.get())
            # print(result.get())
            # print(triggerStrings.get())
            blocks_per_grid = (seq_len + (threads_per_block - 1)) // threads_per_block
            offsets = triggerStrings + cp.arange(triggerStrings.size) * w
            sequence_expander[blocks_per_grid, threads_per_block](sequenceAsInt, result, shifts, w, output)
            # print(offsets.get())
            # print(output.get())
            
            offsets_column = plc.Column(
                data_type=plc.DataType(plc.TypeId.INT32),
                size=len(offsets),
                data=plc.gpumemoryview(offsets.astype(cp.int32)),
                mask=None,
                null_count=0,
                offset=0,
                children=[]
            )

            strings_column = plc.Column(
                data_type=plc.DataType(plc.TypeId.STRING),
                size=offsets.size-1,
                data=plc.gpumemoryview(output),
                mask=None,
                null_count=0,
                offset=0,
                children=[offsets_column]
            )

            phrases_series = cudf.Series.from_pylibcudf(strings_column)

            del triggerStrings
            del offsets
            del temp
            del result
            del shifts
            del output
        else:
            offsets = cp.array([0, sequenceAsInt.size], dtype=cp.int32)
            offsets_column = plc.Column(
                data_type=plc.DataType(plc.TypeId.INT32),
                size=offsets.size,
                data=plc.gpumemoryview(offsets),
                mask=None,
                null_count=0,
                offset=0,
                children=[]
            )
            strings_column = plc.Column(
                data_type=plc.DataType(plc.TypeId.STRING),
                size=offsets.size - 1,
                data=plc.gpumemoryview(sequenceAsInt),
                mask=None,
                null_count=0,
                offset=0,
                children=[offsets_column]
            )
            phrases_series = cudf.Series.from_pylibcudf(strings_column)
        del sequence
        
        # Hashes the phrases with murmurhash3 using cudf built-in function and outputs a series of hashes. Also create dict for curr seq
        curr_parse = phrases_series.hash_values()
        curr_dict = cudf.DataFrame({'phrases': phrases_series, 'hashes': curr_parse})
        del phrases_series
        # parse = cudf.concat([parse, curr_parse], ignore_index=True)

        # tempFileCount += 1
        # curr_parse.to_frame(name="parse").to_parquet(f"temp/tempfile_{tempFileCount}.parquet")
        parse = cudf.concat([parse, curr_parse], ignore_index=True)
        dictionary = cudf.concat([dictionary, curr_dict], ignore_index=True)

        del curr_parse
        del curr_dict

        dictionary = dictionary.drop_duplicates(subset=['hashes']).reset_index(drop=True)
        # print(dict.to_pandas())

        current_bytes = rmm.statistics.get_statistics().current_bytes
        if current_bytes/total_bytes > 0.15:
            # print("File dump.")
            tempFileCount += 1
            parse.to_frame(name="parse").to_parquet(f"{tmp_path}tempfile_{tempFileCount}.parquet")
            del parse
            parse = cudf.Series(dtype='int32')


    # Final step is to sort the dictionary, create a mapping to replace hashes with indices in the parse, then drop hashes from dict.
    dictionary = dictionary.sort_values(by='phrases', ignore_index=True)
    mapping = cudf.Series(dictionary.index, index=dictionary['hashes'])
    # hashes = mapping.index.values.astype("uint32")
    # ranks = mapping.values.astype("uint32")
    # del mapping
    # order = cp.argsort(hashes)
    # hashes = hashes[order]
    # ranks = ranks[order]
    dictionary = dictionary.drop(columns=['hashes'])
    # parse = parse.map(mapping)
    endTime = time.time()
    print(f"Total program runtime:{endTime - programStart:.4f}")
    
    # return dict, parse, tempFileCount, hashes, ranks
    return dictionary, parse, tempFileCount, mapping

def get_file_prefix(filepath):

    filename = os.path.basename(filepath)

    name, ext = os.path.splitext(filename)
    
    if ext in ['.gz', '.bz2', '.xz']:
        name, _ = os.path.splitext(name)
    
    return name

# Add comprehensive timer option

def main():
    parser = argparse.ArgumentParser(description="Placeholder Description", formatter_class=argparse.RawTextHelpFormatter)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--fasta', help="path to input fasta file",type=str)
    input_group.add_argument('-t', '--text', help="path to input text file",type=str)
    parser.add_argument('-w', '--wsize', help="sliding window size", default=5, type=int)
    parser.add_argument('-p', '--mod', help="hash modulus", default=15, type=int)
    parser.add_argument('-o', '--output', help="output files prefix", type=str)
    parser.add_argument('-d', '--tmp', default="tmp/", help="directory for temporary files", type=str)

    args = parser.parse_args()
    
    if not args.fasta:
        file = args.text
    else:
        file = args.fasta
    
    if not os.path.isdir(args.tmp):
        os.makedirs(args.tmp)

    if not args.output:
        output_prefix = get_file_prefix(args.input)
    else:
        output_prefix = args.output

    # dict, parse, tempFileCount, hashes, ranks = gpuPFP(file, args.wsize, args.mod)
    dictionary, parse, tempFileCount, mapping = gpuPFP(file, args.wsize, args.mod, args.tmp)

    dictionary[['phrases']].to_csv(
        f"{output_prefix}.dict",
        index=False,        # no index column
        header=False,       # no header row
        sep=",",            # unused (single column → no commas)
        lineterminator="\x03"
    )
    with open(f"{output_prefix}.dict", "r+b") as f:
        f.seek(-1, os.SEEK_END)  # back up one byte from EOF
        f.truncate()       
    del dictionary

    with open(output_prefix + '.parse', 'wb') as fh:
        for i in range(1, tempFileCount + 1):
            # Read back this chunk from Parquet:
            temp_df = cudf.read_parquet(f"{args.tmp}tempfile_{i}.parquet")
            # temp_df["parse"] is int64 (the hash in that chunk)
            # Map → its “rank” via our mapping Series
            ranks64 = temp_df["parse"].map(mapping)
            ranks32 = ranks64.astype('uint32').to_cupy()

            # Write the little-endian 4-byte reps for this chunk
            ranks32.tofile(fh)

            # Delete the temp file immediately
            os.remove(f"{args.tmp}tempfile_{i}.parquet")
        # print(rmm.statistics.get_statistics())

        ranks64 = parse.map(mapping)
        ranks32 = ranks64.astype('uint32').to_cupy()  # now each element is 4 bytes
        ranks32.tofile(fh)
        fh.close()
    # shutil.rmtree(args.tmp)



    # with open(output_prefix + '.parse', 'wb') as fh:
    #     for i in range(1, tempFileCount + 1):
    #         # Read back this chunk from Parquet:
    #         temp_df = cudf.read_parquet(f"temp/tempfile_{i}.parquet")
    #         # temp_df["parse"] is int64 (the hash in that chunk)
    #         # Map → its “rank” via our mapping Series
    #         temp_parse = temp_df["parse"].astype("uint32").values

    #         idxs = cp.searchsorted(hashes, temp_parse)

    #         # Cast to 32-bit unsigned before dumping bytes:
    #         final = ranks[idxs]  # now each element is 4 bytes

    #         # Write the little-endian 4-byte reps for this chunk
    #         fh.write(final.tobytes())

    #         # Delete the temp file immediately
    #         os.remove(f"temp/tempfile_{i}.parquet")
    #     print(rmm.statistics.get_statistics())
    #     temp_parse = parse.astype("uint32").values
    #     idxs = cp.searchsorted(hashes, temp_parse)
    #     final = ranks[idxs]

    #     # ranks64 = parse.map(mapping)
    #     # ranks32 = ranks64.astype('uint32')  # now each element is 4 bytes
    #     fh.write(final.tobytes())



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
