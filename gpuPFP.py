import cupy as cp
from numba import cuda
import rmm
import rmm.statistics
import cudf
import time
import shutil
import gc
import glob
import gzip
import argparse
import os
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.numba import RMMNumbaManager
import pylibcudf as plc
free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
# rmm.reinitialize(pool_allocator=True)
print(f"GPU memory (CuPy): {free_bytes/1e9:.2f} GB free / {total_bytes/1e9:.2f} GB total")
# pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource())
pool = rmm.mr.CudaAsyncMemoryResource()
# rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource())
rmm.mr.set_current_device_resource(pool)

rmm.statistics.enable_statistics()
cudf.set_option("memory_profiling", True)
cuda.set_memory_manager(RMMNumbaManager)
cp.cuda.set_allocator(rmm_cupy_allocator)



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
        start = time.time()

        buffer = bytearray()
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                continue
                # if buffer:
                #     start_pad = bytearray(b'\x01' * w) if firstSeq else bytearray(b'\x02' * w)
                #     end_pad = bytearray(b'\x02' * w)
		
                #     end = time.time()
                #     print(f"Read one sequence runtime:{end - start:.4f}")
		
                #     yield start_pad + buffer + end_pad
		
                #     start = time.time()
		
                #     buffer.clear()
                #     byteCount = 0
                #     firstSeq = False
            buffer.extend(line.encode('utf-8'))
            byteCount += len(line)
            if byteCount >= 524288000:
                # print(byteCount)
                start_pad = bytearray(b'\x01' * w) if firstSeq else bytearray(b'\x02' * w)
                end_pad = bytearray(b'\x02' * w)

                end = time.time()
                # print(f"Read one sequence runtime:{end - start:.4f}")

                yield start_pad + buffer + end_pad

                start = time.time()

                buffer.clear()
                byteCount = 0
                firstSeq = False
        if buffer:
            start_pad = bytearray(b'\x01' * w) if firstSeq else bytearray(b'\x02' * w)
            end_pad = bytearray(b'\x01' * w)

            end = time.time()
            # print(f"Read one sequence runtime:{end - start:.4f}")

            yield start_pad + buffer + end_pad
        file.close()        

def batch_process(phrase_chunks, offset_chunks, tmp_path, tempFileCount):
    # start = time.time()
    seq_arr = cp.concatenate(phrase_chunks)
    offsets_arr = cp.concatenate(offset_chunks)
    # end = time.time()
    # print(f"Concatenate char and offset arrays:{end - start:.4f}")

    # start = time.time()
    offsets_column = plc.Column(
        data_type=plc.DataType(plc.TypeId.INT32),
        size=offsets_arr.size,
        data=plc.gpumemoryview(offsets_arr),
        mask=None,
        null_count=0,
        offset=0,
        children=[]
    )
    strings_column = plc.Column(
        data_type=plc.DataType(plc.TypeId.STRING),
        size=offsets_arr.size - 1,
        data=plc.gpumemoryview(seq_arr),
        mask=None,
        null_count=0,
        offset=0,
        children=[offsets_column]
    )
    phrases_series = cudf.Series.from_pylibcudf(strings_column)
    # end = time.time()
    # print(f"Create cudf string series batch:{end - start:.4f}")

    # start = time.time()
    parse = phrases_series.hash_values()
    # end = time.time()
    # print(f"Hash values batch:{end - start:.4f}")

    # start = time.time()
    dictionary = cudf.DataFrame({'phrases': phrases_series, 'hashes': parse})
    # end = time.time()
    # print(f"Create dataframe batch:{end - start:.4f}")

    # start = time.time()
    dictionary = dictionary.drop_duplicates(subset=['hashes']).reset_index(drop=True)
    # end = time.time()
    # print(f"Drop duplicates from dict batch:{end - start:.4f}")

    # print("File dump.")
    # start = time.time()
    parse = parse.to_frame(name="parse")
    parse.to_parquet(f"{tmp_path}/parse/{tempFileCount}.parquet")
    dictionary.to_parquet(f"{tmp_path}/dict/{tempFileCount}.parquet")
    # end = time.time()
    # print(f"Write temp files to parquet:{end - start:.4f}")


def build_dict(tmp_path, batch_size=2):
    dict_dir = os.path.join(tmp_path, "dict")
    all_files = sorted(glob.glob(os.path.join(dict_dir, "*.parquet")))
    dictionary = cudf.DataFrame({"phrases": cudf.Series(dtype="str"),
                                 "hashes":  cudf.Series(dtype="int64")})

    # process in batches
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i : i + batch_size]
        # read 1..batch_size files in one call
        df_batch = cudf.read_parquet(batch_files)
        # append
        dictionary = cudf.concat([dictionary, df_batch], ignore_index=True)
        # drop any duplicates seen so far
        dictionary = dictionary.drop_duplicates(subset=["hashes"])\
                               .reset_index(drop=True)
        # delete the on-disk chunks
        for f in batch_files:
            os.remove(f)

    # final sort by phrase text
    dictionary = dictionary.sort_values(by="phrases", ignore_index=True)
    # build the final mapping: hash → row index
    mapping = cudf.Series(dictionary.index, index=dictionary["hashes"])
    # drop the hashes column from the output dict
    dictionary = dictionary.drop(columns=["hashes"])

    return dictionary, mapping

def gpuPFP(input, w, p, tmp_path):
    
    tempFileCount = 0
    phrase_chunks = []
    offset_chunks = []
    cum_offset = 0
    
    # dictionary = cudf.DataFrame({'phrases': cudf.Series(dtype='str'), 'hashes': cudf.Series(dtype='int64')})
    # parse = cudf.Series(dtype='int32')

    programStart = time.time()
    for sequence in read_fasta(input, w):
        if cum_offset == 0:
            phrase_chunks.clear()
            offset_chunks.clear()
        seq_len = len(sequence)

        # Turn the sequence into ints for finding trigger string (rolling hash)
        # start = time.time()
        sequenceAsInt = cp.array(sequence, dtype=cp.uint8)
        # end = time.time()
        # print(f"Copy one sequence to device:{end - start:.4f}")
        del sequence

        # Store results for trigger string finding. Cupy array prefilled with 0s based on num windows.
        result = cp.zeros(seq_len, dtype=cp.uint8)

        threads_per_block = 128
        blocks_per_grid = (seq_len + threads_per_block - 1) // threads_per_block

        # Call to trigger string finder GPU function, maybe generate prime (31) randomly every time program is called.
        # start = time.time()
        trigger_string_finder[blocks_per_grid, threads_per_block](sequenceAsInt, 31, w, p, result)
        # end = time.time()
        # print(f"Trigger String finder one sequence:{end - start:.4f}")

        # Find all positions at which there is a trigger string. The index in the result matches up to the start position of the trigger string in the seq.
        # start = time.time()
        triggerStrings = cp.where(result == w)[0]
        # end = time.time()
        # print(f"Cupy where function timing:{end - start:.4f}")
        num_phrases = triggerStrings.size - 1

        # Get the phrases using trigger string positions
        # Blows up GPU memory currently
        if num_phrases > 1:
            # Reset so that beginning and end isnt duplicated by kernel
            # start = time.time()
            result[0] = 0
            result[-w] = 0
            # Get the shift for each character based on expansion
            temp = cp.cumsum(result)
            shifts = temp - result
            # Set up output array
            output_len = seq_len + int(temp[-1])
            del temp
            output = cp.empty(output_len, dtype=cp.uint8)
            # end = time.time()
            # print(f"Sequence expander setup:{end - start:.4f}")

            blocks_per_grid = (seq_len + (threads_per_block - 1)) // threads_per_block
            # start = time.time()
            sequence_expander[blocks_per_grid, threads_per_block](sequenceAsInt, result, shifts, w, output)
            # end = time.time()
            # print(f"Sequence expander one sequence:{end - start:.4f}")
            # print(offsets.get())
            # print(output.get())

            phrase_chunks.append(output)
            offsets = triggerStrings + cp.arange(triggerStrings.size) * w
            offsets += cum_offset
            offset_chunks.append(offsets.astype(cp.int32))
            cum_offset += output.size
        else:
            phrase_chunks.append(sequenceAsInt)
            offsets = cp.array([0, sequenceAsInt.size], dtype=cp.int32)
            offsets += cum_offset
            offset_chunks.append(offsets.astype(cp.int32))
            cum_offset += sequenceAsInt.size
        current_bytes = rmm.statistics.get_statistics().current_bytes
        if current_bytes/total_bytes > 0.10:
            tempFileCount += 1
            batch_process(phrase_chunks, offset_chunks, tmp_path, tempFileCount)
            cum_offset = 0

    if tempFileCount > 0:
        if phrase_chunks:
            tempFileCount += 1
            batch_process(phrase_chunks, offset_chunks, tmp_path, tempFileCount)
            cum_offset = 0
        dictionary, mapping = build_dict(tmp_path)
        endTime = time.time()
        print(f"Total program runtime:{endTime - programStart:.4f}")
        return dictionary, None, tempFileCount, mapping
    else:
        # start = time.time()
        seq_arr = cp.concatenate(phrase_chunks)
        offsets_arr = cp.concatenate(offset_chunks)
        # end = time.time()
        # print(f"Concatenate char and offset arrays:{end - start:.4f}")

        # start = time.time()
        offsets_column = plc.Column(
            data_type=plc.DataType(plc.TypeId.INT32),
            size=offsets_arr.size,
            data=plc.gpumemoryview(offsets_arr),
            mask=None,
            null_count=0,
            offset=0,
            children=[]
        )
        strings_column = plc.Column(
            data_type=plc.DataType(plc.TypeId.STRING),
            size=offsets_arr.size - 1,
            data=plc.gpumemoryview(seq_arr),
            mask=None,
            null_count=0,
            offset=0,
            children=[offsets_column]
        )
        phrases_series = cudf.Series.from_pylibcudf(strings_column)
        # end = time.time()
        # print(f"Create cudf string series batch:{end - start:.4f}")

        # start = time.time()
        parse = phrases_series.hash_values()
        # end = time.time()
        # print(f"Hash values batch:{end - start:.4f}")

        # start = time.time()
        dictionary = cudf.DataFrame({'phrases': phrases_series, 'hashes': parse})
        # end = time.time()
        # print(f"Create dataframe batch:{end - start:.4f}")
        # del phrases_series

        # start = time.time()
        dictionary = dictionary.drop_duplicates(subset=['hashes']).reset_index(drop=True)
        # end = time.time()
        # print(f"Drop duplicates from dict batch:{end - start:.4f}")

        # start = time.time()
        dictionary = dictionary.sort_values(by='phrases', ignore_index=True)
        # end = time.time()
        # print(f"Sort full dict:{end - start:.4f}")

        # start = time.time()
        mapping = cudf.Series(dictionary.index, index=dictionary['hashes'])
        # end = time.time()
        # print(f"Create mapping:{end - start:.4f}")

        # start = time.time()
        dictionary = dictionary.drop(columns=['hashes'])
        # end = time.time()
        # print(f"Drop Column:{end - start:.4f}")
        endTime = time.time()
        print(f"Total program runtime:{endTime - programStart:.4f}")
        return dictionary, parse, tempFileCount, mapping
    
    # # return dict, parse, tempFileCount, hashes, ranks
    # return dictionary, parse, tempFileCount, mapping

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
    parser.add_argument('-d', '--tmp', default="tmp", help="directory for temporary files", type=str)

    args = parser.parse_args()
    
    if not args.fasta:
        file = args.text
    else:
        file = args.fasta
    
    if not os.path.isdir(args.tmp):
        os.makedirs(args.tmp)
        os.makedirs(f"{args.tmp}/dict")
        os.makedirs(f"{args.tmp}/parse")

    if not args.output:
        output_prefix = get_file_prefix(args.input)
    else:
        output_prefix = args.output

    # dict, parse, tempFileCount, hashes, ranks = gpuPFP(file, args.wsize, args.mod)
    dictionary, parse, tempFileCount, mapping = gpuPFP(file, args.wsize, args.mod, args.tmp)

    # start = time.time()
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
    # end = time.time()
    # print(f"Write dictionary:{end - start:.4f}")


    # start = time.time()
    with open(output_prefix + '.parse', 'wb') as fh:
        if parse is not None:
            ranks64 = parse.map(mapping)
            ranks32 = ranks64.astype('uint32').to_cupy()  # now each element is 4 bytes
            ranks32.tofile(fh)
        else:
            for i in range(1, tempFileCount + 1):
                # Read back this chunk from Parquet:
                temp_df = cudf.read_parquet(f"{args.tmp}/parse/{i}.parquet")
                # temp_df["parse"] is int64 (the hash in that chunk)
                # Map → its “rank” via our mapping Series
                ranks64 = temp_df["parse"].map(mapping)
                ranks32 = ranks64.astype('uint32').to_cupy()

                # Write the little-endian 4-byte reps for this chunk
                ranks32.tofile(fh)

                # Delete the temp file immediately
                os.remove(f"{args.tmp}/parse/{i}.parquet")
        # print(rmm.statistics.get_statistics())

        
        fh.close()
    # end = time.time()
    # print(f"Write parse:{end - start:.4f}")
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


# sequenceAsInt = cp.array(bytearray(sequence.encode('utf-8')), dtype=cp.uint8)
