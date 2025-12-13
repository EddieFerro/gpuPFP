import cupy as cp
from numba import cuda
import rmm
import rmm.statistics
import cudf
import time
import gc
import argparse
import sys
import os
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.numba import RMMNumbaManager
import pylibcudf as plc
import kvikio
import kvikio.defaults

free_bytes = None
total_bytes = None

# GPU setup function
# Initializes RMM and sets memory managers
# Exits if no GPU is detected
def gpu_setup():
    global free_bytes, total_bytes
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        # cuda.set_memory_manager(RMMNumbaManager)
        # cp.cuda.set_allocator(rmm_cupy_allocator)
        # pool = rmm.mr.PoolMemoryResource(
        #     rmm.mr.ManagedMemoryResource(),
        #     initial_pool_size="50GiB")
        # up = rmm.mr.ManagedMemoryResource()
       
        # mr = rmm.mr.ArenaMemoryResource(rmm.mr.CudaAsyncMemoryResource(initial_pool_size=0), arena_size=int(total_bytes*0.90))
        # ar = rmm.mr.ArenaMemoryResource(up)
        up = rmm.mr.CudaAsyncMemoryResource()
        rmm.mr.set_current_device_resource(up)
        rmm.statistics.enable_statistics()
        cuda.set_memory_manager(RMMNumbaManager)
        cp.cuda.set_allocator(rmm_cupy_allocator)
        cudf.set_option("memory_profiling", True)
    except cp.cuda.runtime.CUDARuntimeError:
        print("ERROR: No NVIDIA GPU detected or CUDA not available.")
        sys.exit(1)

#identify all trigger strings in the sequence
@cuda.jit
def trigger_string_finder(sequence, prime, w, p, result):
    idx = cuda.grid(1)
    start = idx
    if start + w <= sequence.size:
        hash_val = 0
        for i in range(w):
            # simple rolling hash
            hash_val = (hash_val * prime + sequence[start + i]) & 0xFFFFFFFF
        # check modulo condition
        if hash_val % p == 0 or start == 0 or start+w == sequence.size:
            result[start] = 1

# binary search to find number of duplicated trigger strings that have started by output index x
@cuda.jit(device=True)
def _upper_bound_duplications(S, m, w, x):
    lo = 0
    hi = m
    while lo < hi:
        mid = (lo + hi) // 2
        # y = start of the trigger string at mid in the expanded sequence
        y = S[mid] + w + mid * w
        if y <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo  # # of gaps whose start <= x

# expand the sequence by duplicating trigger strings
@cuda.jit
def expand_sequence(input, n, S, m, w, out):
    j = cuda.grid(1)
    # output length
    L = n + m * w
    if j >= L:
        return

    # u is number of duplications that have started by output index j
    u = _upper_bound_duplications(S, m, w, j)

    if u == 0:
        # No duplications started yet, straight copy
        i = j
    else:
        # last duplicated trigger string that has started up to index j
        k = u - 1  
        # position of the start of trigger string k in the expanded sequence
        y = S[k] + w + k * w 
        if j < y + w:
            # output index j lies inside duplicated window of trigger string k, take from original trigger string
            i = S[k] + (j - y)
        else:
            # output index j is from original sequence after u duplications, then shift back by u*w
            i = j - u * w
    out[j] = input[i]

# scatter ranks to output positions
@cuda.jit
def scatter_row_index(offsets, positions, ranks, out):
    r = cuda.grid(1)  # row id
    n_rows = offsets.size - 1
    if r >= n_rows:
        return

    start = offsets[r]
    stop  = offsets[r + 1]
    rank = ranks[r]

    for k in range(start, stop):
        p = positions[k]
        if 0 <= p < out.size:
            out[p] = rank

# read fasta file in chunks
# yields sequence chunks with padding
# each chunk is default 10% of free GPU memory and is read in portions of 10% of the chunk size
# to do: add read_csv functionality to read fasta files without preprocessing
def read_fasta_gpu(file_path, w, threshold = 0.10):
    # Ensures that the library drops to POSIX calls when GPU Direct Storage is not available
    # kvikio.defaults.set("compat_mode", kvikio.CompatMode.AUTO)
    print("Starting read_fasta_gpu")
    total_time = 0
    batch_size = int(total_bytes*threshold)
    read_size = int(batch_size*0.10)

    start = time.time()
    total_size = os.path.getsize(file_path)
    chunk_bytes = min(read_size, total_size)
    batch_bytes = min(batch_size, max(chunk_bytes, total_size))

    buf = cp.empty(batch_bytes, dtype=cp.uint8)
    firstSeq = True
    ones_pad = cp.full((w,), 1, dtype=cp.uint8)
    twos_pad = cp.full((w,), 2, dtype=cp.uint8)

    with kvikio.CuFile(file_path, 'r') as f:
        read_total = 0
        filled = 0
        while read_total < total_size:
            start = time.time()
            to_read = min(chunk_bytes, total_size - read_total)
            n = f.read(buf[filled:filled+to_read], file_offset=read_total)
            end = time.time()
            total_time += (end-start)
            
            read_total += n
            filled += n
            eof = (read_total == total_size)

            if filled == batch_bytes or eof:
                prefix = ones_pad if firstSeq else twos_pad
                firstSeq = False
                suffix = ones_pad if eof else twos_pad
                out = cp.concatenate([prefix, buf[:filled], suffix])
                
                yield out, eof
                
                filled = 0
    print(f"Total read time:{total_time:.4f}")

# process a chunk to find trigger strings and expand the sequence
# returns expanded sequence and offsets dividing phrases
def process_chunk(sequence, w, p):
    seq_len = len(sequence)

    start = time.time()

    # find trigger strings
    result = cp.zeros(seq_len, dtype=cp.uint8)
    threads_per_block = 128
    blocks_per_grid = (seq_len + threads_per_block - 1) // threads_per_block
    trigger_string_finder[blocks_per_grid, threads_per_block](sequence, 31, w, p, result)

    end = time.time()
    print(f"Trigger String finder time: {(end-start):.4f}")

    start = time.time()

    # extract trigger string positions
    triggerStrings = cp.where(result)[0]

    del result
    gc.collect()
    cp.cuda.Device().synchronize()

    num_phrases = triggerStrings.size - 1

    if num_phrases > 1:

        # compute lengths of phrases and offsets
        lens = cp.empty(num_phrases, dtype=cp.uint16)
        diffs = triggerStrings[1:] - triggerStrings[:-1]
        lens = diffs + w
        offset_chunk = cp.empty(num_phrases + 1, dtype=cp.uint64)
        offset_chunk[0] = 0
        offset_chunk[1:] = cp.cumsum(lens)

        del lens
        del diffs
        gc.collect()
        cp.cuda.Device().synchronize()

        # expand sequence by duplicating trigger strings
        expanded_len = seq_len + (num_phrases-1) * w
        triggerStrings = triggerStrings[1:-1]
        seq_chunk = cp.empty(expanded_len, dtype=cp.uint8)
        blocks_per_grid = (expanded_len + (threads_per_block - 1)) // threads_per_block
        expand_sequence[blocks_per_grid, threads_per_block](sequence, seq_len, triggerStrings, (num_phrases-1), w, seq_chunk)

        del sequence
        gc.collect()
        cp.cuda.Device().synchronize()
    else:   
        # no trigger strings found, no expansion
        seq_chunk = sequence
        offset_chunk = cp.array([0, seq_chunk.size], dtype=cp.uint64)
    end = time.time()
    print(f"Expand sequence time: {(end-start):.4f}")
        
    return seq_chunk, offset_chunk
    
# build dictionary for a chunk
def build_chunk_dict(seq_chunk, offset_chunk, cum_offset, tmp_path, tmp_file_count, write = False):
    start = time.time()

    # Create cudf DataFrame from sequence and offsets using pylibcudf columns
    offsets_column = plc.Column(
        data_type=plc.DataType(plc.TypeId.INT64),
        size=offset_chunk.size,
        data=plc.gpumemoryview(offset_chunk),
        mask=None,
        null_count=0,
        offset=0,
        children=[]
    )
    strings_column = plc.Column(
        data_type=plc.DataType(plc.TypeId.STRING),
        size=offset_chunk.size - 1,
        data=plc.gpumemoryview(seq_chunk),
        mask=None,
        null_count=0,
        offset=0,
        children=[offsets_column]
    )
    phrases_series = cudf.Series.from_pylibcudf(strings_column)
    df = cudf.DataFrame({'phrases': phrases_series})

    # Get unique indices for each phrase and adjust by cumulative offset, update cumulative offset
    df.reset_index(drop=False, inplace=True)
    df['index'] = df['index'].astype("uint64")
    max_val = df['index'].max()
    df['index'] = df['index'] + cp.uint64(cum_offset)
    cum_offset += max_val

    # Group by phrases to collect indices
    dictionary = df.groupby(by='phrases').agg({'index':'collect'})

    dictionary.reset_index(drop=False, inplace=True)

    # Write parquet file if there are multiple chunks
    if write:
        end = time.time()
        print(f"Build chunk dictionary time: {(end-start):.4f}")
        
        start = time.time()
        tmp_file = f"{tmp_path}/temp_{tmp_file_count}.parquet"
        dictionary.to_parquet(tmp_file)
        end = time.time()
        print(f"To parquet time: {(end-start):.4f}")

        del dictionary
        del df
        gc.collect()
        cp.cuda.Device().synchronize()
        return tmp_file, cum_offset
    else:
        # Return final dictionary if single chunk
        dictionary = dictionary.sort_values(by='phrases')
        end = time.time()
        print(f"Build final dictionary time: {(end-start):.4f}")
        del df
        gc.collect()
        cp.cuda.Device().synchronize()
        return dictionary, cum_offset

# build final dictionary from temporary chunk files
def build_final_dict(tmp_files):
    # free, _ = cp.cuda.runtime.memGetInfo()
    # # print(free)
    dictionary = None
    total = 0
    total_read = 0

    for file in tmp_files:
        # handle first file separately
        if dictionary is None:
            start = time.time()
            dictionary = cudf.read_parquet(file, columns=["phrases"])
            end = time.time()
            total_read += (end - start)
        else:
            # read next chunk dictionary
            start = time.time()
            temp = cudf.read_parquet(file, columns=["phrases"])
            end = time.time()
            total_read += (end - start)

            start = time.time()

            # merge with existing dictionary and drop duplicates
            combined = cudf.concat([dictionary, temp], ignore_index=True)
            dictionary = combined.drop_duplicates(subset="phrases").reset_index(drop=True)

            end = time.time()
            total += (end-start)

            del combined
            del temp
        gc.collect()
        cp.cuda.runtime.deviceSynchronize()
        # free, _ = cp.cuda.runtime.memGetInfo()
        # dictionary.reset_index(drop=False, inplace=True)
        # print(free)
    # final sort of dictionary
    start = time.time()
    dictionary = dictionary.sort_values(by='phrases')
    end = time.time()
    total += (end-start)

    print(f"Dictionary read parquet time: {(total_read):.4f}")
    print(f"Build final dictionary time: {(total):.4f}")
    cp.cuda.Device().synchronize()
    return dictionary

# build parse from temporary chunk files and final dictionary
def build_parse(dictionary, tmp_files, out):
    total = 0
    total_read = 0

    for file in tmp_files:
        # read chunk parse file
        start = time.time()
        temp = cudf.read_parquet(file)
        end = time.time()
        total_read += (end - start)

        # merge with final dictionary to get ranks
        start = time.time()
        joined = temp.merge(dictionary, on="phrases",how="left")

        # scatter ranks to output positions
        # flat array of all positions of phrases (length = total_positions)
        pos_leaves = joined["index"].list.leaves
        # positions per row (length = n_rows)
        counts = joined["index"].list.len()
        # row ranks (length = n_rows)
        ranks = joined["rank"].values

        # Build offsets on the GPU (length n_rows+1)
        offs = cp.empty(len(counts) + 1, dtype=cp.uint32)
        offs[0] = 0
        cp.cumsum(counts.values, out=offs[1:])

        # Output each phrase rank to its positions
        threads = 128
        blocks  = (len(counts) + threads - 1) // threads
        scatter_row_index[blocks, threads](offs, pos_leaves.values, ranks, out)
        
        end = time.time()
        total += (end-start)

        del joined
        del temp
        del pos_leaves
        del ranks
        del offs
        del counts
        gc.collect()
        cp.cuda.runtime.deviceSynchronize()
        # free, _ = cp.cuda.runtime.memGetInfo()
        # print(free)
        os.remove(file)
    print(f"Parse read parquet time: {(total_read):.4f}")
    print(f"Construct parse time: {(total):.4f}")
    return out

# main gpuPFP function
def gpuPFP(input, args):
    
    w = args.wsize
    p = args.mod
    tmp_path = args.tmp_dir
    threshold = args.threshold
    output_prefix = args.output

    # Temporary files for chunk dictionaries
    tmp_file_count = 0
    tmp_files = []

    # Cumulative offset for phrase indices across chunks
    cum_offset = 0

    # Iterate through chunks of input from fasta file
    for sequence, eof in read_fasta_gpu(input, w, threshold):
        
        # Process chunk to get phrases and offsets
        seq_chunk, offset_chunk = process_chunk(sequence, w, p)

        # Build dictionary for chunk
        if not eof or tmp_file_count > 0:
            # print("Temporary file hit")
            tmp_file, cum_offset = build_chunk_dict(seq_chunk, offset_chunk, cum_offset, tmp_path, tmp_file_count, True)
            tmp_files.append(tmp_file)
            tmp_file_count += 1

            del seq_chunk
            del offset_chunk
            gc.collect()
            cp.cuda.Device().synchronize()

    # Build final dictionary
    if tmp_file_count > 0:
        dictionary = build_final_dict(tmp_files)
    else:
        dictionary, cum_offset = build_chunk_dict(seq_chunk, offset_chunk, cum_offset, tmp_path, tmp_file_count, False)
        del seq_chunk
        del offset_chunk
        gc.collect()
        cp.cuda.Device().synchronize()
       
    # Output dictionary to file
    # todo: add option for parquet output
    start = time.time()

    dictionary.to_csv(
        f"{output_prefix}.dict",
        columns="phrases",
        index=False,        
        header=False,       
        sep=",",            
        lineterminator="\x03"
    )
    with open(f"{output_prefix}.dict", "r+b") as f:
        f.seek(-1, os.SEEK_END)  
        f.truncate()

    end = time.time()
    print(f"Dictionary output time: {(end-start):.4f}")

    start = time.time()
    
    # Prepare for parse output
    dictionary = dictionary.reset_index(drop=True)
    dictionary["rank"] = dictionary.index.astype(cp.uint32)

    # Allocate output array for parse, adjust type based on size
    if cum_offset >= 2**32:
        out = cp.empty(int(cum_offset+1), dtype=cp.uint64)
    else:
        out = cp.empty(int(cum_offset+1), dtype=cp.uint32)

    end = time.time()
    print(f"Prepare parse output time: {(end-start):.4f}")

    # Build parse
    if tmp_file_count > 0:
        out = build_parse(dictionary, tmp_files, out)
    else:
        start = time.time()
        
        dictionary = dictionary.drop(columns="phrases")
        cp.cuda.Device().synchronize()
        
        # scatter ranks to output positions
        # flat array of all positions of phrases (length = total_positions)
        pos_leaves = dictionary["index"].list.leaves
        # positions per row (length = n_rows)
        counts = dictionary["index"].list.len()
        # row ranks (length = n_rows)
        ranks = dictionary["rank"].values

        # Build offsets on the GPU (length n_rows+1)
        offs = cp.empty(len(counts) + 1, dtype=cp.int32)
        offs[0] = 0
        cp.cumsum(counts.values, out=offs[1:])

        # Output each phrase rank to its positions
        threads = 128
        blocks  = (len(counts) + threads - 1) // threads
        scatter_row_index[blocks, threads](offs, pos_leaves.values, ranks, out)
        end = time.time()
        print(f"Construct parse time: {(end-start):.4f}")

        del pos_leaves
        del ranks
        del offs
        del counts
        gc.collect()
        cp.cuda.runtime.deviceSynchronize()

    
    del dictionary
    gc.collect()
    cp.cuda.Device().synchronize()

    # Output parse to file
    # todo: add option for parquet output
    start = time.time()
    with open(output_prefix + '.parse', 'w') as fh:
        out.tofile(fh)
        fh.close()
    end = time.time()
    print(f"Parse output time: {(end-start):.4f}")
    
    return None, None, tmp_file_count
    

def get_file_prefix(filepath):

    filename = os.path.basename(filepath)

    name, ext = os.path.splitext(filename)
    
    if ext in ['.gz']:
        print("Rapid-PFP does not currently support compressed input files.")
        sys.exit(1)
    
    return name

def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated PFP", formatter_class=argparse.RawTextHelpFormatter)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--fasta', help="Path to input fasta file",type=str)
    input_group.add_argument('-t', '--text', help="Path to input text file",type=str)
    parser.add_argument('-w', '--wsize', help="Sliding window size", default=5, type=int)
    parser.add_argument('-p', '--mod', help="Modulu used during parsing", default=15, type=int)
    parser.add_argument('-o', '--output', help="Output files prefix", type=str)
    parser.add_argument('-d', '--tmp-dir', default="tmp", help="Directory for temporary files", type=str)
    parser.add_argument('--threshold', default=0.10, help="Fraction of free GPU memory to use per input batch (0 < f ≤ 1).", type=float)

    programStart = time.time()
    args = parser.parse_args()
    if args.threshold <= 0 or args.threshold > 1:
        print("ERROR: Threshold must be in the range (0, 1].")
        sys.exit(1)
    
    gpu_setup()

    if not args.fasta:
        file = args.text
    else:
        file = args.fasta

    if not os.path.isdir(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    if not args.output:
        args.output = get_file_prefix(args.input)

    dictionary, parse, tmp_file_count = gpuPFP(file, args)

    print(f"Peak VRAM Bytes:{rmm.statistics.get_statistics().peak_bytes}")
    print(rmm.statistics.default_profiler_records.report())
    endTime = time.time()
    print(f"Total program runtime: {endTime - programStart:.4f}")


if __name__ == '__main__':
    main()
