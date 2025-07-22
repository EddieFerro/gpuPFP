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
pool = rmm.mr.CudaAsyncMemoryResource()
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
        firstSeq = True
        byteCount = 0
        total_time = 0
        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        
        if ext in '.gz':
            file = gzip.open(file_path, 'rt')
        else:
            file = open(file_path, 'r')

        start = time.time()

        buffer = bytearray()
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                continue
            buffer.extend(line.encode('utf-8'))
            byteCount += len(line)
            if byteCount >= 524288000:
                # print(byteCount)
                start_pad = bytearray(b'\x01' * w) if firstSeq else bytearray(b'\x02' * w)
                end_pad = bytearray(b'\x02' * w)

                end = time.time()
                total_time += (end - start)

                yield start_pad + buffer + end_pad

                start = time.time()

                buffer.clear()
                byteCount = 0
                firstSeq = False
        if buffer:
            start_pad = bytearray(b'\x01' * w) if firstSeq else bytearray(b'\x02' * w)
            end_pad = bytearray(b'\x01' * w)

            end = time.time()
            total_time += (end - start)

            yield start_pad + buffer + end_pad
        print(f"Total read time:{total_time:.4f}")
        file.close()        

def batch_process(phrase_chunks, offset_chunks, tmp_path, tempFileCount):
    seq_arr = cp.concatenate(phrase_chunks)
    offsets_arr = cp.concatenate(offset_chunks)

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

    parse = cudf.DataFrame({'phrases': phrases_series})
    parse.to_parquet(f"{tmp_path}/parse/{tempFileCount}.parquet")

    dictionary = parse.drop_duplicates(subset='phrases').reset_index(drop=True)
    dictionary.to_parquet(f"{tmp_path}/dict/{tempFileCount}.parquet")
  


def build_dict(tmp_path, batch_size=1):
    dict_dir = os.path.join(tmp_path, "dict")
    all_files = sorted(glob.glob(os.path.join(dict_dir, "*.parquet")))
    dictionary = cudf.DataFrame({"phrases": cudf.Series(dtype="str")})

    # process in batches
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i : i + batch_size]

        df_batch = cudf.read_parquet(batch_files)

        dictionary = cudf.concat([dictionary, df_batch], ignore_index=True)

        dictionary = dictionary.drop_duplicates(subset='phrases').reset_index(drop=True)

        for f in batch_files:
            os.remove(f)

    dictionary = dictionary.drop_duplicates(subset='phrases').reset_index(drop=True)
    dictionary = dictionary.sort_values(by="phrases", ignore_index=True)
    dictionary = dictionary.reset_index(drop=False)

    return dictionary, None

def gpuPFP(input, w, p, tmp_path, threshold):
    
    tempFileCount = 0
    phrase_chunks = []
    offset_chunks = []
    cum_offset = 0

    programStart = time.time()
    for sequence in read_fasta(input, w):
        if cum_offset == 0:
            phrase_chunks.clear()
            offset_chunks.clear()
        seq_len = len(sequence)

        sequenceAsInt = cp.array(sequence, dtype=cp.uint8)
        del sequence

        result = cp.zeros(seq_len, dtype=cp.uint8)

        threads_per_block = 128
        blocks_per_grid = (seq_len + threads_per_block - 1) // threads_per_block
        trigger_string_finder[blocks_per_grid, threads_per_block](sequenceAsInt, 31, w, p, result)

        triggerStrings = cp.where(result == w)[0]
        num_phrases = triggerStrings.size - 1

        if num_phrases > 1:
            # Reset so that beginning and end isnt duplicated by kernel
            result[0] = 0
            result[-w] = 0
            # Get the shift for each character based on expansion
            temp = cp.cumsum(result)
            shifts = temp - result
            # Set up output array
            output_len = seq_len + int(temp[-1])
            del temp
            output = cp.empty(output_len, dtype=cp.uint8)

            blocks_per_grid = (seq_len + (threads_per_block - 1)) // threads_per_block
            sequence_expander[blocks_per_grid, threads_per_block](sequenceAsInt, result, shifts, w, output)

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
        if current_bytes/total_bytes > threshold:
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

        return dictionary, None, tempFileCount
    
    else:
        seq_arr = cp.concatenate(phrase_chunks)
        offsets_arr = cp.concatenate(offset_chunks)

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

        parse = cudf.DataFrame({'phrases': phrases_series})

        dictionary = parse.drop_duplicates(subset='phrases').reset_index(drop=True)
        dictionary = dictionary.sort_values(by='phrases', ignore_index=True)
        dictionary = dictionary.reset_index(drop=False)

        endTime = time.time()
        print(f"Total program runtime:{endTime - programStart:.4f}")

        return dictionary, parse, tempFileCount
    

def get_file_prefix(filepath):

    filename = os.path.basename(filepath)

    name, ext = os.path.splitext(filename)
    
    if ext in ['.gz', '.bz2', '.xz']:
        name, _ = os.path.splitext(name)
    
    return name

def main():
    parser = argparse.ArgumentParser(description="Placeholder Description", formatter_class=argparse.RawTextHelpFormatter)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--fasta', help="path to input fasta file",type=str)
    input_group.add_argument('-t', '--text', help="path to input text file",type=str)
    parser.add_argument('-w', '--wsize', help="sliding window size", default=5, type=int)
    parser.add_argument('-p', '--mod', help="hash modulus", default=15, type=int)
    parser.add_argument('-o', '--output', help="output files prefix", type=str)
    parser.add_argument('-d', '--tmp', default="tmp", help="directory for temporary files", type=str)
    parser.add_argument('-l', '--limit', default=0.10, help="GPU memory threshold for sequence processing", type=float)

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

    dictionary, parse, tempFileCount = gpuPFP(file, args.wsize, args.mod, args.tmp, args.limit)


    with open(output_prefix + '.parse', 'wb') as fh:
        if parse is not None:
            ranks64 = parse[["phrases"]].merge(dictionary, on=["phrases"])
            ranks32 = ranks64['index'].astype('uint32').to_cupy()
            ranks32.tofile(fh)
        else:
            for i in range(1, tempFileCount + 1):
                temp_df = cudf.read_parquet(f"{args.tmp}/parse/{i}.parquet")
                ranks64 = temp_df[["phrases"]].merge(dictionary, on=["phrases"])
                ranks32 = ranks64['index'].astype('uint32').to_cupy()
                ranks32.tofile(fh)
                os.remove(f"{args.tmp}/parse/{i}.parquet")
        fh.close()


    dictionary[['phrases']].to_csv(
        f"{output_prefix}.dict",
        index=False,        
        header=False,       
        sep=",",            
        lineterminator="\x03"
    )
    with open(f"{output_prefix}.dict", "r+b") as f:
        f.seek(-1, os.SEEK_END)  
        f.truncate()       
    del dictionary

    print(f"Peak VRAM Bytes:{rmm.statistics.get_statistics().peak_bytes}")



if __name__ == '__main__':
    main()


