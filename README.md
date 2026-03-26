# `GPU-PFP`
`GPU-PFP` is a GPU implementation of Prefix-Free Parsing created in Python using GPU-accelerated libraries, like RAPIDSAI's CUDF.

## Installation
1. Clone the repository
```
git clone git@github.com:EddieFerro/gpuPFP.git
cd gpuPFP
```
2. Set up the virtual environment
```
conda env create -f env.yml
conda activate my-env
```
Note: It may be necessary to install device specific libraries to be able to utilize this tool. 

## Usage
`GPU-PFP` takes as input a fasta or text file, uncompressed or gzip compressed, and outputs the Prefix-free parse of the input, namely a dictionary file (.dict) and a parse file (.parse). It also takes in two user defined parameters, `w` and `p`, that determine the parsing of the input file. More information regarding these parameters can be found in the original PFP paper given below.
```
@article{boucher2019prefix,
  title={Prefix-free parsing for building big BWTs},
  author={Boucher, Christina and Gagie, Travis and Kuhnle, Alan and Langmead, Ben and Manzini, Giovanni and Mun, Taher},
  journal={Algorithms for Molecular Biology},
  volume={14},
  number={1},
  pages={1--15},
  year={2019},
  publisher={BioMed Central}
}
```
## Parameters
```
usage: gpuPFP.py [-h] (-f FASTA | -t TEXT) [-w WSIZE] [-p MOD] [-o OUTPUT] [-d TMP] [-l LIMIT]

GPU-Accelerated PFP

options:
  -h, --help                     show this help message and exit
  -f FASTA, --fasta FASTA        Path to input fasta file
  -t TEXT, --text TEXT           Path to input text file
  -w WSIZE, --wsize WSIZE        Sliding window size
  -p MOD, --mod MOD              Modulo used during parsing
  -o OUTPUT, --output OUTPUT     Output files prefix
  -d TMP_DIR, --tmp-dir TMP_DIR  Directory for temporary files
  --threshold THRESHOLD          Fraction of free GPU memory to use per input batch (0 < f ≤ 1).
```