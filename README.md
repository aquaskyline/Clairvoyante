## Clairvoyante

## Prerequisition
Make sure you have Tensorflow >= 1.0.0 installed, the following commands install the lastest CPU version of Tensorflow:
```
pip install tensorflow
pip install blosc
pip install intervaltree
pip install numpy
python -c 'import tensorflow as tf; print(tf.__version__)'
```
To do variant calling base on trained models, CPU is enough. To train a new model, a high-end GPU along with the GPU version of Tensorflow is needed.


## Quick start with variant calling 
You have a slow way and a quick way to get some demo variant calls. The slow way generates required files from VCF and BAM files. The fast way downloads the required files.
### I have plenty of time
```
wget 'https://www.dropbox.com/s/3ukdlywikir3cx5/testingData.tar.gz'
tar -xf testingData.tar
cd dataPrepScripts
sh PrepDataBeforeDemo.sh
```
### I need some results now
```
wget 'https://www.dropbox.com/s/twxe6kyv6k3owz4/training.tar.gz'
tar -zxf training.tar.gz
```
### Call variants
```
cd training
python ../clairvoyante/callVar.py --chkpnt_fn ../trainedModels/illumina2/full_round4-less17-031351 --tensor_fn tensor_can_chr21 --call_fn tensor_can_chr21.call
less tensor_can_chr21.call
```

## Variant call format
chromosome position genotype zygosity type indeLength
<br>
zygosity: HET or HOM
<br>
type: SNP, INS or DEL
<br>
indelLength: 0, 1, 2, 3, 4, >4
<br>

## Quick start with model training
```
wget 'https://drive.google.com/file/d/0B4zabL1qoORCTUNTaGdxdTdTS0k/view?usp=sharing'
tar -xf testingData.tar
cd clairvoyante
python demoRun.py
```

## About the testing data
The testing dataset 'testingData.tar' includes 1) the Illumina alignments of chr21 and chr22 from [GIAB Github](ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/NIST_NA12878_HG001_HiSeq_300x/NHGRI_Illumina300X_novoalign_bams/HG001.GRCh38_full_plus_hs38d1_analysis_set_minus_alts.300x.bam), downsampled to 50x. 2) the truth variants v3.3.2 from [GIAB](ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/NA12878_HG001/NISTv3.3.2/GRCh38).
<br>
GRCh38 was used. 


## Run on MARCC
```
module load tensorflow/r1.0
```
Clairvoyante requires 'intervaltree' library. However, loading the tensorflow module in MARCC changes the default path of python executables and libraries. Please use 'sys.path.append' to add the path where 'intervaltree' is installed.

