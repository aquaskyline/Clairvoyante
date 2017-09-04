## Clairvoyante - A deep neural network based variant caller

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
Contact: Ruibang Luo  
Email: rluo5@jhu.edu  

## Introduction
With the increasing throughput and reliability of sequencing technologies in the recent years, it's getting common that medical doctors rely on sequencing to make better diagnostics to cancers and rare diseases. Among the interpretable results that sequencing can provide, genetics variants that have been reported in renowed databases such as Clinvar and HGMD, with evidence showing that they associate with certain symptomes and therapies, are considered as actionable genetic variant candidiates. However, eventhough the state-of-the-art variant callers achieve better precision and sensitivity, it is still not uncommon for the variant callers to produce false positive variants with somehow a pretty high probability, which makes them unable to be tell apart from the true negatives. The situation gets even worse when the callers are set to favor sensitivity over percision, which is often the case in clinical practices. The false positives variants, if not being sanitized, will lead to spurious clinical diagnosis. Instead of relying only on what a variant caller tells, clinical doctors usually verify the correctness of actionable genetic variants by eyes, with the help of "IGV" or "SAMtools tview". Clairvoyante was designed to expedite the process and save the doctors from eye checking the variant, using a deep neural network based eye, trained with millions of samples on how a true variant "looks like" in different settings.  

## Prerequisition
### Basics
Make sure you have Tensorflow >= 1.0.0 installed, the following commands install the lastest CPU version of Tensorflow:  
```
pip install tensorflow
pip install blosc
pip install intervaltree
pip install numpy
```
To check the version of Tensorflow you have installed:  
```
python -c 'import tensorflow as tf; print(tf.__version__)'
```
To do variant calling using trained models, CPU will suffice. Tensorflow will use all available CPU cores by default. To train a new model, a high-end GPU along with the GPU version of Tensorflow is needed. To install the GPU version of tensorflow:  
```
pip install tensorflow-gpu
```
Clairvoyante was written in Python2. It can be translated to Python3 using "2to3" just like other projects. Just to mention that you need to use pip3 to install the dependencies listed above.  

### Speed up with PyPy
Without a change to the code, using PyPy python intepreter on some tensorflow independent modules such as "dataPrepScripts/ExtractVariantCandidates.py" and "dataPrepScripts/CreateTensor.py" gives a 5-10 times speed up. Pypy python intepreter can be installed by apt-get, yum, Homebrew (I'm using this), Macports and etc. If you have no root access to your system, the offical website of Pypy provides a portable binary distribution for Linux. Following is a rundown extracted from Pypy's website on how to install the binaries.  
```
wget https://bitbucket.org/squeaky/portable-pypy/downloads/pypy-5.8-1-linux_x86_64-portable.tar.bz2
tar -jxf pypy-5.8-1-linux_x86_64-portable.tar.bz2
cd pypy-5.8-linux_x86_64-portable/bin
./pypy -m ensurepip
./pip install -U pip wheel
# Use pypy as an inplace substitution of python to run the scripts in dataPrepScripts/
```

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
python ../clairvoyante/callVar.py --chkpnt_fn ../trainedModels/fullv3-illumina-bwa-hg38/full_round9-000499 --tensor_fn tensor_can_chr21 --call_fn tensor_can_chr21.call
less tensor_can_chr21.call
```

## Variant call output format
chromosome position genotype zygosity type indeLength  
zygosity: HET or HOM  
type: SNP, INS or DEL  
indelLength: 0, 1, 2, 3, 4, >4  

## Quick start with model training
```
wget 'https://drive.google.com/file/d/0B4zabL1qoORCTUNTaGdxdTdTS0k/view?usp=sharing'
tar -xf testingData.tar
cd clairvoyante
python demoRun.py
```

## About the testing data
The testing dataset 'testingData.tar' includes:  
1) the Illumina alignments of chr21 and chr22 on GRCh38 from [GIAB Github](ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/NIST_NA12878_HG001_HiSeq_300x/NHGRI_Illumina300X_novoalign_bams/HG001.GRCh38_full_plus_hs38d1_analysis_set_minus_alts.300x.bam), downsampled to 50x.  
2) the truth variants v3.3.2 from [GIAB](ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/NA12878_HG001/NISTv3.3.2/GRCh38).  

## Notes
### Run on MARCC
```
module load tensorflow/r1.0
```
Clairvoyante requires 'intervaltree' and 'blosc' library. However, loading the tensorflow module in MARCC changes the default path to the python executables and libraries. Please use 'sys.path.append' to add the path where the libraries are installed.  

