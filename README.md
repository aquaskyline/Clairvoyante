# Clairvoyante - A deep neural network based variant caller
[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
Contact: Ruibang Luo  
Email: rluo5@jhu.edu  
***
## Introduction
With the increasing throughput and reliability of sequencing technologies in the recent years, it's getting common that medical doctors rely on sequencing to make better diagnostics to cancers and rare diseases. Among the interpretable results that sequencing can provide, genetics variants that have been reported in renowed databases such as Clinvar and HGMD, with evidence showing that they associate with certain symptomes and therapies, are considered as actionable genetic variant candidiates. However, eventhough the state-of-the-art variant callers achieve better precision and sensitivity, it is still not uncommon for the variant callers to produce false positive variants with somehow a pretty high probability, which makes them unable to be tell apart from the true negatives. The situation gets even worse when the callers are set to favor sensitivity over percision, which is often the case in clinical practices. The false positives variants, if not being sanitized, will lead to spurious clinical diagnosis. Instead of relying only on what a variant caller tells, clinical doctors usually verify the correctness of actionable genetic variants by eyes, with the help of `IGV` or `SAMtools tview`. Clairvoyante was designed to expedite the process and save the doctors from eye checking the variant, using a deep neural network based eye, trained with millions of samples on how a true variant "looks like" in different settings.  
***

## Gallery
### Network topology
![Network Topology](https://raw.githubusercontent.com/aquaskyline/Clairvoyante/rbDev/gallery/Network.png)
### Tensor examples
![Tensor examples](https://raw.githubusercontent.com/aquaskyline/Clairvoyante/rbDev/gallery/Tensors.png)
### Activations of the conv1 hidden layer to a non-variant tensor
![conv1](https://raw.githubusercontent.com/aquaskyline/Clairvoyante/rbDev/gallery/Conv1.png)
***
## Prerequisition
### Basics
Make sure you have Tensorflow ≥ 1.0.0 installed, the following commands install the lastest CPU version of Tensorflow:  
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
Clairvoyante was written in Python2 (tested on Python 2.7.10 in Linux and Python 2.7.13 in MacOS). It can be translated to Python3 using "2to3" just like other projects.

### Speed up with PyPy
Without a change to the code, using PyPy python intepreter on some tensorflow independent modules such as `dataPrepScripts/ExtractVariantCandidates.py` and `dataPrepScripts/CreateTensor.py` gives a 5-10 times speed up. Pypy python intepreter can be installed by apt-get, yum, Homebrew (I'm using this), Macports and etc. If you have no root access to your system, the offical website of Pypy provides a portable binary distribution for Linux. Following is a rundown extracted from Pypy's website on how to install the binaries.  
```
wget https://bitbucket.org/squeaky/portable-pypy/downloads/pypy-5.8-1-linux_x86_64-portable.tar.bz2
tar -jxf pypy-5.8-1-linux_x86_64-portable.tar.bz2
cd pypy-5.8-linux_x86_64-portable/bin
./pypy -m ensurepip
./pip install -U pip wheel
# Use pypy as an inplace substitution of python to run the scripts in dataPrepScripts/
```
To guarantee a good user experience, pypy must be installed to run `callVarBam.py` (call variants from BAM).
Tensorflow is optimized using Cython thus not compatible with `pypy`. For the list of scripts compatible to `pypy`, please refer to the **Folder Stucture and Program Descriptions** section.
*Pypy is by far the best Python JIT intepreter, you can donate to [the project](https://pypy.org) if you like it.*
***

## Quick Start with Variant Calling
You have a slow way and a quick way to get some demo variant calls. The slow way generates required files from VCF and BAM files. The fast way downloads the required files.
### Download testing dataset
#### I have plenty of time
```
wget 'https://www.dropbox.com/s/3ukdlywikir3cx5/testingData.tar.gz'
tar -xf testingData.tar
cd dataPrepScripts
sh PrepDataBeforeDemo.sh
```
#### I need some results now
```
wget 'https://www.dropbox.com/s/twxe6kyv6k3owz4/training.tar.gz'
tar -zxf training.tar.gz
```

### Call variants
#### Call variants from a BAM file and a trained model
```
cd training
python ../clairvoyante/callVarBam.py --chkpnt_fn ../trainedModels/fullv3-illumina-bwa-hg001-hg38/full_round9-000499 --bam_fn ../testingData/chr21/chr21.bam --ref_fn ../testingData/chr21/chr21.fa --call_fn tensor_can_chr21.vcf --ctgName chr21
less tensor_can_chr21.vcf
```
#### Call variants from the tensors of candidate variant and a trained model
```
cd training
python ../clairvoyante/callVar.py --chkpnt_fn ../trainedModels/fullv3-illumina-bwa-hg001-hg38/full_round9-000499 --tensor_fn tensor_can_chr21 --call_fn tensor_can_chr21.vcf
less tensor_can_chr21.vcf
```
***

## VCF Output Format
`clairvoyante/callVar.py` outputs variants in VCF format with version 4.1 specifications.  
Clairvoyante can predict the exact length of insertions and deletions shorter than or equal to 4bp. For insertions and deletions with a length between 5bp to 15bp, callVar guesses the length from input tensors. The indels with guessed length are denoted with a `LENGUESS` info tag. Although the guessed indel length might be incorrect, users can still benchmark Clairvoyante's sensitivity by matching the indel positions to other callsets. For indels longer than 15bp, `callVar.py` outputs them as SV without providing an alternative allele. To fit into a different usage scenario, Clairvoyante allows users to extend its model easily to support exact length prediction on longer indels by adding categories to the model output. However, this requires additional training data on the new categories. Users can also increase the length limit from where an indel is outputted as a SV by increasing the parameter flankingBaseNum from 16bp to a higher value. This extends the flanking bases to be considered with a candidate variant. 
***

## Build a Model
### Quick start with a model training demo
```
wget 'https://drive.google.com/file/d/0B4zabL1qoORCTUNTaGdxdTdTS0k/view?usp=sharing'
tar -xf testingData.tar
cd clairvoyante
python demoRun.py
```
### Jupyter notebook interactive training

Please visit: `jupyter_nb/demo.ipynb`
***

## Visualization
### Visualizing Input Tensors, Activiation in Hidden Layers and Comparing Predicted Results
`jupyter_nb/visualization.ipynb`

### Tensorboard
The `--olog_dir` option provided in the training scripts outputs a folder of log files readable by the Tensorboard. It can be used to visualize the dynamics of parameters during training at each epoch.
You can also use the PCA and t-SNE algorithms provided by TensorBoard in the `Embedding` page to analyze the predictions made by a model. `clairvoyante/getEmbedding.py` helps you to prepare a folder for the purpose.
***

## Folder Stucture and Program Descriptions
*You can also run the program to get the parameter details.*


`dataPrepScripts/` | Data Preparation Scripts. Outputs are gzipped unless using standard output. Scripts in this folder are compatible with `pypy`.
--- | ---
`ExtractVariantCandidates.py`| Extract the position of variant candidiates. Input: BAM; Reference FASTA. Important options: --threshold "Minimum alternative allelic fraction to report a candidate"; --minCoverage "Minimum coverage to report a candidate".
`GetTruth.py`| Extract the variants from a truth VCF. Input: VCF.
`CreateTensor.py`| Create tensors for candidates or truth variants. Input: A candidate list; BAM; Reference FASTA. Important option: --considerleftedge "Count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor. Enable if you are 1) using reads shorter than 100bp, 2) using  a tensor with flanking length longer than 16bp, and 3) you are using amplicon sequencing or other sequencing technologies, in which reads starting positions are random is not a basic assumption".
`PairWithNonVariants.py`| Pair truth variant tensors with non-variant tensors. Input: Truth variants tensors; Candidate variant tensors. Important options: --amp x "1-time truth variants + x-time non-vairants".
`ChooseItemInBed.py`| Helper script. Output the items overlapping with input the BED file.
`CountNumInBed.py`| Helper script. Count the number of items overlapping with the input BED file.
`param.py`| Global parameters for the scripts in the folder.
`PrepDataBeforeDemo.sh`| A **Demo** showing how to prepare data for model training.
`PrepDataBeforeDemo.pypy.sh`| The same demo but using pypy in place of python. `pypy` is highly recommended. It's easy to install, and makes the scripts run 5-10 times faster.


`clairvoyante/` | Model Training and Variant Caller Scripts. Scripts in this folder are NOT compatible with `pypy`. Please run with `python`.
--- | ---
`callVar.py `| Call variants using candidate variant tensors.
`callVarBam.py` | Call variants directly from a BAM file.
`demoRun.py` | A **Demo** showing how to training a model from scratch.
`evaluate.py` | Evaluate a model in terms of base change, zygosity, variant type and indel length.
`param.py` |  Hyperparameters for model training and other global parameters for the scripts in the folder.
`tensor2Bin.py` |  Create a compressed binary tensors file to facilitate and speed up future usage. Input: Mixed tensors by PairWithNonVariants.py; Truth variants by GetTruth.py and a BED file marks the high confidence regions in the reference genome.
`train.py` |  Training a model using adaptive learning rate decay. By default the learning rate will decay for three times. Input a binary tensors file created by Tensor2Bin.py is highly recommended.
`trainNonstop.py` |  Helper script. Train a model continuously using the same learning rate and l2 regularization lambda.
`trainWithoutValidationNonstop.py` | Helper script. Train a model continuously using the same learning rate and l2 regularization lambda. Take all the input tensors as training data and do not calculate loss in the validation data.
`calTrainDevDiff.py` | Helper script. Calculate the training loss and validation loss on a trained model.
`getEmbedding.py` | Prepare a folder readable by Tensorboard for visualzing predicted results.
`clairvoyante_v3.py` | Clairvoyante netowork topology v3.
`clairvoyante_v3_slim.py` | Clairvoyante netowork topology v3 slim. With 10 times less parameters than the full network, it trains about 1.5 times faster than the full network. It performs only about 1% less in precision and recall rates for Illumina data.
`utils_v2.py` | Helper functions to the netowork.


*GIAB provides a BED file that marks the high confidence regions in the reference. The models perform better by using only the truth variants in these regions for training. If you don't have a BED file, you can input a BED file that covers the whole genome.*
***

## About the Testing Data
The testing dataset 'testingData.tar' includes:  
1) the Illumina alignments of chr21 and chr22 on GRCh38 from [GIAB Github](ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/NIST_NA12878_HG001_HiSeq_300x/NHGRI_Illumina300X_novoalign_bams/HG001.GRCh38_full_plus_hs38d1_analysis_set_minus_alts.300x.bam), downsampled to 50x.  
2) the truth variants v3.3.2 from [GIAB](ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/NA12878_HG001/NISTv3.3.2/GRCh38).  
***

## Miscellaneous
### Run on MARCC
```
module load tensorflow/r1.0
```
Clairvoyante requires 'intervaltree' and 'blosc' library. However, loading the tensorflow module in MARCC changes the default path to the python executables and libraries. Please use 'sys.path.append' to add the path where the libraries are installed.  

