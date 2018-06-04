# For convenience, this script reused chr21 and chr22 in the demo dataset.
# In spite of this, please treat chr21 and chr22 as two different genomes,
# and the commands used below were used for combining training samples from
# multiple genomes.

set -e
#######################################################################
# This section is the same as preparing the samples on a single genome for
# model training
if [ "$1" != "skipPrep" ] ; then
mkdir ../training
pypy ../dataPrepScripts/ExtractVariantCandidates.py --bam_fn ../testingData/chr21/chr21.bam --ref_fn ../testingData/chr21/chr21.fa --can_fn ../training/can_chr21_sampled --ctgName chr21 --ctgStart 10269870 --ctgEnd 46672937 --gen4Training --genomeSize 3000000000 --candidates 7000000 &
pypy ../dataPrepScripts/ExtractVariantCandidates.py --bam_fn ../testingData/chr22/chr22.bam --ref_fn ../testingData/chr22/chr22.fa --can_fn ../training/can_chr22_sampled --ctgName chr22 --ctgStart 18924717 --ctgEnd 49973797 --gen4Training --genomeSize 3000000000 --candidates 7000000 &
# Tips: Instead of using ExtractVariantCandidates.py plus the --gen4Training option, you can also try
# pypy ../dataPrepScripts/RandomSampling.py --ref_fn ../testingData/chr21/chr21.fa --can_fn ../training/can_chr21_sampled --ctgName chr21 --ctgStart 10269870 --ctgEnd 46672937 --genomeSize 3000000000 --candidates 7000000
# pypy ../dataPrepScripts/RandomSampling.py --ref_fn ../testingData/chr22/chr22.fa --can_fn ../training/can_chr22_sampled --ctgName chr22 --ctgStart 18924717 --ctgEnd 49973797 --genomeSize 3000000000 --candidates 7000000
# Tips: To speed up, you can pipe the results of ExtractVariantCandidates.py or RandomSampling.py directly into CreateTensor.py, by not specifying the --can_fn option in both.
pypy ../dataPrepScripts/GetTruth.py --vcf_fn ../testingData/chr21/chr21.vcf --var_fn ../training/var_chr21 --ctgName chr21 &
pypy ../dataPrepScripts/GetTruth.py --vcf_fn ../testingData/chr22/chr22.vcf --var_fn ../training/var_chr22 --ctgName chr22 &
wait

gzip -dc ../training/var_chr21 | awk '$2>10269870 && $2<=46672937' | gzip -c > ../training/var_chr21_sampled &
gzip -dc ../training/var_chr22 | awk '$2>18924717 && $2<=49973797' | gzip -c > ../training/var_chr22_sampled &
wait

pypy ../dataPrepScripts/CreateTensor.py --bam_fn ../testingData/chr21/chr21.bam --can_fn ../training/var_chr21_sampled --ref_fn ../testingData/chr21/chr21.fa --tensor_fn ../training/tensor_var_chr21_sampled --ctgName chr21 --ctgStart 10269870 --ctgEnd 46672937 &
pypy ../dataPrepScripts/CreateTensor.py --bam_fn ../testingData/chr22/chr22.bam --can_fn ../training/var_chr22_sampled --ref_fn ../testingData/chr22/chr22.fa --tensor_fn ../training/tensor_var_chr22_sampled --ctgName chr22 --ctgStart 18924717 --ctgEnd 49973797 &
wait

pypy ../dataPrepScripts/CreateTensor.py --bam_fn ../testingData/chr21/chr21.bam --can_fn ../training/can_chr21_sampled --ref_fn ../testingData/chr21/chr21.fa --tensor_fn ../training/tensor_can_chr21_sampled --ctgName chr21 --ctgStart 10269870 --ctgEnd 46672937 &
pypy ../dataPrepScripts/CreateTensor.py --bam_fn ../testingData/chr22/chr22.bam --can_fn ../training/can_chr22_sampled --ref_fn ../testingData/chr22/chr22.fa --tensor_fn ../training/tensor_can_chr22_sampled --ctgName chr22 --ctgStart 18924717 --ctgEnd 49973797 &
wait ; fi
#######################################################################

#######################################################################
# This section shows how to combine the samples of multiple genomes
# The key is to add a prefix to the chromosome id of different genomes
# to allow Clairvoyante to treat samples from different genomes as
# different samples. Different prefixes are needed for different genomes.
# In this example, we used "u" for chr21 (genome 1), and "v" for chr22
# (genome 2).

awk '{print "u"$0}' ../testingData/chr21/chr21.bed > ../training/chr21.bed_prefixed
awk '{print "v"$0}' ../testingData/chr22/chr22.bed > ../training/chr22.bed_prefixed
cat ../training/chr21.bed_prefixed ../training/chr22.bed_prefixed > ../training/bed_prefixed

gzip -dc ../training/var_chr21_sampled | awk '{print "u"$0}' | gzip -c > ../training/var_chr21_sampled_prefixed
gzip -dc ../training/var_chr22_sampled | awk '{print "v"$0}' | gzip -c > ../training/var_chr22_sampled_prefixed
cat ../training/var_chr21_sampled_prefixed ../training/var_chr22_sampled_prefixed > ../training/var_mul_sampled_prefixed

gzip -dc ../training/tensor_can_chr21_sampled | awk '{print "u"$0}' | gzip -c > ../training/tensor_can_chr21_sampled_prefixed
gzip -dc ../training/tensor_can_chr22_sampled | awk '{print "v"$0}' | gzip -c > ../training/tensor_can_chr22_sampled_prefixed
cat ../training/tensor_can_chr21_sampled_prefixed ../training/tensor_can_chr22_sampled_prefixed > ../training/tensor_can_mul_sampled_prefixed

gzip -dc ../training/tensor_var_chr21_sampled | awk '{print "u"$0}' | gzip -c > ../training/tensor_var_chr21_sampled_prefixed
gzip -dc ../training/tensor_var_chr22_sampled | awk '{print "v"$0}' | gzip -c > ../training/tensor_var_chr22_sampled_prefixed
cat ../training/tensor_var_chr21_sampled_prefixed ../training/tensor_var_chr22_sampled_prefixed > ../training/tensor_var_mul_sampled_prefixed

pypy ../dataPrepScripts/PairWithNonVariants.py --tensor_can_fn ../training/tensor_can_mul_sampled_prefixed --tensor_var_fn ../training/tensor_var_mul_sampled_prefixed --bed_fn ../training/bed_prefixed --output_fn ../training/tensor_can_mix_prefixed --amp 2
python ../clairvoyante/tensor2Bin.py --tensor_fn ../training/tensor_can_mix_prefixed --var_fn ../training/var_mul_sampled_prefixed --bed_fn ../training/bed_prefixed --bin_fn ../training/tensor_combined.bin
#######################################################################
