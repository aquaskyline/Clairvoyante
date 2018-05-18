set -e
mkdir ../training
python ../dataPrepScripts/ExtractVariantCandidates.py --bam_fn ../testingData/chr21/chr21.bam --ref_fn ../testingData/chr21/chr21.fa --can_fn ../training/can_chr21_sampled --ctgName chr21 --ctgStart 10269870 --ctgEnd 46672937 --gen4Training --genomeSize 3000000000 --candidates 7000000 &
python ../dataPrepScripts/ExtractVariantCandidates.py --bam_fn ../testingData/chr22/chr22.bam --ref_fn ../testingData/chr22/chr22.fa --can_fn ../training/can_chr22_sampled --ctgName chr22 --ctgStart 18924717 --ctgEnd 49973797 --gen4Training --genomeSize 3000000000 --candidates 7000000 &
python ../dataPrepScripts/GetTruth.py --vcf_fn ../testingData/chr21/chr21.vcf --var_fn ../training/var_chr21 --ctgName chr21 &
python ../dataPrepScripts/GetTruth.py --vcf_fn ../testingData/chr22/chr22.vcf --var_fn ../training/var_chr22 --ctgName chr22 &
wait

gzip -dc ../training/var_chr21 | awk '$2>10269870 && $2<=46672937' | gzip -c > ../training/var_chr21_sampled &
gzip -dc ../training/var_chr22 | awk '$2>18924717 && $2<=49973797' | gzip -c > ../training/var_chr22_sampled &
wait

python ../dataPrepScripts/CreateTensor.py --bam_fn ../testingData/chr21/chr21.bam --can_fn ../training/var_chr21_sampled --ref_fn ../testingData/chr21/chr21.fa --tensor_fn ../training/tensor_var_chr21_sampled --ctgName chr21 --ctgStart 10269870 --ctgEnd 46672937 &
python ../dataPrepScripts/CreateTensor.py --bam_fn ../testingData/chr22/chr22.bam --can_fn ../training/var_chr22_sampled --ref_fn ../testingData/chr22/chr22.fa --tensor_fn ../training/tensor_var_chr22_sampled --ctgName chr22 --ctgStart 18924717 --ctgEnd 49973797 &
wait

python ../dataPrepScripts/CreateTensor.py --bam_fn ../testingData/chr21/chr21.bam --can_fn ../training/can_chr21_sampled --ref_fn ../testingData/chr21/chr21.fa --tensor_fn ../training/tensor_can_chr21_sampled --ctgName chr21 --ctgStart 10269870 --ctgEnd 46672937 &
python ../dataPrepScripts/CreateTensor.py --bam_fn ../testingData/chr22/chr22.bam --can_fn ../training/can_chr22_sampled --ref_fn ../testingData/chr22/chr22.fa --tensor_fn ../training/tensor_can_chr22_sampled --ctgName chr22 --ctgStart 18924717 --ctgEnd 49973797 &
wait

cat ../testingData/chr21/chr21.bed ../testingData/chr22/chr22.bed > ../training/bed &
cat ../training/var_chr21_sampled ../training/var_chr22_sampled > ../training/var_mul_sampled &
cat ../training/tensor_can_chr21_sampled ../training/tensor_can_chr22_sampled > ../training/tensor_can_mul_sampled &
cat ../training/tensor_var_chr21_sampled ../training/tensor_var_chr22_sampled > ../training/tensor_var_mul_sampled &
wait

python ../dataPrepScripts/PairWithNonVariants.py --tensor_can_fn ../training/tensor_can_mul_sampled --tensor_var_fn ../training/tensor_var_mul_sampled --bed_fn ../training/bed --output_fn ../training/tensor_can_mix_sampled --amp 2
python ../clairvoyante/tensor2Bin.py --tensor_fn ../training/tensor_can_mix_sampled --var_fn ../training/var_mul_sampled --bed_fn ../training/bed --bin_fn ../training/tensor.bin
