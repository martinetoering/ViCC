#!/bin/bash
#SBATCH --job-name=retr-rgb
#SBATCH -p gpu_titanrtx_short
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=75G

source /home/mtoering/miniconda3/etc/profile.d/conda.sh
conda activate vicc # e.g. use your own conda env
module load 2019
module load cuDNN
cd /home/mtoering/ViCC/src/eval

# Eval: Retrieval

DATASET_PATH='/home/mtoering/data/'

TEST_PREFIX_PATH="/home/mtoering/ViCC/runs/pretrain/cross/c2-flowm/"
TEST_DIR_PATH="vicc2-k1920-t0.1_ucf101-2stream-2clip-128_s3d-e100_bs24_optimsgd-cosTrue-b_lr0.6-f_lr0.0006-wd1e-06-lr0.001-sched[120,160]_seq2-len32-ds1_nmb_p300-f_p0-vfa[0,1,2,3]-nmb_v[2,2]-eps0.05-sh_i3-e_q25-w_e0"
TEST_MODEL_PATH="/model/save_epoch99.pth.tar"
TEST_PATH="${TEST_PREFIX_PATH}${TEST_DIR_PATH}${TEST_MODEL_PATH}"
echo ${TEST_PATH}
FEATURE_PATH="feature_99"

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_classifier.py \
--net s3d \
--dataset ucf101 \
--which_split 1 \
--seq_len 32 \
--ds 1 \
--gpu 0,1,2,3 \
--batch_size 48 \
--workers 8 \
--ten_crop \
--retrieval \
--dataset_root ${DATASET_PATH} \
--prefix test \
--dirname ${FEATURE_PATH} \
--test ${TEST_PATH} \

