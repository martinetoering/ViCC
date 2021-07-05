#!/bin/bash
#SBATCH --job-name=lin-rgb
#SBATCH -p gpu_titanrtx
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=75G

source /home/mtoering/miniconda3/etc/profile.d/conda.sh
conda activate vicc # e.g. use your own conda env
module load 2019
module load cuDNN
cd /home/mtoering/ViCC/src/eval

# Eval: Linear probe

DATASET_PATH='/home/mtoering/data/'

EXPERIMENT_PATH="/home/mtoering/ViCC/runs" # general folder of experiment
PREFIX="lin-rgb-2"
SUBFOLDER="eval"
mkdir -p ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}

TEST_PREFIX_PATH="/home/mtoering/ViCC/runs/pretrain/cross/c2-flowm/"
TEST_DIR_PATH="vicc2-k1920-t0.1_ucf101-2stream-2clip-128_s3d-e100_bs24_optimsgd-cosTrue-b_lr0.6-f_lr0.0006-wd1e-06-lr0.001-sched[120,160]_seq2-len32-ds1_nmb_p300-f_p0-vfa[0,1,2,3]-nmb_v[2,2]-eps0.05-sh_i3-e_q25-w_e0"
TEST_MODEL_PATH="/model/save_epoch99.pth.tar"
TEST_PATH="${TEST_PREFIX_PATH}${TEST_DIR_PATH}${TEST_MODEL_PATH}"
echo ${TEST_PATH}

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_classifier.py \
--dataset 'ucf101' \
--net 's3d' \
--gpu 0,1,2,3 \
--batch_size 32 \
--epochs 100 \
--train_what last \
--schedule 60 80 \
--optim sgd \
--lr 1e-1 \
--wd 1e-3 \
--ten_crop \
--prefix ${EXPERIMENT_PATH} \
--name_prefix ${PREFIX} \
--workers 16 \
--pretrain ${TEST_PATH} \
--backbone_ratio 10 \
--dataset_root ${DATASET_PATH} 


# Test

TEST_PREFIX_PATH="/home/mtoering/ViCC/runs/eval/"
TEST_DIR_PATH="/ucf101-128_sp1_lincls_s3d_bs128_lr0.1_dp0.9_wd0.001_seq1_len32_ds1_train-last_SGD"  # change path based on parameters above
TEST_MODEL_PATH="/model/epoch99.pth.tar"
TEST_PATH="${TEST_PREFIX_PATH}${PREFIX}${TEST_DIR_PATH}${TEST_MODEL_PATH}"
echo ${TEST_PATH}


CUDA_VISIBLE_DEVICES=0,1,2,3 python main_classifier.py \
--dataset 'ucf101' \
--net 's3d' \
--gpu 0,1,2,3 \
--batch_size 32 \
--ten_crop \
--prefix ${EXPERIMENT_PATH} \
--workers 16 \
--test ${TEST_PATH} \
--dataset_root ${DATASET_PATH} 
