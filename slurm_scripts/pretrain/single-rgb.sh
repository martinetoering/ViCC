#!/bin/bash
#SBATCH --job-name=s3d-rgb
#SBATCH -p gpu_titanrtx
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=38:00:00
#SBATCH --mem=150G

source /home/mtoering/miniconda3/etc/profile.d/conda.sh
conda activate vicc # e.g. use your own conda env
module load 2019
module load cuDNN
cd /home/mtoering/ViCC/src

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

DATASET_PATH='/home/mtoering/data/'
EXPERIMENT_PATH="/home/mtoering/ViCC/runs" # general experiment folder
PREFIX="single/rgb"
SUBFOLDER="pretrain"
mkdir -p ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}

srun --output=${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/%j.out --error=${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/%j.err --label python -u main_single.py \
--net 's3d' \
--model 'vicc' \
--dataset 'ucf101-2clip' \
--seq_len 32 \
--num_seq 2 \
--ds 1 \
--batch_size 48 \
--lr 1e-3 \
--schedule 120 160 \
--wd 1e-6 \
--cos true \
--base_lr 0.6 \
--final_lr 0.0006 \
\
--resume '' \
--pretrain '' \
--test '' \
--epochs 500 \
--start_epoch 0 \
--save_epoch 149 \
--save_epoch_freq 100 \
--print_freq 5 \
--tb_background false \
--optim sgd \
--save_freq 1 \
--img_dim 128 \
--dataset_root ${DATASET_PATH} \
--prefix ${EXPERIMENT_PATH} \
--name_prefix ${PREFIX} \
\
--workers 12 \
--moco-dim 128 \
--moco-k 1920 \
--moco-m 0.999 \
--moco-t 0.1 \
--views_for_assign 0 1 \
--nmb_views 2 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--improve_numerical_stability false \
--nmb_prototypes 300 \
\
--epoch_queue_starts 150 \
--freeze_prototypes_nepochs 100 \
--warmup_epochs 0 \
--start_warmup 0 \
--dist-url $dist_url \
--sync_bn none \
--use_fp16 false \


# comment out below if you dont want evaluation on retrieval here
wait
cd /home/mtoering/ViCC/src/eval
FILE_PATH="temp_path.txt"
source ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/${FILE_PATH}
echo ${MODEL_PATH}
rm ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/${FILE_PATH}

TEST_MODEL_PATH="save_epoch299.pth.tar"
TEST_PATH=${MODEL_PATH}/${TEST_MODEL_PATH}
echo ${TEST_PATH}

# Test retrieval ucf
FEATURE_PATH="feature_49"
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
--test ${TEST_PATH} 

# Test retrieval hmdb
FEATURE_PATH="hmdb_feature_299"

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_classifier.py \
--net s3d \
--dataset hmdb51 \
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