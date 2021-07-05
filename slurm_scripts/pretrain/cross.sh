#!/bin/bash
#SBATCH --job-name=c1-flowm 
#SBATCH -p gpu_titanrtx
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=38:00:00
#SBATCH --mem=180G

source /home/mtoering/miniconda3/etc/profile.d/conda.sh
conda activate vicc # e.g. use your own conda env
module load 2019
module load cuDNN
cd /home/mtoering/ViCC/src

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

# change the following
DATASET_PATH='/home/mtoering/data/'
EXPERIMENT_PATH="/home/mtoering/ViCC/runs" # general experiment folder
PREFIX="cross/c1-flowm"
SUBFOLDER="pretrain"
mkdir -p ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}

# 1
# RGB checkpoint: to optimize. put your model path here, and update paths in the following cycles. e.g.:
PRETRAIN_1="/home/mtoering/ViCC/runs/pretrain/single/rgb/vicc-k1920-t0.1_ucf101-2clip-128_s3d-e500_bs48_optimsgd-cosTrue-b_lr0.6-f_lr0.0006-wd1e-06-lr0.001-sched[120,160]_seq2-len32-ds1_nmb_p300-f_p100-vfa[0,1]-nmb_v[2]-eps0.05-sh_i3-e_q150-w_e0/model/save_epoch299.pth.tar"
# Flow checkpoint: only sampler. put your model path here, and update paths in the following cycles. e.g.:
PRETRAIN_2="/home/mtoering/ViCC/runs/pretrain/single/flow/vicc-k1920-t0.1_ucf101-f-2clip-128_s3d-e500_bs48_optimsgd-cosTrue-b_lr0.6-f_lr0.0006-wd1e-06-lr0.001-sched[120,160]_seq2-len32-ds1_nmb_p300-f_p100-vfa[0,1]-nmb_v[2]-eps0.05-sh_i3-e_q200-w_e0/model/save_epoch299.pth.tar"

srun --output=${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/%j.out --error=${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/%j.err --label python -u main_cross.py \
--net 's3d' \
--model 'vicc2' \
--topk 5 \
--dataset 'ucf101-2stream-2clip' \
--seq_len 32 \
--num_seq 2 \
--ds 1 \
--batch_size 24 \
--lr 1e-3 \
--schedule 120 160 \
--wd 1e-6 \
--cos true \
--base_lr 0.6 \
--final_lr 0.0006 \
\
--resume '' \
--pretrain ${PRETRAIN_1} ${PRETRAIN_2} \
--test '' \
--epochs 100 \
--start_epoch 0 \
--save_epoch 24 \
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
--views_for_assign 0 1 2 3 \
--nmb_views 2 2 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--improve_numerical_stability false \
--nmb_prototypes 300 \
\
--epoch_queue_starts 25 \
--freeze_prototypes_nepochs 0 \
--warmup_epochs 0 \
--start_warmup 0 \
--dist-url $dist_url \
--sync_bn none \
--use_fp16 true \


#!/bin/bash
#SBATCH --job-name=c1-rgbm 
#SBATCH -p gpu_titanrtx
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=38:00:00
#SBATCH --mem=180G

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
PREFIX="cross/c1-rgbm"
SUBFOLDER="pretrain"
mkdir -p ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}

# 2
# Flow checkpoint: to optimize
PRETRAIN_1="/home/mtoering/ViCC/runs/pretrain/single/flow/vicc-k1920-t0.1_ucf101-f-2clip-128_s3d-e500_bs48_optimsgd-cosTrue-b_lr0.6-f_lr0.0006-wd1e-06-lr0.001-sched[120,160]_seq2-len32-ds1_nmb_p300-f_p100-vfa[0,1]-nmb_v[2]-eps0.05-sh_i3-e_q200-w_e0/model/save_epoch299.pth.tar"
# RGB checkpoint (rgb cycle 1 checkpoint): only sampler
PRETRAIN_2="/home/mtoering/ViCC/runs/pretrain/cross/c1-flowm/vicc2-k1920-t0.1_ucf101-2stream-2clip-128_s3d-e100_bs24_optimsgd-cosTrue-b_lr0.6-f_lr0.0006-wd1e-06-lr0.001-sched[120,160]_seq2-len32-ds1_nmb_p300-f_p0-vfa[0,1,2,3]-nmb_v[2,2]-eps0.05-sh_i3-e_q25-w_e0/model/save_epoch99.pth.tar"

srun --output=${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/%j.out --error=${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/%j.err --label python -u main_cross.py \
--net 's3d' \
--model 'vicc2' \
--topk 5 \
--dataset 'ucf101-2stream-2clip' \
--seq_len 32 \
--num_seq 2 \
--ds 1 \
--batch_size 24 \
--lr 1e-3 \
--schedule 120 160 \
--wd 1e-6 \
--cos true \
--base_lr 0.6 \
--final_lr 0.0006 \
\
--resume '' \
--pretrain ${PRETRAIN_1} ${PRETRAIN_2} \
--test '' \
--epochs 100 \
--start_epoch 0 \
--save_epoch 24 \
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
--reverse \
--workers 12 \
--moco-dim 128 \
--moco-k 1920 \
--moco-m 0.999 \
--moco-t 0.1 \
--views_for_assign 0 1 2 3 \
--nmb_views 2 2 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--improve_numerical_stability false \
--nmb_prototypes 300 \
--epoch_queue_starts 25 \
--freeze_prototypes_nepochs 0 \
--warmup_epochs 0 \
--start_warmup 0 \
--dist-url $dist_url \
--sync_bn none \
--use_fp16 true \




#!/bin/bash
#SBATCH --job-name=c2-flowm 
#SBATCH -p gpu_titanrtx
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=38:00:00
#SBATCH --mem=180G

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
PREFIX="cross/c2-flowm"
SUBFOLDER="pretrain"
mkdir -p ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}

# 3
# RGB checkpoint (rgb cycle 1 checkpoint): to optimize
PRETRAIN_1="/home/mtoering/ViCC/runs/pretrain/cross/c1-flowm/vicc2-k1920-t0.1_ucf101-2stream-2clip-128_s3d-e100_bs24_optimsgd-cosTrue-b_lr0.6-f_lr0.0006-wd1e-06-lr0.001-sched[120,160]_seq2-len32-ds1_nmb_p300-f_p0-vfa[0,1,2,3]-nmb_v[2,2]-eps0.05-sh_i3-e_q25-w_e0/model/save_epoch99.pth.tar"
# Flow checkpoint (flow cycle 1 checkpoint): only sampler
PRETRAIN_2="/home/mtoering/ViCC/runs/pretrain/cross/c1-rgbm/vicc2-k1920-t0.1_ucf101-2stream-2clip-128_s3d-e100_bs24_optimsgd-cosTrue-b_lr0.6-f_lr0.0006-wd1e-06-lr0.001-sched[120,160]_seq2-len32-ds1_nmb_p300-f_p0-vfa[0,1,2,3]-nmb_v[2,2]-eps0.05-sh_i3-e_q25-w_e0/model/save_epoch99.pth.tar"

srun  --output=${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/%j.out --error=${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/%j.err --label python -u main_cross.py \
--net 's3d' \
--model 'vicc2' \
--topk 5 \
--dataset 'ucf101-2stream-2clip' \
--seq_len 32 \
--num_seq 2 \
--ds 1 \
--batch_size 24 \
--lr 1e-3 \
--schedule 120 160 \
--wd 1e-6 \
--cos true \
--base_lr 0.6 \
--final_lr 0.0006 \
\
--resume '' \
--pretrain ${PRETRAIN_1} ${PRETRAIN_2} \
--test '' \
--epochs 100 \
--start_epoch 0 \
--save_epoch 24 \
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
--views_for_assign 0 1 2 3 \
--nmb_views 2 2 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--improve_numerical_stability false \
--nmb_prototypes 300 \
\
--epoch_queue_starts 25 \
--freeze_prototypes_nepochs 0 \
--warmup_epochs 0 \
--start_warmup 0 \
--dist-url $dist_url \
--sync_bn none \
--use_fp16 true \


# comment out below if you dont want evaluation on retrieval here
wait
cd /home/mtoering/ViCC/src/eval
FILE_PATH="temp_path.txt"
source ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/${FILE_PATH}
echo ${MODEL_PATH}
rm ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/${FILE_PATH}

TEST_MODEL_PATH="save_epoch99.pth.tar"
TEST_PATH=${MODEL_PATH}/${TEST_MODEL_PATH}
echo ${TEST_PATH}

# Test retrieval ucf
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
--test ${TEST_PATH} 



#!/bin/bash
#SBATCH --job-name=c2-rgbm 
#SBATCH -p gpu_titanrtx
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=38:00:00
#SBATCH --mem=180G

source /home/mtoering/miniconda3/etc/profile.d/conda.sh
conda activate vicc # e.g. use your own conda env
module load 2019
module load cuDNN
cd /home/mtoering/ViCC/src
nvidia-smi

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

DATASET_PATH='/home/mtoering/data/'
EXPERIMENT_PATH="/home/mtoering/ViCC/runs" # general experiment folder
PREFIX="cross/c2-rgbm"
SUBFOLDER="pretrain"
mkdir -p ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}

# 4
# Flow checkpoint (flow cycle 1 checkpoint): to optimize
PRETRAIN_1="/home/mtoering/ViCC/runs/pretrain/cross/c1-rgbm/vicc2-k1920-t0.1_ucf101-2stream-2clip-128_s3d-e100_bs24_optimsgd-cosTrue-b_lr0.6-f_lr0.0006-wd1e-06-lr0.001-sched[120,160]_seq2-len32-ds1_nmb_p300-f_p0-vfa[0,1,2,3]-nmb_v[2,2]-eps0.05-sh_i3-e_q25-w_e0/model/save_epoch99.pth.tar"
# RGB checkpoint (rgb cycle 2 checkpoint): only sampler
PRETRAIN_2="/home/mtoering/ViCC/runs/pretrain/cross/c2-flowm/vicc2-k1920-t0.1_ucf101-2stream-2clip-128_s3d-e100_bs24_optimsgd-cosTrue-b_lr0.6-f_lr0.0006-wd1e-06-lr0.001-sched[120,160]_seq2-len32-ds1_nmb_p300-f_p0-vfa[0,1,2,3]-nmb_v[2,2]-eps0.05-sh_i3-e_q25-w_e0/model/save_epoch99.pth.tar"

srun --output=${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/%j.out --error=${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/%j.err --label python -u main_cross.py \
--net 's3d' \
--model 'vicc2' \
--topk 5 \
--dataset 'ucf101-2stream-2clip' \
--seq_len 32 \
--num_seq 2 \
--ds 1 \
--batch_size 24 \
--lr 1e-3 \
--schedule 120 160 \
--wd 1e-6 \
--cos true \
--base_lr 0.6 \
--final_lr 0.0006 \
\
--resume '' \
--pretrain ${PRETRAIN_1} ${PRETRAIN_2} \
--test '' \
--epochs 100 \
--start_epoch 0 \
--save_epoch 24 \
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
--reverse \
--workers 12 \
--moco-dim 128 \
--moco-k 1920 \
--moco-m 0.999 \
--moco-t 0.1 \
--views_for_assign 0 1 2 3 \
--nmb_views 2 2 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--improve_numerical_stability false \
--nmb_prototypes 300 \
\
--epoch_queue_starts 25 \
--freeze_prototypes_nepochs 0 \
--warmup_epochs 0 \
--start_warmup 0 \
--dist-url $dist_url \
--sync_bn none \
--use_fp16 true \


# comment out below if you dont want evaluation on retrieval here
wait
cd /home/mtoering/ViCC/src/eval
FILE_PATH="temp_path.txt"
source ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/${FILE_PATH}
echo ${MODEL_PATH}
rm ${EXPERIMENT_PATH}/$SUBFOLDER/${PREFIX}/${FILE_PATH}

TEST_MODEL_PATH="save_epoch99.pth.tar"
TEST_PATH=${MODEL_PATH}/${TEST_MODEL_PATH}
echo ${TEST_PATH}

# Test retrieval ucf
FEATURE_PATH="feature_99"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_classifier.py \
--net s3d \
--dataset ucf101-f \
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
