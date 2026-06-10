#!/bin/bash
#$ -cwd
#$ -V
#$ -l node_q=1
#$ -l h_rt=3:00:00
#$ -N p3d_profile
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/Pixal3DMD/logs/profile_teacher.$JOB_ID.log

# Fine-grained per-block timing of the Pixal3D teacher over 66 Toys4k objects (2 warmup).
# Env block mirrors 3D_dmd/scripts/launch_pixal3d_arm.sh's run_driver (colleague conda env,
# pixal3d_extra PYTHONPATH shim, CUDA 12.8, HF caches). cwd MUST be the Pixal3D clone root.
# Launch:  qsub -g tga-koike-shanda2 /gs/fs/tga-koike-shanda2/sk/Pixal3DMD/scripts/profile_teacher.sh
set -u
cd /gs/fs/tga-koike-shanda2/sk/Pixal3D

PYTHONPATH=.:/gs/fs/tga-koike-shanda2/sk/envs/pixal3d_extra \
PYTHONNOUSERSITE=1 \
HF_HOME=/gs/fs/tga-koike-shanda2/sk/hf_cache \
TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch \
CUDA_HOME=/gs/fs/tga-koike-shanda2/niumuyao/cuda-12.8 \
LD_LIBRARY_PATH="/gs/fs/tga-koike-shanda2/niumuyao/cuda-12.8/lib64" \
CUDA_VISIBLE_DEVICES=0 \
/gs/fs/tga-koike-shanda2/niumuyao/miniconda3/envs/pixal3d/bin/python -u \
/gs/fs/tga-koike-shanda2/sk/Pixal3DMD/scripts/profile_teacher.py \
  --num-objects 66 --warmup 2 \
  --out-dir /gs/fs/tga-koike-shanda2/sk/scratch/pixal3d_profile
rc=$?
echo "profile rc=$rc"
exit $rc
