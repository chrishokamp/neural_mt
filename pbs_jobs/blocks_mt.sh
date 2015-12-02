#!/bin/sh
#PBS -N atn-en-es
#PBS -m ea
#PBS -M chris.hokamp@gmail.com
#PBS -l nodes=1:ppn=20
#PBS -l walltime=100:00:00 
#PBS -q GpuQ
#PBS -A dcu01 

### setting up theano with CUDA
module load dev

module load gcc/5.1.0 # fixes GLIB error, but throws nvcc error, which is why we need to also add the next lib
module load gcc/4.9.2 
module load intel/2015-u3 #mkl
module load cuda/7.0

# this hack is critical, otherwise things break
export LD_LIBRARY_PATH=/ichec/packages/gcc/5.1.0/lib64/:$LD_LIBRARY_PATH
export THEANO_FLAGS='on_unused_input=warn'

TEST_DIR=/ichec/home/users/chokamp/projects/neural_mt
cd ${TEST_DIR}

echo 'Testing the blocks machine translation example'

python -m machine_translation 

# tee stdout and monitor in detachable tmux session

# anaconda packages
# /ichec/packages/python/anaconda/1.9.2/lib/python2.7/site-packages/
#export PYTHONPATH=$PYTHONPATH:/ichec/packages/python/anaconda/1.9.2/lib/python2.7/site-packages/

### other things for your .bashrc
# A pbs job will also set this automatically?
#export WORKDIR=/ichec/work/dcu01/chokamp/

