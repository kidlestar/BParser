#!/bin/bash
#SBATCH --job-name=python
#SBATCH --gres=gpu:1
#SBATCH --output=./modelLinearCRF10/slurm.out
#SBATCH --error=./modelLinearCRF10/slurm.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

srun python debug_linear.py -m Linear -g 0 -b 190 -r 0 -o 1 -e 5000 -l 2 -t /users/xudong.zhang/comp/old/de_gsd-ud-train.conllu -d /users/xudong.zhang/comp/old/de_gsd-ud-dev.conllu -u /users/xudong.zhang/comp/old/de_gsd-ud-test.conllu -s ./modelLinearCRF10/

#srun python debug_linear.py -m Linear -g 0 -b 190 -r 0 -o 1 -e 5000 -l 2 -t /users/xudong.zhang/data/train.conll -d /users/xudong.zhang/data/dev.conll -u /users/xudong.zhang/data/test.conll
