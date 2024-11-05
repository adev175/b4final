#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python main.py \
python main.py \
--datasetName VimeoSepTuplet \
--datasetPath 'E:\KIEN\ANHProject\data\Vimeo_septuplet/' \
--batch_size 8 \
--max_epoch 200 \
--val_batch_size 16 \
--checkpoint_dir 'E:\KIEN\ANHProject\checkpoint\RSTSCANet_epoch140.pth' \
--resume True