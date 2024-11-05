#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python main.py \
python test.py \
--datasetName VimeoSepTuplet \
--datasetPath 'E:\KIEN\vimeo_septuplet' \
--checkpoint_dir 'E:\ANH\ANHProject\checkpoint\RSTSCANet_best.pth' \
--test_batch_size 16 \
--save_folder "test_result_02" \
--save_images True