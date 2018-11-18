set -ex
python train.py --dataroot ./datasets/simple_grid --name planet_mars_gen_3 --model pix2pix  --which_direction AtoB --dataset_mode simple_grid --loadSize 540 --fineSize 256 --input_nc 4
