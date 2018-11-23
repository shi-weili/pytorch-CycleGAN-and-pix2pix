set -ex
python train.py --dataroot ./datasets/simple_grid --name planet_mars_gen_11 --model pix2pix  --which_direction AtoB --dataset_mode simple_grid_no_longi --loadSize 540 --fineSize 256 --input_nc 3 --batchSize 1 --norm batch
