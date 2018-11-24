set -ex
python train.py --dataroot ./datasets/simple_grid --name planet_mars_gen_14 --model pix2pix  --which_direction AtoB --dataset_mode simple_grid_clongi --loadSize 540 --fineSize 256 --input_nc 4 --batchSize 32 --norm batch --niter 200 --niter_decay 200
