set -ex
python test.py --dataroot ./datasets/oversized_grid --how_many 722 --name planet_mars_gen_15 --model pix2pix --which_direction AtoB --dataset_mode simple_grid_clongi --loadSize 2048 --fineSize 256 --input_nc 4 --results_dir ./datasets/oversized_grid/results --norm batch
