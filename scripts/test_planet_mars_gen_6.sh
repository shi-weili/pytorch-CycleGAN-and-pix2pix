set -ex
python test.py --dataroot ./datasets/oversized_grid --how_many 722 --name planet_mars_gen_5 --model pix2pix --which_direction AtoB --dataset_mode simple_grid_no_longi --loadSize 1024 --fineSize 1024 --input_nc 3 --results_dir ./datasets/oversized_grid/results
