python run_har.py --model "sech_kan" --num_grids 2 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_2__0";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_4__0";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 8 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_8__0";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 16 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_16__0";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 32 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_32__0";
sleep 5s;

python run_har.py --model "sech_kan" --num_grids 2 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_2__1";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_4__1";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 8 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_8__1";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 16 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_16__1";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 32 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_32__1";
sleep 5s;

python run_har.py --model "sech_kan" --num_grids 2 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_2__2";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_4__2";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 8 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_8__2";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 16 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_16__2";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 32 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_32__2";
sleep 5s;


python run_crop.py --model "sech_kan" --num_grids 2 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_2__0";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_4__0";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 8 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_8__0";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 16 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_16__0";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 32 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_32__0";
sleep 5s;

python run_crop.py --model "sech_kan" --num_grids 2 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_2__1";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_4__1";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 8 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_8__1";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 16 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_16__1";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 32 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_32__1";
sleep 5s;

python run_crop.py --model "sech_kan" --num_grids 2 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_2__2";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_4__2";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 8 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_8__2";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 16 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_16__2";
sleep 5s;
python run_crop.py --model "sech_kan" --num_grids 32 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_32__2";
sleep 5s;


python run_ele_dev.py --model "sech_kan" --num_grids 2 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_2__0";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_4__0";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 8 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_8__0";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 16 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_16__0";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 32 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_grid_32__0";
sleep 5s;

python run_ele_dev.py --model "sech_kan" --num_grids 2 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_2__1";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_4__1";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 8 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_8__1";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 16 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_16__1";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 32 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_grid_32__1";
sleep 5s;

python run_ele_dev.py --model "sech_kan" --num_grids 2 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_2__2";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_4__2";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 8 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_8__2";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 16 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_16__2";
sleep 5s;
python run_ele_dev.py --model "sech_kan" --num_grids 32 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_grid_32__2";
sleep 5s;