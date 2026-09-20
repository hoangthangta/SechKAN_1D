python run_har.py --model "mlp" --hidden_layers "256" --batch_size 64 --epochs 20 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0";
sleep 5s;
python run_har.py --model "cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0";
sleep 5s;
python run_har.py --model "resnet_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0";
sleep 5s;
python run_har.py --model "ds_cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0";
sleep 5s;
python run_har.py --model "efficient_kan" --batch_size 64 --hidden_layers "26" --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0";
sleep 5s;

python run_har.py --model "mlp" --hidden_layers "256" --batch_size 64 --epochs 20 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_1";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 1 --note "run_1";
sleep 5s;
python run_har.py --model "cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 1 --note "run_1";
sleep 5s;
python run_har.py --model "resnet_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 1 --note "run_1";
sleep 5s;
python run_har.py --model "ds_cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 1 --note "run_1";
sleep 5s;
python run_har.py --model "efficient_kan" --batch_size 64 --hidden_layers "26" --epochs 20 --scheduler "OneCycleLR" --seed 1 --note "run_1";
sleep 5s;

python run_har.py --model "mlp" --hidden_layers "256" --batch_size 64 --epochs 20 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_2";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 2 --note "run_2";
sleep 5s;
python run_har.py --model "cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 2 --note "run_2";
sleep 5s;
python run_har.py --model "resnet_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 2 --note "run_2";
sleep 5s;
python run_har.py --model "ds_cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 2 --note "run_2";
sleep 5s;
python run_har.py --model "efficient_kan" --batch_size 64 --hidden_layers "26" --epochs 20 --scheduler "OneCycleLR" --seed 2 --note "run_2";
sleep 5s;


python run_har.py --model "mlp" --hidden_layers "256" --batch_size 64 --epochs 20 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed 3 --note "run_3";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 3 --note "run_3";
sleep 5s;
python run_har.py --model "cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 3 --note "run_3";
sleep 5s;
python run_har.py --model "resnet_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 3 --note "run_3";
sleep 5s;
python run_har.py --model "ds_cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 3 --note "run_3";
sleep 5s;
python run_har.py --model "efficient_kan" --batch_size 64 --hidden_layers "26" --epochs 20 --scheduler "OneCycleLR" --seed 3 --note "run_3";
sleep 5s;


python run_har.py --model "mlp" --hidden_layers "256" --batch_size 64 --epochs 20 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed 4 --note "run_4";
sleep 5s;
python run_har.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 4 --note "run_4";
sleep 5s;
python run_har.py --model "cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 4 --note "run_4";
sleep 5s;
python run_har.py --model "resnet_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 4 --note "run_4";
sleep 5s;
python run_har.py --model "ds_cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 4 --note "run_4";
sleep 5s;
python run_har.py --model "efficient_kan" --batch_size 64 --hidden_layers "26" --epochs 20 --scheduler "OneCycleLR" --seed 4 --note "run_4";
sleep 5s;
