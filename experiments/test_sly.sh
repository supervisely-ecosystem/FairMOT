cd ./src
python train.py mot --exp_id mot20_ft_mix_dla34 --num_epochs 20 --lr_step '15' --data_cfg '../src/lib/cfg/sly_mot.json'
cd ..
