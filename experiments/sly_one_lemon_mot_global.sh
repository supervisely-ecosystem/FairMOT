cd src
#python train.py mot --exp_id sly_mot_global_1 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 28 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo'
#python train.py mot --exp_id sly_mot_global_2 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.1
#python train.py mot --exp_id sly_mot_global_3 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.5
#python train.py mot --exp_id sly_mot_global_4 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.5  --arch 'resdcn_34'
#python train.py mot --exp_id sly_mot_global_5 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.5  --arch 'resdcn_50'
#python train.py mot --exp_id sly_mot_global_6 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.5  --arch 'resfpndcn_34'
#python train.py mot --exp_id sly_mot_global_7 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.5  --arch 'hrnet_18'
#python train.py mot --exp_id sly_mot_global_8 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.1  --arch 'resdcn_34'
#python train.py mot --exp_id sly_mot_global_9 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.1  --arch 'resdcn_50'
#python train.py mot --exp_id sly_mot_global_10 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.1  --arch 'resfpndcn_34'
#python train.py mot --exp_id sly_mot_global_11 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.1  --arch 'hrnet_18'
#python train.py mot --exp_id sly_mot_global_12 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 1  --arch 'resdcn_34'
#python train.py mot --exp_id sly_mot_global_13 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 1  --arch 'resdcn_50'
#python train.py mot --exp_id sly_mot_global_14 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 1  --arch 'resfpndcn_34'
#python train.py mot --exp_id sly_mot_global_15 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 1  --arch 'hrnet_18'
#python train.py mot --exp_id sly_mot_global_16 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 1
#python train.py mot --exp_id sly_mot_global_17 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.1 --ltrb False
#python train.py mot --exp_id sly_mot_global_18 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 0.1 --off_weight 0.1
#python train.py mot --exp_id sly_mot_global_19 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 1 --off_weight 0.1 --id_weight 0.1
python train.py mot --exp_id sly_mot_global_21 --num_epochs 20 --lr_step 12 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 5 --wh_weight 1

cd ..
