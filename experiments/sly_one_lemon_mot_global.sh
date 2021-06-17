cd src
python train.py mot --exp_id sly_mot_global_1 --num_epochs 10 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 1e-4 --batch_size 28 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo'
python train.py mot --exp_id sly_mot_global_2 --num_epochs 10 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 2e-4 --batch_size 28 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo'
python train.py mot --exp_id sly_mot_global_3 --num_epochs 10 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 3e-4 --batch_size 28 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo'
python train.py mot --exp_id sly_mot_global_4 --num_epochs 10 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 4e-5 --batch_size 28 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo'
python train.py mot --exp_id sly_mot_global_5 --batch_size 10 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json
cd ..
