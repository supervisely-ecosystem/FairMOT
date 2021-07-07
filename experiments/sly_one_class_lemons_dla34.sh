cd src
python train.py mot --exp_id sly_one_class_lemons --num_classes 1 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --multi_loss fix
cd ..
