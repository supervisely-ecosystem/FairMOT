cd src
python train.py mot --exp_id sly_one_class_lemons8 --num_classes 1 --num_epochs 25 --lr_step 18 --data_cfg ../src/lib/cfg/sly_mot.json --val_intervals 1 --det_thres 0.9 --conf_thres 0.9
cd ..
