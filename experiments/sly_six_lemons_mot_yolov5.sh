cd src
python train.py mot --exp_id sly_mot_yolov5 --num_epochs 20 --lr_step 15 --data_cfg ../src/lib/cfg/sly_mot.json --lr 5e-4 --batch_size 28 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo'
cd ..
