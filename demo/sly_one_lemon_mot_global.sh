cd src
python demo.py mot --load_model /FairMOT/exp/mot/sly_mot_global_1/ --conf_thres 0.4 --output-root ../demo_output/ --input-video ../videos/ --output-format video --arch 'yolo'
python demo.py mot --load_model /FairMOT/exp/mot/sly_mot_global_2/ --conf_thres 0.4 --output-root ../demo_output/ --input-video ../videos/ --output-format video --arch 'yolo'
python demo.py mot --load_model /FairMOT/exp/mot/sly_mot_global_3/ --conf_thres 0.4 --output-root ../demo_output/ --input-video ../videos/ --output-format video --arch 'yolo'
python demo.py mot --load_model /FairMOT/exp/mot/sly_mot_global_4/ --conf_thres 0.4 --output-root ../demo_output/ --input-video ../videos/ --output-format video --arch 'yolo'
python demo.py mot --load_model /FairMOT/exp/mot/sly_mot_global_5/ --conf_thres 0.4 --output-root ../demo_output/ --input-video ../videos/ --output-format video
cd ..
