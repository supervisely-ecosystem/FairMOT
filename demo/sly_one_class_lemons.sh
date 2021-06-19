cd src
python demo.py mot --num_classes 1 --load_model /FairMOT/exp/mot/sly_one_class_lemons/ --conf_thres 0.4 --output-root ../demo_output/ --input-video ../videos/ --output-format video
cd ..
