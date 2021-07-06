docker-compose up --build -d
CONTAINER_ID=$(docker ps -aqf "name=remote-dev_fairmot")
docker exec -it $CONTAINER_ID sh -c "cd /DCNv2/ && ./make.sh"
docker exec -it $CONTAINER_ID sh -c "tmux new-session -s ses0 -n script -d"
docker exec -it $CONTAINER_ID bash -c "tmux send-keys -t ses0 'cd /FairMOT/ && sh ./exp_and_demo.sh && cd src && python remove_frames_dirs.py' C-m "
echo 'done'
