docker-compose up --build -d
CONTAINER_ID=$(docker ps -aqf "name=remote-dev_fairmot")
docker exec -it $CONTAINER_ID sh -c "cd /DCNv2/ && ./make.sh"

