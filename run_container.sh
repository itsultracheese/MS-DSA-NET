mkdir -p ./outputs

docker run -it \
    --name container_fcd \
    --gpus '"device=0,1"' \
    --network=host \
    --publish 1239:1239 \
    --mount type=bind,src="./inputs",target=/app/inputs \
    --mount type=bind,src="./outputs",target=/app/outputs \
    --rm \
    fcd