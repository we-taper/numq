#!/usr/bin/env bash

docker build --tag numq_test --file docker/Dockerfile .
docker run --gpus all --rm -it numq_test
docker image rm numq_test
