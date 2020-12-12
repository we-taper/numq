#!/usr/bin/env bash

docker build --tag numq --file Dockerfile ..
docker run --gpus all --rm numq
