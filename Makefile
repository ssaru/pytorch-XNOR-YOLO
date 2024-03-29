all: test

base_image=nvidia/cuda:11.3.0-runtime-ubuntu20.04
python_ver=3.8.10
image_name=pytorch-xnor-net:v0
container_name=pytorch-xnor-net

dev-env: set-git set-pre-commit set-test set-pipreqs
dep: extract-requirements
test: pytest
build: bulid-docker
clean: clean-pyc clean-test

#### dev-env ####
set-git:
	git config --local commit.template .gitmessage.txt

set-pre-commit:
	python3 -m pip install --no-cache-dir pre-commit==2.11.1
	pre-commit install

set-test:
	python3 -m pip install --no-cache-dir pytest==6.2.1 pytest-cov==2.10.1 pytest_xdist==2.2.0

set-pipreqs:
	python3 -m pip install --no-cache-dir pipreqs==0.4.10
#################

#####  dep  #####
extract-requirements:
	pipreqs --force --savepath requirements.txt src

# -n 1
pytest:
	pytest -o log_cli=true --disable-pytest-warnings --cov-report term-missing tests/

check:
	pre-commit run -a
#################

## train conv ##
train-conv:
	python3 main.py train --dataset-config conf/conv/data/data.yml --model-config conf/conv/model/model.yml --runner-config conf/conv/training/training.yml

train-mlp:
	python3 main.py train --dataset-config conf/mlp/data/data.yml --model-config conf/mlp/model/model.yml --runner-config conf/mlp/training/training.yml

#### docker #####
build-docker:
	docker build -f docker/Dockerfile -t $(image_name) . --build-arg BASE_IMAGE=$(base_image) --build-arg PYTHON_VER=$(python_ver) --no-cache

run-docker:
	docker run -i -t -d --shm-size=8G --init --name $(container_name) $(image_name)

exec-docker:
	docker exec -it $(container_name) /bin/bash

rm-docker:
	docker stop $(container_name) && docker rm $(container_name)
#################

####  clean  ####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf tests/output
	rm -rf *.log
	rm -rf *.png
	rm -rf src/measure/input/detection-results/*
	rm -rf src/measure/input/ground-truth/*
#################
