HF_TOKEN := $(shell cat token)
UID := $(shell id -u)
GID := $(shell id -g)
DOCKER_PROGRESS    ?= auto
USER_ID            ?= $(shell id --user)
USER_NAME          ?= $(shell id --user --name)
GROUP_ID           ?= $(shell id --group)
GROUP_NAME         ?= $(shell id --group --name)
IMAGE_TAG_SUFFIX   ?= -$(USER_NAME)
define add_local_user
	docker build \
		--progress $(DOCKER_PROGRESS) \
		--build-arg BASE_IMAGE_WITH_TAG=$(1) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg GROUP_NAME=$(GROUP_NAME) \
		--file Dockerfile \
		--tag myimage$(IMAGE_TAG_SUFFIX) \
		..
endef

build-vllm-image:
	$(call add_local_user,flashinfer_vllm_dev:7204195724929729558)

vllm-setup:
	git clone hypdeb/vllm
	git checkout dan_branch
	VLLM_USE_PRECOMPILED=1 pip install --editable .
	pip install flashinfer-python --index-url https://gitlab-master.nvidia.com/api/v4/projects/179694/packages/pypi/simple

force-reinstall-tke:
	pip install "trtllm-kernel-export @ git+ssh://git@gitlab.com:nvidia/tensorrt-llm/private/tensorrt-llm-kernel-export.git" --force-reinstall --no-input
run-vllm:
	docker run -it --gpus all \
		-v $(shell pwd):$(shell pwd) \
		-v $(shell pwd)/vllm/utils/docker/:/dockercmd:ro \
		-v $(shell pwd)/tmp/hf_cache:/llm_cache/ \
		-v $(SSH_AUTH_SOCK):/ssh-agent \
		-e SSH_AUTH_SOCK=/ssh-agent \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e HF_HOME=$(shell pwd)/tmp/hf_cache \
		--ipc=host \
		-w $(shell pwd) \
		--entrypoint /bin/bash \
		myimage-dblanaru 

build-flashinfer-wheel:
	export FLASHINFER_ENABLE_AOT=1; \
	export TORCH_CUDA_ARCH_LIST='9.0+PTX'; \
	cd 3rdparty; \
	rm -rf flashinfer; \
	git clone https://github.com/flashinfer-ai/flashinfer.git --recursive; \
	cd flashinfer; \
	git checkout v0.2.6.post1 --recurse-submodules; \
	pip install --no-build-isolation --verbose .; \
	pip install build ;\
	python -m flashinfer.aot ;\
	python -m build --no-isolation --wheel

push-flashinfer-wheel:
	# pip install twine
	TWINE_PASSWORD=$(shell cat gitlab_token) \
	TWINE_USERNAME=scout \
	python3 -m twine upload --repository-url https://gitlab-master.nvidia.com/api/v4/projects/179694/packages/pypi 3rdparty/flashinfer/dist/* --verbose

vllm-sample-flashinfer:
	VLLM_ATTENTION_BACKEND=FLASHINFER python vllm_sample.py --model /trt_llm_data/llm-models/llama-3.1-model/Meta-Llama-3.1-8B --enforce-eager --batch-size 3 --output-len 10 --num-iters 1 --num-iters-warmup 0 --prompts-file z_hacky_layer_test/sample_prompts.txt

vllm-sample-flashattn:
	VLLM_ATTENTION_BACKEND=FLASH_ATTN python vllm_sample.py --model /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct-FP8 --enforce-eager --batch-size 3 --output-len 10 --num-iters 1 --num-iters-warmup 0 --prompts-file z_hacky_layer_test/sample_prompts.txt

vllm-sample:
	VLLM_ATTENTION_BACKEND=TKE python vllm_sample.py --model /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct-FP8 --enforce-eager --batch-size 3 --output-len 10 --num-iters 1 --num-iters-warmup 0 --prompts-file z_hacky_layer_test/sample_prompts.txt 

build-model8b-edgar4:
	python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer=meta-llama/Llama-3.1-8B token-norm-dist --num-requests=30 --input-mean=2048 --output-mean=128 --input-stdev=0 --output-stdev=0  > ./tmp/synthetic_2048_128.txt
	trtllm-bench --workspace=./tmp --model meta-llama/Llama-3.1-8B build  --dataset ./tmp/synthetic_2048_128.txt

run-trtllm:
	make -C docker dan-vllm_run DOCKER_RUN_ARGS="-e HF_TOKEN=$(HF_TOKEN) -e HF_HOME=/code/tensorrt_llm/tmp/hf_cache" LOCAL_USER=1

run-base-vllm:
	docker rm -f vllm_flashinfer
	docker run --name vllm_flashinfer --gpus all --ipc host --shm-size 1g -e VLLM_ATTENTION_BACKEND=FLASHINFER -e HF_TOKEN=$(HF_TOKEN) -v $(shell pwd):$(shell pwd) -e HF_HOME=$(shell pwd)/tmp/hf_cache -p 8000:8000 vllm/vllm-openai:latest --model meta-llama/Llama-3.1-8B --dtype float16 --chat-template $(shell pwd)/examples/tool_chat_template_llama3.1_json.jinja
	docker logs -f vllm_flashinfer

trt-llm-setup:
	python scripts/build_wheel.py --use_ccache -p -a native
 
make send-requests:
	bash -c '
	curl -s -X POST http://localhost:8000/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d "{\"model\":\"meta-llama/Llama-3.1-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short poem about the ocean.\"}],\"max_tokens\":10}" 

	curl -s -X POST http://localhost:8000/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d "{\"model\":\"meta-llama/Llama-3.1-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain quantum entanglement in simple terms.\"}],\"max_tokens\":10}" 

	curl -s -X POST http://localhost:8000/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d "{\"model\":\"meta-llama/Llama-3.1-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"List three benefits of exercise.\"}],\"max_tokens\":10}" 

	wait
	echo "Request 1:"; cat req1.json; echo
	echo "Request 2:"; cat req2.json; echo
	echo "Request 3:"; cat req3.json
	'