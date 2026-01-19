OUTPUT_PATH := /scratch/results
WEIGHTS_PATH := /scratch/mistral-medium-2507

TP_SIZE := 4

SHARED_ARGS := --no-enable-prefix-caching --served-model-name default_model
MISTRAL_ARGS := --config-format mistral --load-format mistral --tokenizer-mode mistral

install-nsys:
	@. /etc/lsb-release && \
	UBUNTU_RELEASE=$$(echo "$$DISTRIB_RELEASE" | tr -d .) && \
	UBUNTU_RELEASE_TO_GET_OLD_KEY=1804 && \
	KERNEL_ARCH=$$(uname -m) && \
	DPKG_ARCH=$$(dpkg --print-architecture) && \
	OLD_KEY=7fa2af80 && \
	OLD_KEY_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$${UBUNTU_RELEASE_TO_GET_OLD_KEY}/$${KERNEL_ARCH}/$${OLD_KEY}.pub" && \
	echo "Updating and installing prerequisites..." && \
	sudo apt update && \
	sudo apt install -y --no-install-recommends gnupg ca-certificates wget && \
	echo "Adding NVIDIA DevTools repository..." && \
	echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$${UBUNTU_RELEASE}/$${DPKG_ARCH} /" | sudo tee /etc/apt/sources.list.d/nvidia-devtools.list && \
	echo "Adding NVIDIA public key..." && \
	sudo wget -O - "$$OLD_KEY_URL" | sudo apt-key add - && \
	echo "Installing nsight-systems-cli..." && \
	sudo apt update && \
	sudo apt install -y nsight-systems-cli



cuda-deps:
	sudo apt-get update && sudo apt-get install -y \
		cuda-nvrtc-dev-12-9 \
		libcublas-dev-12-9 \
		libcusparse-dev-12-9 \
		libcusolver-dev-12-9 \
		ccache

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install uv
	. .venv/bin/activate && uv pip install -r requirements/build.txt
	. .venv/bin/activate && uv pip install -r requirements/cuda.txt \
		--prerelease=allow \
		--index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cu129 \
		--force-reinstall
	. .venv/bin/activate && uv pip install -e . -v \
		--prerelease=allow \
		--index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cu129
	. .venv/bin/activate && uv pip install nvidia-lm-eval math_verify nvtx

serve:
	vllm serve $(WEIGHTS_PATH) \
	--tensor-parallel-size $(TP_SIZE) \
	$(MISTRAL_ARGS) \
	$(SHARED_ARGS)

serve-fp8:
	vllm serve $(WEIGHTS_PATH) \
	--tensor-parallel-size $(TP_SIZE) \
	$(MISTRAL_ARGS) \
	$(SHARED_ARGS) \
	--kv-cache-dtype fp8

serve-tke:
	vllm serve $(WEIGHTS_PATH) \
	--attention-config.backend TKE \
	--tensor-parallel-size $(TP_SIZE) \
	$(MISTRAL_ARGS) \
	$(SHARED_ARGS)

serve-tke-fp8:
	vllm serve $(WEIGHTS_PATH) \
	--attention-config.backend TKE \
	--tensor-parallel-size $(TP_SIZE) \
	$(MISTRAL_ARGS) \
	$(SHARED_ARGS) \
	--kv-cache-dtype fp8

eval-gsm8k:
	nemo-evaluator run_eval \
		--eval_type gsm8k \
		--model_id default_model \
		--model_url http://localhost:8000/v1/completions \
		--model_type completions \
		--output_dir ./ \
		--override config.params.parallelism=1

query-completion:
	curl http://localhost:8000/v1/completions \
	    -H "Content-Type: application/json" \
	    -d '{"model": "default_model", "prompt": "What is the capital of France?", "max_tokens": 128, "temperature": 0}'

query-chat:
	curl -X POST http://localhost:8000/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 256, "model": "default_model"}'