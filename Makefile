OUTPUT_PATH := /scratch/results
WEIGHTS_PATH := /scratch/mistral-medium-2507

TP_SIZE := 4

SHARED_ARGS := --no-enable-prefix-caching --served-model-name default_model
MISTRAL_ARGS := --config-format mistral --load-format mistral --tokenizer-mode mistral

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

serve-tke:
	vllm serve $(WEIGHTS_PATH) \
	--enforce-eager \
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