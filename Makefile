OUTPUT_PATH ?= .
# MODEL_PATH ?= /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-70B-Instruct-FP8
MODEL_PATH ?= /scratch/usr/quantized_model
LLAMA_8B_PATH ?= /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct
MODEL_PATH_MISTRAL ?= /scratch/mistral-medium-2508_v1.0/weights/mistral-medium-2508/
TOKENIZER_PATH_MISTRAL ?= /scratch/Mistral-Small-3.2-24B-Instruct-2506
TP_SIZE ?= 4

LONG_MODEL_FLAG := VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
FLASH_ATTN_FLAGS := $(LONG_MODEL_FLAG) VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1
TKE_FLAGS := $(LONG_MODEL_FLAG) VLLM_ATTENTION_BACKEND=TKE
FLASH_INFER_FLAGS := $(LONG_MODEL_FLAG) VLLM_ATTENTION_BACKEND=FLASHINFER_VLLM_V1

NSYS_PROFILE ?= 0
ifeq ($(NSYS_PROFILE), 1)
	NSYS_PROFILE_CMD := VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile -t cuda,nvtx,python-gil --python-sampling true -o ${OUTPUT_PATH}/vllm-sample-profile-tke.nsys-rep --trace-fork-before-exec true --cuda-graph-trace node -f true
else
	NSYS_PROFILE_CMD :=
endif

build-vllm-image:
	$(call add_local_user,flashinfer_vllm_dev:7204195724929729558)

vllm-setup:
	VLLM_USE_PRECOMPILED=1 pip install --editable .[bench]
	# pip install flashinfer-python --index-url https://gitlab-master.nvidia.com/api/v4/projects/179694/packages/pypi/simple

vllm-cpp-rebuild:
	CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .

force-reinstall-tke:
	pip install "trtllm-kernel-export @ git+ssh://git@gitlab.com/nvidia/tensorrt-llm/private/tensorrt-llm-kernel-export.git" --force-reinstall --no-input
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

vllm-sample-70b:
	VLLM_ATTENTION_BACKEND=TKE python vllm_sample.py --model /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-70B-Instruct-FP8 --enforce-eager --batch-size 3 --output-len 10 --num-iters 1 --num-iters-warmup 0 --prompts-file z_hacky_layer_test/sample_prompts.txt 

vllm-sample-70b-tp4:
	VLLM_ATTENTION_BACKEND=TKE python vllm_sample.py --model /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-70B-Instruct-FP8 --enforce-eager --batch-size 3 --output-len 10 --num-iters 1 --num-iters-warmup 0 --prompts-file z_hacky_layer_test/sample_prompts.txt --tp 4

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

benchmark-latency: TKE_BACKEND := TKE
benchmark-latency: FLASH_BACKEND := FLASH_ATTN
benchmark-latency: INPUT_LEN := 80000
benchmark-latency: NUM_ITERS_WARMUP := 1
benchmark-latency: BATCH_SIZE := 1
benchmark-latency: NUM_ITERS := 25

# Note: we need VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and to specify the max model length explicitly because vLLM reads the max model length from the tokenizer for some reason.
# It is unclear why the tokenizer would need to know the max model length at all, even less clear why this would be the source of truth for model length.
# I guess there is some explanation for this, but I am clueless at this time.
benchmark-latency:
	VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=$(TKE_BACKEND) $(NSYS_PROFILE_CMD) python benchmarks/benchmark_latency.py \
		--model $(MODEL_PATH) \
		--tensor-parallel-size $(TP_SIZE) \
		--quantization modelopt \
		--input-len $(INPUT_LEN) \
		--num-iters-warmup $(NUM_ITERS_WARMUP) \
		--batch-size $(BATCH_SIZE) \
		--num-iters $(NUM_ITERS) \
		--kv-cache-dtype fp8 \
		--enforce-eager
	# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 LLM_ATTENTION_BACKEND=$(FLASH_BACKEND) $(NSYS_PROFILE_CMD) python benchmarks/benchmark_latency.py \
	# 	--model $(MODEL_PATH) \
	# 	--tensor-parallel-size $(TP_SIZE) \
	# 	--quantization modelopt \
	# 	--input-len $(INPUT_LEN) \
	# 	--num-iters-warmup $(NUM_ITERS_WARMUP) \
	# 	--batch-size $(BATCH_SIZE) \
	# 	--num-iters $(NUM_ITERS) \
	# 	--kv-cache-dtype fp8 \
	# 	--enforce-eager

###########################################################################################
######################## Utilities ########################################################
###########################################################################################

delete-vllm-cache:
	rm -rf ~/.cache/vllm

###########################################################################################
######################## Samples ##########################################################
###########################################################################################

vllm-sample-flashinfer-v1: delete-vllm-cache
	$(FLASH_INFER_FLAGS) $(NSYS_PROFILE_CMD) python vllm_sample.py \
	--model /scratch/usr/quantized_model/ \
	--batch-size 1 \
	--prompts-file sample_prompts.txt \
	--num-iters 1 \
	--num-iters-warmup 0 \
	--tensor-parallel-size 4 > flashinfer.txt 2>&1

vllm-sample-tke: delete-vllm-cache
	$(TKE_FLAGS) $(NSYS_PROFILE_CMD) python vllm_sample.py \
	--model /scratch/usr/quantized_model/ \
	--batch-size 8 \
	--prompts-file sample_prompts.txt \
	--num-iters 1 \
	--num-iters-warmup 0 \
	--kv-cache-dtype fp8 \
	--enforce-eager \
	--enable-specdec-metrics \
	--tensor-parallel-size 4 > tke_out.txt 2>&1
	# nsys stats --force-export=true --timeunit milliseconds  vllm-sample-profile-tke.nsys-rep > nsys_txt


vllm-sample-flashinfer: delete-vllm-cache
	$(FLASH_INFER_FLAGS) $(NSYS_PROFILE_CMD) python vllm_sample.py \
	--model /scratch/usr/quantized_model/ \
	--batch-size 8 \
	--prompts-file sample_prompts.txt \
	--num-iters 1 \
	--num-iters-warmup 0 \
	--kv-cache-dtype fp8 \
	--enforce-eager \
	--enable-specdec-metrics \
	--tensor-parallel-size 4 > tke_out.txt 2>&1
	# nsys stats --force-export=true --timeunit milliseconds  vllm-sample-profile-tke.nsys-rep > nsys_txt
vllm-sample-dataset: delete-vllm-cache
	$(TKE_FLAGS) $(NSYS_PROFILE_CMD) python vllm_sample.py \
	--model /scratch/usr/quantized_model/ \
	--batch-size 8 \
	--prompts-file dataset_preprocessed.txt \
	--max-samples 50 \
	--num-iters 1 \
	--num-iters-warmup 0 \
	--kv-cache-dtype fp8 \
	--enforce-eager \
	--output-json nonspeculative_dataset_output_v2.json \
	--tensor-parallel-size 4 > tke_nonspeculative_out_v2.txt 2>&1

vllm-sample-dataset-speculative: delete-vllm-cache
	$(TKE_FLAGS) $(NSYS_PROFILE_CMD) python vllm_sample.py \
	--model /scratch/usr/quantized_model/ \
	--batch-size 8 \
	--dataset-path /scratch/usr/AA_dataset.txt \
	--max-samples 50 \
	--num-iters 1 \
	--num-iters-warmup 0 \
	--kv-cache-dtype fp8 \
	--enforce-eager \
	--enable-specdec-metrics \
	--num-speculative-tokens 7 \
	--prompt-lookup-max 3 \
	--output-json speculative_dataset_output.json \
	--save-preprocessed dataset_preprocessed.txt \
	--tensor-parallel-size 4 > tke_speculative_out.txt 2>&1

vllm-sample-flash-attn: delete-vllm-cache
	$(FLASH_ATTN_FLAGS) $(NSYS_PROFILE_CMD) python vllm_sample.py \
	--model /scratch/usr/quantized_model/ \
	--batch-size 1 \
	--prompts-file sample_prompts.txt \
	--num-iters 1 \
	--num-iters-warmup 0 \
	--kv-cache-dtype fp8 \
	--enforce-eager \
	--enable-specdec-metrics \
	--tensor-parallel-size 4  > flash_attn.txt 2>&1
	# nsys stats --force-export=true --timeunit milliseconds  vllm-sample-profile-flashattn.nsys-rep > nsys_flash_attn.txt

vllm-sample-flash-attn-draft: delete-vllm-cache
	$(FLASH_ATTN_FLAGS) $(NSYS_PROFILE_CMD) python vllm_sample.py \
	--model /scratch/usr/quantized_model/ \
	--batch-size 1 \
	--prompts-file sample_prompts.txt \
	--num-iters 1 \
	--num-iters-warmup 0 \
	--kv-cache-dtype fp8 \
	--tensor-parallel-size 4 \
	--draft-model-path $(DRAFT_MODEL_PATH) \
	--num-speculative-tokens 5 > flash_attn_draft.txt 2>&1


all-samples: vllm-sample-flash-attn vllm-sample-tke vllm-sample-flashinfer-v1

###########################################################################################
######################## Accuracy #########################################################
###########################################################################################

ACCURACY_BATCH_SIZE ?= 16

# Infra
install-lm-eval:
	cd .. && git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
	cd ../lm-evaluation-harness && pip install -e . -v
	pip install git+https://github.com/felipemaiapolo/tinyBenchmarks

# Small accuracy tests
serve_tke:
	$(TKE_FLAGS) vllm serve $(MODEL_PATH) --tensor-parallel-size $(TP_SIZE) --quantization modelopt --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --enforce-eager --no-enable-prefix-caching --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'

serve_flashinfer:
	$(FLASH_INFER_FLAGS) vllm serve $(MODEL_PATH) --tensor-parallel-size $(TP_SIZE) --quantization modelopt --gpu-memory-utilization 0.95 --enforce-eager --no-enable-prefix-caching --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'

query:
	curl http://localhost:8000/v1/completions \
	    -H "Content-Type: application/json" \
	    -d '{"model": "default_model", "prompt": "The capital of france is ", "max_tokens": 15, "temperature": 0}'

	echo


lm-eval-tiny-hellaswag-tke-fp8-ngram: delete-vllm-cache
	$(TKE_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyHellaswag-tke.json \
		--model_args '{"pretrained": "$(MODEL_PATH)", "tensor_parallel_size": $(TP_SIZE), "quantization": "modelopt", "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8", "speculative_config": {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}}'

lm-eval-tiny-hellaswag-tke-fp8-eager: delete-vllm-cache
	$(TKE_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyHellaswag-tke.json \
		--model_args '{"enforce_eager": true, "pretrained": "$(MODEL_PATH)", "tensor_parallel_size": $(TP_SIZE), "quantization": "modelopt", "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8"}' \
		 > tke_out_eager.txt 2>&1

lm-eval-tiny-hellaswag-tke-fp8: delete-vllm-cache
	$(TKE_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyHellaswag-tke.json \
		--model_args '{"pretrained": "$(MODEL_PATH)", "tensor_parallel_size": $(TP_SIZE), "quantization": "modelopt", "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8"}' \
		 > tke_out.txt 2>&1

lm-eval-tiny-hellaswag-tke-llama-8b: delete-vllm-cache
	$(TKE_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyHellaswag-tke.json \
		--model_args '{"pretrained": "$(LLAMA_8B_PATH)", "tensor_parallel_size": $(TP_SIZE), "gpu_memory_utilization": 0.95}' \
		 > tke_out.txt 2>&1

lm-eval-tiny-hellaswag-tke-llama-8b-fp8: delete-vllm-cache
	$(TKE_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyHellaswag-tke.json \
		--model_args '{"pretrained": "$(LLAMA_8B_PATH)", "tensor_parallel_size": $(TP_SIZE), "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8"}' \
		 > tke_out_llama_8b_fp8.txt 2>&1

lm-eval-tiny-hellaswag-tke-mistral-fp8-ngram: delete-vllm-cache
	$(TKE_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyHellaswag-tke.json \
		--model_args '{"pretrained": "$(MODEL_PATH_MISTRAL)", "tensor_parallel_size": $(TP_SIZE), "tokenizer": "$(TOKENIZER_PATH_MISTRAL)", "tokenizer_mode": "mistral", "load_format": "mistral", "config_format": "mistral", "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8", "speculative_config": {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}}'

lm-eval-tiny-hellaswag-tke-mistral-fp8: delete-vllm-cache
	$(TKE_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyHellaswag-tke.json \
		--model_args '{"pretrained": "$(MODEL_PATH_MISTRAL)", "tensor_parallel_size": $(TP_SIZE), "tokenizer": "$(TOKENIZER_PATH_MISTRAL)", "tokenizer_mode": "mistral", "load_format": "mistral", "config_format": "mistral", "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8"}'


lm-eval-tiny-hellaswag-flash-attn: delete-vllm-cache
	$(FLASH_ATTN_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyHellaswag-flash-attn.json \
		--model_args "pretrained=$(MODEL_PATH),tensor_parallel_size=$(TP_SIZE),quantization=modelopt,gpu_memory_utilization=0.95,kv_cache_dtype=fp8"

lm-eval-tiny-hellaswag-flashinfer: delete-vllm-cache
	$(FLASH_INFER_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyHellaswag-flashinfer.json \
		--model_args "pretrained=$(MODEL_PATH),tensor_parallel_size=$(TP_SIZE),quantization=modelopt,gpu_memory_utilization=0.95"


lm-eval-tiny-tinyGSM8k-tke: delete-vllm-cache
	$(TKE_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyGSM8k \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyGSM8k-tke.json \
		--model_args "pretrained=$(MODEL_PATH),tensor_parallel_size=$(TP_SIZE),quantization=modelopt,gpu_memory_utilization=0.95,kv_cache_dtype=fp8"

lm-eval-tiny-tinyGSM8k-flash-attn: delete-vllm-cache
	$(FLASH_ATTN_FLAGS) lm_eval \
		--model vllm \
		--tasks tinyGSM8k \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-tinyGSM8k-flash-attn.json \
		--model_args "pretrained=$(MODEL_PATH),tensor_parallel_size=$(TP_SIZE),quantization=modelopt,gpu_memory_utilization=0.95,kv_cache_dtype=fp8"

small-accuracy-tests: lm-eval-tiny-hellaswag-tke lm-eval-tiny-hellaswag-flash-attn lm-eval-tiny-tinyGSM8k-tke lm-eval-tiny-tinyGSM8k-flash-attn

# Big accuracy tests

lm-eval-hellaswag-tke: delete-vllm-cache
	$(TKE_FLAGS) lm_eval \
		--model vllm \
		--tasks hellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-hellaswag-tke.json \
		--model_args '{"pretrained": "$(MODEL_PATH)", "tensor_parallel_size": $(TP_SIZE), "quantization": "modelopt", "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8", "speculative_config": {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}}'

lm-eval-GSM8k-tke: delete-vllm-cache
	$(TKE_FLAGS) lm_eval \
		--model vllm \
		--tasks GSM8k \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-GSM8k-tke.json \
		--model_args '{"pretrained": "$(MODEL_PATH)", "tensor_parallel_size": $(TP_SIZE), "quantization": "modelopt", "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8", "speculative_config": {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}}'

lm-eval-hellaswag-flash-attn: delete-vllm-cache
	$(FLASH_ATTN_FLAGS) lm_eval \
		--model vllm \
		--tasks hellaswag \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-hellaswag-flash-attn.json \
		--model_args '{"pretrained": "$(MODEL_PATH)", "tensor_parallel_size": $(TP_SIZE), "quantization": "modelopt", "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8", "speculative_config": {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}}'

lm-eval-GSM8k-flash-attn: delete-vllm-cache
	$(FLASH_ATTN_FLAGS) lm_eval \
		--model vllm \
		--tasks GSM8k \
		--batch_size $(ACCURACY_BATCH_SIZE) \
		--output_path $(OUTPUT_PATH)/lm-eval-results-GSM8k-flash-attn.json \
		--model_args '{"pretrained": "$(MODEL_PATH)", "tensor_parallel_size": $(TP_SIZE), "quantization": "modelopt", "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8", "speculative_config": {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}}'

big-accuracy-tests: lm-eval-hellaswag-tke lm-eval-GSM8k-tke lm-eval-hellaswag-flash-attn lm-eval-GSM8k-flash-attn

all-accuracy-tests: small-accuracy-tests big-accuracy-tests

###########################################################################################
######################## Benchmarking ####################################################
###########################################################################################

# Serving benchmarks using Python script (recommended)
benchmark-serving:
	python benchmark_serving_runner.py \
		--model $(MODEL_PATH) \
		--tp-size $(TP_SIZE) \
		--output-path $(OUTPUT_PATH)

# Analyze benchmark results and create CSV summary
analyze-results:
	python analyze_benchmark_results.py \
		--input-dir $(OUTPUT_PATH) \
		--output $(OUTPUT_PATH)/benchmark_results_summary.csv

# Run full benchmark and analysis pipeline
benchmark-and-analyze: benchmark-serving analyze-results
	@echo "Benchmark and analysis completed!"
	@echo "Results saved in: $(OUTPUT_PATH)/"
	@echo "CSV summary: $(OUTPUT_PATH)/benchmark_results_summary.csv"

vllm-serve-flash-attn: FLASH_BACKEND := FLASH_ATTN
vllm-serve-flash-attn:
	 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=$(FLASH_BACKEND) $(NSYS_PROFILE_CMD) vllm serve \
	 	$(MODEL_PATH) \
	 	--tensor-parallel-size $(TP_SIZE) \
		--config-format mistral \
		--load-format mistral \
		--tokenizer-mode mistral \
		--tokenizer $(TOKENIZER_PATH) \
	 	--kv-cache-dtype fp8 \
		--served-model-name default_model \
		--enforce-eager

vllm-serve-tke-mistral: TKE_BACKEND := TKE
vllm-serve-tke-mistral:
	 VLLM_ATTENTION_BACKEND=$(TKE_BACKEND) $(NSYS_PROFILE_CMD) vllm serve \
	 	$(MODEL_PATH_MISTRAL) \
	 	--tensor-parallel-size $(TP_SIZE) \
		--config-format mistral \
		--load-format mistral \
		--tokenizer-mode mistral \
		--tokenizer $(TOKENIZER_PATH_MISTRAL) \
		--served-model-name default_model

vllm-serve-tke-mistral-fp8: TKE_BACKEND := TKE
vllm-serve-tke-mistral-fp8:
	 VLLM_ATTENTION_BACKEND=$(TKE_BACKEND) $(NSYS_PROFILE_CMD) vllm serve \
	 	$(MODEL_PATH_MISTRAL) \
	 	--tensor-parallel-size $(TP_SIZE) \
		--config-format mistral \
		--load-format mistral \
		--tokenizer-mode mistral \
		--tokenizer $(TOKENIZER_PATH_MISTRAL) \
		--served-model-name default_model \
	 	--kv-cache-dtype fp8

vllm-serve-flash-attn-mistral-fp8:
	 VLLM_ATTENTION_BACKEND=FLASH_ATTN $(NSYS_PROFILE_CMD) vllm serve \
	 	$(MODEL_PATH_MISTRAL) \
	 	--tensor-parallel-size $(TP_SIZE) \
		--config-format mistral \
		--load-format mistral \
		--tokenizer-mode mistral \
		--tokenizer $(TOKENIZER_PATH_MISTRAL) \
		--served-model-name default_model \
	 	--kv-cache-dtype fp8

vllm-serve-fa-mistral: FLASH_BACKEND := FLASH_ATTN_VLLM_V1
vllm-serve-fa-mistral:
	 VLLM_ATTENTION_BACKEND=$(FLASH_BACKEND) $(NSYS_PROFILE_CMD) vllm serve \
	 	$(MODEL_PATH_MISTRAL) \
	 	--tensor-parallel-size $(TP_SIZE) \
		--config-format mistral \
		--load-format mistral \
		--tokenizer-mode mistral \
		--tokenizer $(TOKENIZER_PATH_MISTRAL) \
		--served-model-name default_model

vllm-serve-fa-mistral-fp8: FLASH_BACKEND := FLASH_ATTN_VLLM_V1
vllm-serve-fa-mistral-fp8:
	 VLLM_ATTENTION_BACKEND=$(FLASH_BACKEND) $(NSYS_PROFILE_CMD) vllm serve \
	 	$(MODEL_PATH_MISTRAL) \
	 	--tensor-parallel-size $(TP_SIZE) \
		--config-format mistral \
		--load-format mistral \
		--tokenizer-mode mistral \
		--tokenizer $(TOKENIZER_PATH_MISTRAL) \
		--served-model-name default_model \
		--kv-cache-dtype fp8

vllm-serve-tke: TKE_BACKEND := TKE
vllm-serve-tke:
	 VLLM_ATTENTION_BACKEND=$(TKE_BACKEND) $(NSYS_PROFILE_CMD) vllm serve \
	 	$(MODEL_PATH) \
	 	--tensor-parallel-size $(TP_SIZE) \
		--quantization modelopt \
		--served-model-name default_model \
		--enforce-eager

vllm-serve-tke-fp8: TKE_BACKEND := TKE
vllm-serve-tke-fp8:
	 VLLM_ATTENTION_BACKEND=$(TKE_BACKEND) $(NSYS_PROFILE_CMD) vllm serve \
	 	$(MODEL_PATH) \
	 	--tensor-parallel-size $(TP_SIZE) \
		--quantization modelopt \
		--served-model-name default_model \
	 	--kv-cache-dtype fp8

vllm-serve-tke-fp8-eager: TKE_BACKEND := TKE
vllm-serve-tke-fp8-eager:
	 VLLM_ATTENTION_BACKEND=$(TKE_BACKEND) $(NSYS_PROFILE_CMD) vllm serve \
	 	$(MODEL_PATH) \
	 	--tensor-parallel-size $(TP_SIZE) \
		--quantization modelopt \
		--served-model-name default_model \
	 	--kv-cache-dtype fp8 \
		--enforce-eager