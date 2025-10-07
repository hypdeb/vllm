create-venv:
	python3 -m venv .venv

install-deps-cuda-dev:
	pip install -r requirements/build.txt
	pip install -r requirements/cuda.txt
	pip install -r requirements/lint.txt

install-deps-local-flashinfer: install-deps-cuda-dev
	pip install ../flashinfer -v

install-editable-sm90: TORCH_CUDA_ARCH_LIST=9.0
install-editable-sm90:
	pip install -e . -v
	