py_files := *.py

env_ok: requirements.txt
	rm -rf env env_ok
	python3.6 -m venv env
	env/bin/pip install -r requirements.txt
	touch env_ok

.PHONY: fmt
fmt: env_ok
	env/bin/isort $(py_files)
	env/bin/black $(py_files)

.PHONY: test
test: check
	env/bin/python -m unittest discover . -p "*.py"

.PHONY: check
check: env_ok
	env/bin/python -m mypy --check-untyped-defs --ignore-missing-imports $(py_files)
	env/bin/python -m flake8 --select F $(py_files)
	env/bin/isort --check $(py_files)
	env/bin/black --check $(py_files)

.PHONY: run_notebook
run_notebook: env_ok
	env/bin/jupyter notebook

.PHONY: clean
clean:
	rm -rf env_ok env

