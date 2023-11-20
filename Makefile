SHELL := /bin/zsh
PYTHON := python3.8
POETRY := poetry


.PHONY: help \
	check \
	run-precommit \
	run-tests \
	clear-cache \
	create-docs \
	remove-docs \
	remove-local-branches \
	build-package \
	publish-package \
	install-dependencies \
	install-dependencies-precommit \
	install-dependencies-testing \
	install-dependencies-docs \
	install-dependencies-packaging \
	install-dependencies-pythonkit \
	install-dependencies-pdtlz \
	install-dependencies-sparktlz


help:
	@echo 'commands:'
	@echo ' - check                             ;; run pre-commit and tests consecutively'
	@echo ' - run-precommit                     ;; run pre-commit'
	@echo ' - run-tests                         ;; run pytest with coverage report'
	@echo ' - clear-cache                       ;; clear cache files and directories'
	@echo ' - create-docs                       ;; create local documentation files'
	@echo ' - remove-docs                       ;; remove local documentation files'
	@echo ' - remove-local-branches             ;; remove local git branches, except main'
	@echo ' - build-package                     ;; auxiliary command to build sdist and wheel distributions with poetry'
	@echo ' - publish-package                   ;; auxiliary command to publish package to PyPI with poetry'
	@echo ' - install-dependencies              ;; auxiliary command to install all dependencies with poetry if there is no poetry.lock'
	@echo ' - install-dependencies-precommit    ;; auxiliary command to install dependencies with poetry for pre-commit'
	@echo ' - install-dependencies-testing      ;; auxiliary command to install dependencies with poetry for testing'
	@echo ' - install-dependencies-docs         ;; auxiliary command to install dependencies with poetry for documentation'
	@echo ' - install-dependencies-packaging    ;; auxiliary command to install dependencies with poetry for packaging'
	@echo ' - install-dependencies-pythonkit    ;; auxiliary command to install dependencies with poetry for pythonkit'
	@echo ' - install-dependencies-pdtlz        ;; auxiliary command to install dependencies with poetry for pandas toolz'
	@echo ' - install-dependencies-sparktlz     ;; auxiliary command to install dependencies with poetry for spark toolz'


check: run-precommit run-tests


run-precommit:
	pre-commit run --all-files
	@echo


run-tests:
	$(PYTHON) -m pytest --doctest-modules --ignore-glob="src/onekit/sparktlz.py" src/ --cov-report term-missing --cov=src/ tests/
	@echo


clear-cache:
	$(PYTHON) scripts/clear_cache.py
	@echo


create-docs:
	cd docs; make html; cd ..;


remove-docs:
	@rm -rf docs/_build/


remove-local-branches:
	git -P branch | grep -v "main" | grep -v \* | xargs git branch -D


build-package:
	$(POETRY) build


publish-package:
	$(POETRY) publish


install-dependencies: install-dependencies-precommit \
	install-dependencies-testing \
	install-dependencies-docs \
	install-dependencies-packaging \
	install-dependencies-pythonkit \
	install-dependencies-pdtlz \
	install-dependencies-sparktlz


install-dependencies-precommit:
	$(POETRY) add --group precommit \
	autoflake \
	"black[jupyter]" \
	isort \
	flake8 \
	pre-commit \
	pre-commit-hooks


install-dependencies-testing:
	$(POETRY) add --group testing \
	pytest \
	pytest-cov \
	pytest-skip-slow


install-dependencies-docs:
	$(POETRY) add --group docs \
	furo \
	jupyterlab \
	myst-parser \
	nbsphinx \
	sphinx-autoapi \
	sphinx-copybutton \
	time-machine


install-dependencies-packaging:
	$(POETRY) add --group packaging \
	python-semantic-release


install-dependencies-pythonkit:
	$(POETRY) add \
	toolz


install-dependencies-pdtlz:
	$(POETRY) add --group pdtlz \
	"pandas>=0.23.2"


install-dependencies-sparktlz:
	$(POETRY) add --group sparktlz \
	pyspark==3.1.1
