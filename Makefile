.PHONY: build
build: clean-build ## Build wheel file using poetry
	@echo "🚀 Creating wheel file"
	@zip -r dist/libs.zip libs
	@zip -r dist/data.zip KPA_2021_shared_task

.PHONY: clean-build
clean-build: ## clean build artifacts
	@rm -rf dist/*

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help