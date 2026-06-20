# Go variables
GOCMD=go
GOBUILD=go build
GOCLEAN=go clean
GOLINT=golangci-lint run
GOFMT=gofmt
GOTEST=go test
GOMOD=go mod
GORUN=go run
GOVET=go vet

# Project variables
BINARY_NAME=topic
MODULE_PATH=./pkg/
OUTPUT_DIR=./dist

# Default target executed when running `make`.
.DEFAULT_GOAL := help


build:
	@echo "Building application..."
	@mkdir -p $(OUTPUT_DIR)
	(cd $(MODULE_PATH) && CGO_ENABLED=0 $(GOBUILD) -v -ldflags="-s -w" -o ../$(OUTPUT_DIR)/$(BINARY_NAME) .)

run:
	@echo "Running application..."
	(cd $(MODULE_PATH) && $(GORUN) .)

clean:
	@echo "Cleaning up..."
	@rm -rf $(OUTPUT_DIR)
	$(GOCLEAN)

test:
	@echo "Running tests..."
	(cd $(MODULE_PATH) && $(GOTEST) -coverprofile=coverage.out ./...)

bench:
	@echo "Running benchmarks..."
	(cd $(MODULE_PATH) && $(GOTEST) -bench=. -benchmem -count=5 ./...)

bench-save:
	@echo "Running benchmarks and saving output under tmp/benchmarks..."
	@mkdir -p tmp/benchmarks
	(cd $(MODULE_PATH) && $(GOTEST) -bench=. -benchmem -count=5 ./...) | tee tmp/benchmarks/$$(git rev-parse --short HEAD)-$$(date +%Y%m%d%H%M%S).txt

view-coverage:
	@echo "==> Opening coverage report in browser..."
	(cd $(MODULE_PATH) && go tool cover -html=coverage.out)

deps:
	@echo "Tidying dependencies..."
	(cd $(MODULE_PATH) && $(GOMOD) tidy)
	(cd $(MODULE_PATH) && $(GOMOD) vendor)

format:
	@echo "Formatting code..."
	(cd $(MODULE_PATH) && $(GOFMT) -s -w ./)

lint:
	@echo "Linting code..."
	(cd $(MODULE_PATH) && $(GOLINT))

ci-check: format lint test
	@echo "CI checks passed."

help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  build    Build the application binary"
	@echo "  run      Run the application"
	@echo "  test     Run all tests"
	@echo "  bench    Run benchmarks"
	@echo "  bench-save Run benchmarks and save output in tmp/benchmarks"
	@echo "  lint     Lint the source code"
	@echo "  deps     Tidy and vendor dependencies"
	@echo "  clean    Remove build artifacts"
	@echo "  help     Show this help message"
	@echo ""

.PHONY: all build run clean test bench bench-save deps lint help
