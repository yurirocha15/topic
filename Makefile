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
GOVULNCHECK=golang.org/x/vuln/cmd/govulncheck@v1.9.0

# Project variables
BINARY_NAME=topic
# The Go module lives under pkg/ to keep build artifacts, static assets,
# and documentation at the repository root separate from source code.
MODULE_PATH=./pkg/
OUTPUT_DIR=./dist

# Default target executed when running `make`.
.DEFAULT_GOAL := help


build:
	@echo "Building application..."
	@mkdir -p $(OUTPUT_DIR)
	(cd $(MODULE_PATH) && CGO_ENABLED=0 $(GOBUILD) -v -ldflags="-s -w" -o ../$(OUTPUT_DIR)/$(BINARY_NAME) .)

# build-nvml builds dist/topic-nvml, which reads GPU usage through NVML and needs glibc.
build-nvml:
	@echo "Building the NVML application..."
	@mkdir -p $(OUTPUT_DIR)
	(cd $(MODULE_PATH) && CGO_ENABLED=0 $(GOBUILD) -v -tags nvml -ldflags="-s -w" -o ../$(OUTPUT_DIR)/$(BINARY_NAME)-nvml .)

# check-builds builds both binaries, checks that the default one is static, and runs each once.
check-builds: build build-nvml
	@if readelf -l $(OUTPUT_DIR)/$(BINARY_NAME) | grep -q 'program interpreter'; then \
		echo "$(OUTPUT_DIR)/$(BINARY_NAME) must be a static binary"; exit 1; fi
	@for binary in $(OUTPUT_DIR)/$(BINARY_NAME) $(OUTPUT_DIR)/$(BINARY_NAME)-nvml; do \
		$$binary --once --json | python3 -c 'import json, sys; json.load(sys.stdin)' || exit 1; \
		echo "$$binary ran"; done

run:
	@echo "Running application..."
	(cd $(MODULE_PATH) && $(GORUN) .)

clean:
	@echo "Cleaning up..."
	@rm -rf $(OUTPUT_DIR)
	(cd $(MODULE_PATH) && $(GOCLEAN))

test:
	@echo "Running tests..."
	(cd $(MODULE_PATH) && $(GOTEST) -coverprofile=coverage.out ./...)

test-nvml:
	@echo "Running tests for the NVML build..."
	(cd $(MODULE_PATH) && $(GOTEST) -tags nvml ./...)

test-race:
	@echo "Running tests with the race detector..."
	(cd $(MODULE_PATH) && $(GOTEST) -race ./...)

test-integrations:
	@echo "Running integration discovery tests..."
	(cd $(MODULE_PATH) && $(GOTEST) -run 'Test(Discover.*Integration|DiscoverDocker|DiscoverKubernetes|DiscoverNVML|IntegrationParsing|LiveIntegration|WriteJSONSnapshot)' ./...)
	@./scripts/smoke-integration.sh

test-release-notes:
	@echo "Running release notes tests..."
	@./scripts/test-release-notes.sh

e2e-docker:
	@echo "Running Docker E2E integration validation..."
	@./scripts/e2e-docker-integration.sh

e2e-kubernetes:
	@echo "Running Kubernetes E2E integration validation..."
	@./scripts/e2e-kubernetes-integration.sh

e2e-integrations: e2e-docker e2e-kubernetes
	@echo "Integration E2E validation passed."

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

format:
	@echo "Formatting code..."
	(cd $(MODULE_PATH) && $(GOFMT) -s -w ./)

format-check:
	@echo "Checking formatting..."
	@unformatted="$$(cd $(MODULE_PATH) && $(GOFMT) -s -l ./)"; \
	if [ -n "$$unformatted" ]; then echo "Run make format for:"; echo "$$unformatted"; exit 1; fi

vuln:
	@echo "Checking for known vulnerabilities..."
	(cd $(MODULE_PATH) && $(GOCMD) run $(GOVULNCHECK) ./...)

lint:
	@echo "Linting code..."
	(cd $(MODULE_PATH) && golangci-lint config verify && $(GOLINT) && $(GOLINT) --build-tags nvml)

ci-check: format-check lint test test-release-notes
	@echo "CI checks passed."

help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  build    Build the application binary"
	@echo "  build-nvml Build the NVML application binary (needs glibc)"
	@echo "  check-builds Build both binaries and run each once"
	@echo "  run      Run the application"
	@echo "  test     Run all tests"
	@echo "  test-nvml Run all tests for the NVML build"
	@echo "  test-race Run all tests with the race detector"
	@echo "  test-integrations Run fake integration tests and JSON smoke validation"
	@echo "  test-release-notes Test the changelog extraction used for releases"
	@echo "  e2e-docker Run Docker environment integration validation"
	@echo "  e2e-kubernetes Run Kubernetes environment integration validation"
	@echo "  e2e-integrations Run all real environment integration validation"
	@echo "  bench    Run benchmarks"
	@echo "  bench-save Run benchmarks and save output in tmp/benchmarks"
	@echo "  lint     Lint the source code"
	@echo "  format   Format the source code"
	@echo "  format-check Fail if the source code is not formatted"
	@echo "  vuln     Check dependencies and the Go toolchain for known vulnerabilities"
	@echo "  deps     Tidy dependencies"
	@echo "  clean    Remove build artifacts"
	@echo "  help     Show this help message"
	@echo ""

.PHONY: build build-nvml check-builds run clean test test-nvml test-race test-integrations test-release-notes e2e-docker e2e-kubernetes e2e-integrations bench bench-save view-coverage deps format format-check vuln lint ci-check help
