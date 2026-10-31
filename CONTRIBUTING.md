# Contributing

Bug reports and feature requests are welcome on the [GitHub Issues](https://github.com/yurirocha15/topic/issues) page. This guide covers testing `topic`. See [AGENTS.md](./AGENTS.md) for the project's implementation notes.

## Testing

Run the standard validation gates from the repository root:

```bash
make test
make lint
make build
```

Race detection and benchmarks run directly from the Go module:

```bash
cd pkg
go test -race ./...
go test -bench=. -benchmem -count=5 ./...
```

The test suite includes simulated TUI renders at compact, medium, and wide terminal sizes, plus input-flow coverage for filtering, sorting, process details, and clean shutdown.

## Testing Integrations

The normal test suite uses fake Docker, Kubernetes, and NVML providers so it does not require host services:

```bash
make test-integrations
```

This target also runs `scripts/smoke-integration.sh`, which executes `topic --once --json` and validates that Docker, Kubernetes, and NVML integration statuses are present in the JSON snapshot.

To validate that integrations work inside real environments, use the opt-in E2E targets:

```bash
make e2e-docker
make e2e-kubernetes
make e2e-integrations   # runs both
```

`make e2e-docker` requires Docker and `/var/run/docker.sock`. It builds a local image, runs `topic --once --json` inside Docker with the Docker socket mounted, and asserts Docker metadata is actually available.

`make e2e-kubernetes` requires `kubectl` connected to a cluster. By default it builds a local Docker image named `topic:e2e-kubernetes`; if your cluster cannot see local Docker images, publish an image yourself and run:

```bash
TOPIC_KUBE_E2E_BUILD_IMAGE=0 TOPIC_KUBE_E2E_IMAGE=registry.example.com/topic:e2e make e2e-kubernetes
```

For kind clusters, the script automatically loads the local image when the current context starts with `kind-`.

For k3s clusters, the script can import the local image only when it is run as root or passwordless `sudo` is available. Otherwise, publish the image to a registry or manually import it into k3s/containerd before running the target.

For single-node local clusters where the Kubernetes node can access this checkout path, you can avoid image publishing by mounting the built binary into a stock Alpine pod:

```bash
TOPIC_KUBE_E2E_USE_HOSTPATH=1 make e2e-kubernetes
```

Live integration discovery is opt-in:

```bash
TOPIC_LIVE_INTEGRATION_TESTS=1 make test-integrations
```

Live tests assert that discovery returns well-formed statuses without hanging or crashing. They do not require every integration to be available; unavailable Docker, Kubernetes, or NVIDIA/NVML environments are valid results. Use the E2E targets above when you need to prove an integration is available and returning correct metadata in its native environment.
