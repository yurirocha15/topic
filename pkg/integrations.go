package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	dockerSocketPath         = "/var/run/docker.sock"
	procSelfCgroupPath       = "/proc/self/cgroup"
	kubernetesNamespacePath  = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
	kubernetesTokenPath      = "/var/run/secrets/kubernetes.io/serviceaccount/token" // #nosec G101 -- standard Kubernetes service-account token path, not a secret value.
	kubernetesCAPath         = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
	integrationTimeout       = 500 * time.Millisecond
	integrationStatusCount   = 3
	containerIDMinLength     = 12
	containerIDFullLength    = 64
	integrationDocker        = "docker"
	integrationKubernetes    = "kubernetes"
	integrationNVML          = "nvml"
	integrationDisabled      = "disabled"
	integrationUnavailable   = "unavailable"
	integrationAvailable     = "available"
	kubernetesServiceHostEnv = "KUBERNETES_SERVICE_HOST"
	kubernetesServicePortEnv = "KUBERNETES_SERVICE_PORT"
	kubernetesPodNameEnv     = "HOSTNAME"
	kubernetesNodeNameEnv    = "NODE_NAME"
)

func discoverIntegrations(fs FileReader, config AppConfig) (ContainerMetadata, []IntegrationStatus) {
	metadata := ContainerMetadata{}
	statuses := make([]IntegrationStatus, 0, integrationStatusCount)

	dockerMetadata, dockerStatus := discoverDockerMetadata(fs, config)
	statuses = append(statuses, dockerStatus)
	if dockerStatus.Available {
		metadata = mergeMetadata(metadata, dockerMetadata)
	}

	kubernetesMetadata, kubernetesStatus := discoverKubernetesMetadata(fs, config)
	statuses = append(statuses, kubernetesStatus)
	if kubernetesStatus.Available {
		metadata = mergeMetadata(metadata, kubernetesMetadata)
	}

	statuses = append(statuses, discoverNVMLStatus(config))
	return metadata, statuses
}

func discoverDockerMetadata(fs FileReader, config AppConfig) (ContainerMetadata, IntegrationStatus) {
	if config.DisableDocker {
		return ContainerMetadata{}, IntegrationStatus{Name: integrationDocker, Detail: integrationDisabled}
	}
	if _, err := os.Stat(dockerSocketPath); err != nil {
		return ContainerMetadata{}, IntegrationStatus{Name: integrationDocker, Detail: integrationUnavailable}
	}
	containerID := containerIDFromCgroup(fs)
	if containerID == "" {
		return ContainerMetadata{}, IntegrationStatus{Name: integrationDocker, Detail: "container id not found"}
	}
	metadata, err := queryDockerContainer(containerID)
	if err != nil {
		return ContainerMetadata{}, IntegrationStatus{Name: integrationDocker, Detail: err.Error()}
	}
	metadata.Runtime = integrationDocker
	return metadata, IntegrationStatus{Name: integrationDocker, Available: true, Detail: integrationAvailable}
}

func containerIDFromCgroup(fs FileReader) string {
	content, err := readStringFromFile(procSelfCgroupPath, fs)
	if err != nil {
		return ""
	}
	for _, field := range strings.FieldsFunc(content, func(r rune) bool {
		return r == '/' || r == ':' || r == '\n'
	}) {
		field = strings.TrimSpace(field)
		if len(field) >= containerIDMinLength && isContainerID(field) {
			if len(field) > containerIDFullLength {
				field = field[len(field)-containerIDFullLength:]
			}
			return field
		}
	}
	return ""
}

func isContainerID(value string) bool {
	for _, r := range value {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			return false
		}
	}
	return true
}

func queryDockerContainer(containerID string) (ContainerMetadata, error) {
	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var dialer net.Dialer
			return dialer.DialContext(ctx, "unix", dockerSocketPath)
		},
	}
	client := http.Client{Transport: transport, Timeout: integrationTimeout}
	ctx, cancel := context.WithTimeout(context.Background(), integrationTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodGet,
		"http://docker/containers/"+containerID+"/json",
		http.NoBody,
	)
	if err != nil {
		return ContainerMetadata{}, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return ContainerMetadata{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return ContainerMetadata{}, errors.New(resp.Status)
	}
	var payload struct {
		ID     string `json:"Id"`
		Name   string `json:"Name"`
		Config struct {
			Image  string            `json:"Image"`
			Labels map[string]string `json:"Labels"`
		} `json:"Config"`
	}
	if err = json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return ContainerMetadata{}, err
	}
	return ContainerMetadata{
		ID:     payload.ID,
		Name:   strings.TrimPrefix(payload.Name, "/"),
		Image:  payload.Config.Image,
		Labels: payload.Config.Labels,
	}, nil
}

func discoverKubernetesMetadata(fs FileReader, config AppConfig) (ContainerMetadata, IntegrationStatus) {
	if config.DisableKubernetes {
		return ContainerMetadata{}, IntegrationStatus{Name: integrationKubernetes, Detail: integrationDisabled}
	}
	host := os.Getenv(kubernetesServiceHostEnv)
	pod := os.Getenv(kubernetesPodNameEnv)
	if host == "" || pod == "" {
		return ContainerMetadata{}, IntegrationStatus{Name: integrationKubernetes, Detail: integrationUnavailable}
	}
	namespaceBytes, err := fs.ReadFile(kubernetesNamespacePath)
	if err != nil {
		return ContainerMetadata{}, IntegrationStatus{Name: integrationKubernetes, Detail: "namespace not available"}
	}
	tokenBytes, err := fs.ReadFile(kubernetesTokenPath)
	if err != nil {
		return ContainerMetadata{}, IntegrationStatus{Name: integrationKubernetes, Detail: "token not available"}
	}
	namespace := strings.TrimSpace(string(namespaceBytes))
	metadata := ContainerMetadata{
		Runtime:   integrationKubernetes,
		Namespace: namespace,
		Pod:       pod,
		Node:      os.Getenv(kubernetesNodeNameEnv),
	}
	token := strings.TrimSpace(string(tokenBytes))
	if podMetadata, queryErr := queryKubernetesPod(fs, host, namespace, pod, token); queryErr == nil {
		metadata = mergeMetadata(metadata, podMetadata)
	}
	return metadata, IntegrationStatus{Name: integrationKubernetes, Available: true, Detail: integrationAvailable}
}

func queryKubernetesPod(
	fs FileReader,
	host string,
	namespace string,
	pod string,
	token string,
) (ContainerMetadata, error) {
	port := os.Getenv(kubernetesServicePortEnv)
	if port == "" {
		port = "443"
	}
	certPool := x509.NewCertPool()
	if ca, err := fs.ReadFile(kubernetesCAPath); err == nil {
		certPool.AppendCertsFromPEM(ca)
	}
	client := http.Client{
		Timeout: integrationTimeout,
		Transport: &http.Transport{TLSClientConfig: &tls.Config{
			RootCAs:    certPool,
			MinVersion: tls.VersionTLS12,
		}},
	}
	url := "https://" + net.JoinHostPort(host, port) + "/api/v1/namespaces/" + namespace + "/pods/" + pod
	ctx, cancel := context.WithTimeout(context.Background(), integrationTimeout)
	defer cancel()
	// #nosec G107,G704 -- host/port come from Kubernetes service environment in the current pod.
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return ContainerMetadata{}, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	// #nosec G704 -- request URL is constrained to the Kubernetes API service above.
	resp, err := client.Do(req)
	if err != nil {
		return ContainerMetadata{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return ContainerMetadata{}, errors.New(resp.Status)
	}
	return parseKubernetesPod(resp.Body)
}

func parseKubernetesPod(reader io.Reader) (ContainerMetadata, error) {
	var payload struct {
		Metadata struct {
			Name      string            `json:"name"`
			Namespace string            `json:"namespace"`
			Labels    map[string]string `json:"labels"`
		} `json:"metadata"`
		Spec struct {
			NodeName   string `json:"nodeName"`
			Containers []struct {
				Image string `json:"image"`
			} `json:"containers"`
		} `json:"spec"`
	}
	if err := json.NewDecoder(reader).Decode(&payload); err != nil {
		return ContainerMetadata{}, err
	}
	image := ""
	if len(payload.Spec.Containers) > 0 {
		image = payload.Spec.Containers[0].Image
	}
	return ContainerMetadata{
		Runtime:   integrationKubernetes,
		Name:      payload.Metadata.Name,
		Pod:       payload.Metadata.Name,
		Namespace: payload.Metadata.Namespace,
		Node:      payload.Spec.NodeName,
		Image:     image,
		Labels:    payload.Metadata.Labels,
	}, nil
}

func discoverNVMLStatus(config AppConfig) IntegrationStatus {
	if config.DisableNVML {
		return IntegrationStatus{Name: integrationNVML, Detail: integrationDisabled}
	}
	if _, err := os.Stat("/proc/driver/nvidia/version"); err == nil {
		return IntegrationStatus{
			Name:      integrationNVML,
			Available: true,
			Detail:    "nvidia driver visible; nvidia-smi fallback enabled",
		}
	}
	if _, err := os.Stat(filepath.Join("/usr/lib", "libnvidia-ml.so")); err == nil {
		return IntegrationStatus{
			Name:      integrationNVML,
			Available: true,
			Detail:    "NVML library visible; nvidia-smi fallback enabled",
		}
	}
	return IntegrationStatus{Name: integrationNVML, Detail: integrationUnavailable}
}

func mergeMetadata(base ContainerMetadata, next ContainerMetadata) ContainerMetadata {
	if next.Runtime != "" {
		base.Runtime = next.Runtime
	}
	if next.ID != "" {
		base.ID = next.ID
	}
	if next.Name != "" {
		base.Name = next.Name
	}
	if next.Image != "" {
		base.Image = next.Image
	}
	if next.Namespace != "" {
		base.Namespace = next.Namespace
	}
	if next.Pod != "" {
		base.Pod = next.Pod
	}
	if next.Node != "" {
		base.Node = next.Node
	}
	if len(next.Labels) > 0 {
		base.Labels = next.Labels
	}
	return base
}
