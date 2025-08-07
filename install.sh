#!/bin/sh
#
# This script downloads and installs the latest release of 'topic'.
# It supports both x86_64 (amd64) and arm64 architectures.
set -e

# --- Configuration ---
REPO="yurirocha15/topic"
BINARY_NAME="topic"
PRIMARY_INSTALL_DIR="/usr/local/bin"
FALLBACK_INSTALL_DIR="$HOME/.local/bin"

# --- Helper Functions ---
info() {
    printf '\033[32m[INFO]\033[0m %s\n' "$1"
}

warn() {
    printf '\033[33m[WARN]\033[0m %s\n' "$1"
}

error() {
    printf '\033[31m[ERROR]\033[0m %s\n' "$1" >&2
    exit 1
}

# http_get downloads a file from a URL and outputs it to stdout.
# It checks for curl first, then falls back to wget.
http_get() {
    URL="$1"
    if command -v curl >/dev/null 2>&1; then
        curl -sL "$URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$URL"
    else
        error "Either curl or wget is required to download files."
    fi
}

# --- Main Script ---

# 1. Detect the system architecture.
info "Detecting system architecture..."
ARCH=$(uname -m)
case $ARCH in
    x86_64 | amd64)
        ARCH="amd64"
        ;;
    aarch64 | aarch64_be | arm64 | armv8b | armv8l)
        ARCH="arm64"
        ;;
    i386 | i686)
        error "32-bit architectures are not supported by this script."
        ;;
    *)
        error "Unsupported architecture: $ARCH. Only x86_64 (amd64) and arm64 are supported."
        ;;
esac
info "Architecture detected: $ARCH"

# 2. Find the latest release URL.
info "Fetching latest release information from GitHub..."
API_URL="https://api.github.com/repos/$REPO/releases/latest"
# Use jq if available for more robust JSON parsing, otherwise fall back to grep/cut.
if command -v jq >/dev/null 2>&1; then
    DOWNLOAD_URL=$(http_get "$API_URL" | jq -r ".assets[] | select(.name | contains(\"linux-${ARCH}\")) | .browser_download_url")
else
    warn "jq is not installed. Falling back to grep, which may be less reliable."
    DOWNLOAD_URL=$(http_get "$API_URL" | grep "browser_download_url.*-linux-${ARCH}" | cut -d '"' -f 4)
fi

if [ -z "$DOWNLOAD_URL" ]; then
    error "Could not find a download URL for the latest release for architecture $ARCH."
fi
info "Found download URL: $DOWNLOAD_URL"

# 3. Download the binary to a temporary file.
info "Downloading binary..."
TMP_FILE=$(mktemp)
http_get "$DOWNLOAD_URL" > "$TMP_FILE"
chmod +x "$TMP_FILE"

# 4. Determine install location and install.
INSTALL_DIR=""
if [ -w "$PRIMARY_INSTALL_DIR" ]; then
    INSTALL_DIR="$PRIMARY_INSTALL_DIR"
    info "Installing to system-wide directory: $INSTALL_DIR"
    install -m 755 "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
elif command -v sudo >/dev/null 2>&1; then
    INSTALL_DIR="$PRIMARY_INSTALL_DIR"
    info "Write permission denied to $INSTALL_DIR. Attempting with sudo."
    printf "Please enter your password if prompted.\n"
    sudo install -m 755 "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
else
    INSTALL_DIR="$FALLBACK_INSTALL_DIR"
    info "Sudo not available or failed. Falling back to user-local install: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    install -m 755 "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"

    case ":$PATH:" in
        *":$INSTALL_DIR:"*)
            ;;
        *)
            warn "Your PATH does not seem to include $INSTALL_DIR."
            warn "You may need to add it to your shell's startup file (e.g., ~/.profile, ~/.bashrc, or ~/.zshrc):"
            warn "  export PATH=\"\$PATH:$INSTALL_DIR\""
            ;;
    esac
fi

# 5. Clean up and verify.
rm -f "$TMP_FILE"
info "Verifying installation..."
if ! command -v "$BINARY_NAME" >/dev/null 2>&1; then
    if [ -x "$INSTALL_DIR/$BINARY_NAME" ]; then
        warn "'$BINARY_NAME' was installed to $INSTALL_DIR, but this directory is not in your PATH."
        error "Installation succeeded, but you won't be able to run '$BINARY_NAME' directly."
    else
        error "Installation failed. Could not find '$BINARY_NAME' in $INSTALL_DIR."
    fi
fi

VERSION=$("$BINARY_NAME" --version 2>/dev/null || echo "(version not available)")
info "Successfully installed $BINARY_NAME version: $VERSION"
info "You can now run '$BINARY_NAME' from anywhere in your terminal."
