#!/bin/bash
#
# This script downloads and installs the latest release of 'topic' for your system.
# It will attempt to install system-wide, but will fall back to a local user
# installation if it lacks the necessary permissions.
#
# Usage: curl -s https://raw.githubusercontent.com/yurirocha15/topic/master/install.sh | bash
#

set -e

# --- Configuration ---
REPO="yurirocha15/topic"
BINARY_NAME="topic"
PRIMARY_INSTALL_DIR="/usr/local/bin"
FALLBACK_INSTALL_DIR="$HOME/.local/bin"

# --- Helper Functions ---
info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

warn() {
    echo -e "\033[33m[WARN]\033[0m $1"
}

error() {
    echo -e "\033[31m[ERROR]\033[0m $1" >&2
    exit 1
}

# --- Main Script ---

# 1. Detect the system architecture.
info "Detecting system architecture..."
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
DOWNLOAD_URL=$(curl -s $API_URL | grep "browser_download_url.*-linux-${ARCH}" | cut -d '"' -f 4)

if [ -z "$DOWNLOAD_URL" ]; then
    error "Could not find a download URL for the latest release for architecture $ARCH."
fi
info "Found download URL: $DOWNLOAD_URL"

# 3. Download the binary to a temporary file.
info "Downloading binary..."
TMP_FILE=$(mktemp)
curl -sL -o "$TMP_FILE" "$DOWNLOAD_URL"
chmod +x "$TMP_FILE"

# 4. Determine install location and install.
INSTALL_DIR=""
# Try to install system-wide first.
if [ -w "$PRIMARY_INSTALL_DIR" ]; then
    INSTALL_DIR="$PRIMARY_INSTALL_DIR"
    info "Installing to system-wide directory: $INSTALL_DIR"
    install -m 755 "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
elif command -v sudo &> /dev/null; then
    INSTALL_DIR="$PRIMARY_INSTALL_DIR"
    info "Write permission denied to $INSTALL_DIR. Attempting with sudo."
    echo "Please enter your password if prompted."
    sudo install -m 755 "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
else
    # Fallback to user-local installation.
    INSTALL_DIR="$FALLBACK_INSTALL_DIR"
    info "Sudo not available or failed. Falling back to user-local install: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    install -m 755 "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"

    # Check if the user's PATH includes the fallback directory.
    case ":$PATH:" in
        *":$INSTALL_DIR:"*)
            # Already in PATH
            ;;
        *)
            warn "Your PATH does not seem to include $INSTALL_DIR."
            warn "You may need to add it to your shell's startup file (e.g., ~/.bashrc or ~/.zshrc):"
            warn "  export PATH=\"\$PATH:$INSTALL_DIR\""
            ;;
    esac
fi

# 5. Clean up and verify.
rm -f "$TMP_FILE"
info "Verifying installation..."
if ! command -v $BINARY_NAME &> /dev/null; then
    error "Installation failed. '$BINARY_NAME' not found in your PATH."
fi

VERSION=$($BINARY_NAME --version 2>/dev/null || echo "(version not available)")
info "Successfully installed $BINARY_NAME version: $VERSION"
info "You can now run '$BINARY_NAME' from anywhere in your terminal."
