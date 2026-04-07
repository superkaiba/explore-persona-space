#!/bin/bash
# Cluster detection script
# Detects which cluster environment we're running on based on environment fingerprints

detect_cluster() {
    # Mila cluster - check hostname pattern or unique paths
    if [[ "$HOSTNAME" == *".server.mila.quebec"* ]] || [[ -d "/network/projects" ]]; then
        echo "mila"
        return 0
    fi

    # Compute Canada - check for CC-specific paths/env vars
    if [[ -d "/cvmfs/soft.computecanada.ca" ]] || [[ -n "$CC_CLUSTER" ]]; then
        echo "computecanada"
        return 0
    fi

    # Hyperbolic - check for their specific markers
    if [[ -n "$HYPERBOLIC_API_KEY" ]] || [[ -f "/etc/hyperbolic-host" ]]; then
        echo "hyperbolic"
        return 0
    fi

    # RunPod - Docker container markers
    if [[ -d "/runpod-volume" ]] || [[ -n "$RUNPOD_POD_ID" ]]; then
        echo "runpod"
        return 0
    fi

    # Fallback - local/unknown environment
    echo "local"
    return 0
}

# Export the detected cluster if sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    export DOTFILES_CLUSTER=$(detect_cluster)
fi
