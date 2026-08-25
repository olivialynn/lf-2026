#!/usr/bin/env bash
set -euo pipefail

output_dir="/Users/orl/code/lsdb-plus/liv-lf/10 - EzTaoX/milliquas_raw"
urls=(
    "https://quasars.org/milliquas.fits.zip"
)

mkdir -p "$output_dir"

download() {
    local url=$1
    local target=$2

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --progress-bar -o "$target" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$target" "$url"
    else
        echo "Missing curl or wget. Install one and retry." >&2
        exit 1
    fi
}

for url in "${urls[@]}"; do
    echo "Downloading: $url"
    filename=$(basename "$url")
    temp_file="$output_dir/$filename.partial"
    final_file="$output_dir/$filename"

    download "$url" "$temp_file"
    mv "$temp_file" "$final_file"
    echo "Saved: $final_file"
done

echo "Done."
