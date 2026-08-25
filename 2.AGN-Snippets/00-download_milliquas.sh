
#!/usr/bin/env bash
#
# Fetch the Milliquas v8 FITS catalog and unpack it into a directory that
# contains ONLY FITS files, ready to be handed to hats-import as input_path.

set -euo pipefail
IFS=$'\n\t'

base_dir="/Users/orl/code/lsdb-plus/liv-lf/10 - EzTaoX"

# Archives are staged here. Keep this OUT of the hats-import input path --
# input_path globs every file it finds, and FitsReader will choke on a .zip.
archive_dir="$base_dir/milliquas_archives"

# This is what you pass to ImportArguments(input_path=...). FITS only.
raw_dir="$base_dir/milliquas_raw"

urls=(
    "https://quasars.org/milliquas.fits.zip"
)

mkdir -p "$archive_dir" "$raw_dir"
 
command -v unzip >/dev/null 2>&1 || {
    echo "Missing unzip. Install it and retry." >&2
    exit 1
}
 
fetch() {
    local url=$1 target=$2
    if command -v curl >/dev/null 2>&1; then
        # -C - resumes a partial file. If the server ignores Range requests
        # this can append rather than restart, which is why we verify the
        # archive afterwards instead of trusting the exit code alone.

        curl -L --fail --retry 3 --retry-delay 5 --progress-bar \
             -C - -o "$target" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --continue --tries=3 --progress=bar:force -O "$target" "$url"
    else
        echo "Missing curl or wget. Install one and retry." >&2
        exit 1
    fi
}

for url in "${urls[@]}"; do
    filename=$(basename "$url")
    archive="$archive_dir/$filename"
 
    if [[ -f "$archive" ]] && unzip -tqq "$archive" >/dev/null 2>&1; then
        echo "Already have a valid $filename, skipping download."
    else
        echo "Downloading: $url"
        fetch "$url" "$archive"
 
        # Catches truncation, an HTML error page served with a 200, and a
        # botched resume. Delete on failure so the next run starts clean.
        if ! unzip -tqq "$archive" >/dev/null 2>&1; then
            echo "Downloaded file is not a valid zip archive: $archive" >&2
            echo "Removing it. Re-run to try again." >&2
            rm -f "$archive"
            exit 1
        fi
        echo "Saved: $archive"
    fi

    # -j flattens any directory structure in the archive, -o overwrites so
    # re-running is idempotent.
    echo "Extracting FITS from $filename"
    unzip -o -j "$archive" '*.fits' -d "$raw_dir"
done
 
shopt -s nullglob
extracted=("$raw_dir"/*.fits)
if (( ${#extracted[@]} == 0 )); then
    echo "No .fits files ended up in $raw_dir -- check the archive contents." >&2
    exit 1
fi
 
echo
echo "Ready for hats-import. input_path = $raw_dir"
ls -lh "$raw_dir"/*.fits
