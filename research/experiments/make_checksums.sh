#!/bin/sh
# SHA-256 of every raw result file; verify with: cd experiments/results && shasum -a 256 -c SHA256SUMS
cd "$(dirname "$0")/results" && find . -maxdepth 1 -type f | sed 's|^./||' | grep -v -E '^(SHA256SUMS|.*\.log|.*\.done|.*\.stage|.*\.pid)$' | sort | xargs shasum -a 256 > SHA256SUMS
echo "wrote $(wc -l < SHA256SUMS) checksums"
