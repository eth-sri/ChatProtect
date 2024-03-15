set -e

prefix="$(dirname "$0")"

echo "P / R / F1"
python3 $prefix/detection_prf1.py
