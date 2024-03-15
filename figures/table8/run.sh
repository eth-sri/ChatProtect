set -e

prefix="$(dirname "$0")"

echo "self-contra. generated"
python3 $prefix/sc_trigger.py
