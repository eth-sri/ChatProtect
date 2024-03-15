set -e

prefix="$(dirname "$0")"

echo "self-contra. predicted"
python3 $prefix/temperature_generation_trigger.py
echo "precision"
python3 $prefix/temperature_generation_precision.py
