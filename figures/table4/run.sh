set -e

prefix="$(dirname "$0")"

echo "P / R / F1"
python3 $prefix/detection_prf1.py
echo "self-contra. reduced"
python3 $prefix/sc_mitigate.py
echo "informative facts retained"
python3 $prefix/informativeness.py
