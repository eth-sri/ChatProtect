set -e

prefix="$(dirname "$0")"

echo "self-contra. reduced"
python3 $prefix/sc_mitigate.py
echo "informative facts retained"
python3 $prefix/informativeness.py
echo "perplexity increased"
echo "uncomment to compute (slow)"
# python3 $prefix/perplexity.py
