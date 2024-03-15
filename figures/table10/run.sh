set -e

prefix="$(dirname "$0")"

echo "self-contra. reduced"
python3 $prefix/sc_mitigate.py
echo "informative facts retained"
python3 $prefix/informativeness.py
echo "perplexity"
if [ "1" = "$MEASURE_PPL"  ]; then
  python3 $prefix/perplexity.py
else
  echo "skipping (run \`export MEASURE_PPL=1\` to compute)"
fi
