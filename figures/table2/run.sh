set -e

prefix="$(dirname "$0")"

echo "avg number of sentences"
python3 $prefix/sentences_generated.py
echo "perplexity"
if [ "1" = "$MEASURE_PPL"  ]; then
  python3 $prefix/perplexity.py
else
  echo "skipping (run \`export MEASURE_PPL=1\` to compute)"
fi
echo "self-contra. generated"
python3 $prefix/sc_trigger.py
