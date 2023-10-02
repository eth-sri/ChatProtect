set -e

prefix="$(dirname "$0")"

echo "avg number of sentences"
python3 $prefix/sentences_generated.py
echo "perplexity"
echo "uncomment to compute (slow)"
# python3 $prefix/perplexity.py
echo "self-contra. generated"
python3 $prefix/sc_trigger.py
