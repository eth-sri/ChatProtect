#!/usr/bin/env bash
set -e

echo "Revision step 2"
# generate sentence + alternative sentences pairs /w tag for inconsistency for the revised description (Figure 1 + 2)
python3 pipeline/1_generate_sentences.py --new m2 "$@"
# generate new descriptions based on the revised description and the tags (second step of revise, Figure 3)
python3 pipeline/2_generate_new_descriptions.py --mode 3 "$@"

echo "Revision step 3"
# generate sentence + alternative sentences pairs /w tag for inconsistency for the revised description (Figure 1 + 2)
python3 pipeline/1_generate_sentences.py --new m3 "$@"
# generate new descriptions based on the revised description and the tags (third step of revise, Figure 3)
python3 pipeline/2_generate_new_descriptions.py --mode 4 "$@"

echo "Revision step 4"
# generate sentence + alternative sentences pairs /w tag for inconsistency for the revised description (Figure 1 + 2)
python3 pipeline/1_generate_sentences.py --new m4 "$@"
# generate new descriptions based on the revised description and the tags (final step of revise, Figure 3)
python3 pipeline/2_generate_new_descriptions.py --mode 5 "$@"
