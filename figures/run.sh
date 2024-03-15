set -e

prefix="$(dirname "$0")"

echo "Main paper"
echo "----------"

echo "Table 2"
bash $prefix/table2/run.sh
echo

echo "Table 3"
bash $prefix/table3/run.sh
echo

echo "Table 4"
bash $prefix/table4/run.sh
echo

echo "Table 5"
bash $prefix/table5/run.sh
echo

echo "Appendix"
echo "----------"

echo "Figure 5"
bash $prefix/figure5/run.sh
echo

echo "Table 6"
bash $prefix/table6/run.sh
echo

echo "Table 7"
bash $prefix/table7/run.sh
echo

echo "Table 8"
bash $prefix/table8/run.sh
echo

echo "Table 9"
bash $prefix/table9/run.sh
echo

echo "Table 10"
bash $prefix/table10/run.sh
echo

echo "Table 11"
bash $prefix/table11/run.sh
echo

echo "Figure 7/8"
bash $prefix/figure78/run.sh
echo
