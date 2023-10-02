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

echo "Appendix"
echo "----------"

echo "Figure 6"
bash $prefix/figure6/run.sh
echo

echo "Table 5"
bash $prefix/table5/run.sh
echo

echo "Table 6"
bash $prefix/table6/run.sh
echo

echo "Table 7"
bash $prefix/table7/run.sh
echo

echo "Figure 7/8"
bash $prefix/figure78/run.sh
echo
