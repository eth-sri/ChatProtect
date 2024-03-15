for i in 0.00 0.25 0.50 0.75; do
  echo "Temperature: $i"
  python3 pipeline/measure_results_json.py --test_dir "output/test/chatgpt/chatgpt/s3/test_m33/t$i" --prf1 --override_sent
done
echo "Temperature: 1.00"
python3 pipeline/measure_results.py --test_dir output/test/chatgpt/chatgpt/s3/test_m3 --test_sentence_dir test/test/sentences/chatgpt --prf1 --old --override_sent
