echo "ChatGPT / ChatGPT"
python3 pipeline/measure_results.py --test_dir output/test/chatgpt/chatgpt/s3/test_m3 --test_sentence_dir test/test/sentences/chatgpt --prf1 --old --override_sent
echo "ChatGPT / GPT-4"
python3 pipeline/measure_results.py --test_dir output/test/gpt4/chatgpt/s3/test_m3 --test_sentence_dir test/test/sentences/chatgpt --prf1 --old --override_sent
echo "GPT-4 / ChatGPT"
python3 pipeline/measure_results.py --test_dir output/test/chatgpt/gpt4/s3/test_m3 --test_sentence_dir test/test/sentences/gpt4 --prf1 --old --override_sent
echo "GPT-4 / GPT-4"
python3 pipeline/measure_results_json.py --test_dir output/test/gpt4/gpt4/s3/test_m3 --prf1 --override_sent
