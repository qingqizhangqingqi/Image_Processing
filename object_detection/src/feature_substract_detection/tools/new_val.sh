CONFIG=$1
MODEL=$2
GPU=$3

python tools/new_test.py --config ${CONFIG} --model ${MODEL} --run_gpu ${GPU}
python tools/voc_eval.py eval/result.pkl ${CONFIG}
