CONFIG=$1
MODEL=$2
GPU=$3

python tools/test.py ${CONFIG} ${MODEL} --gpu=${GPU} --out=eval/result.pkl 
python tools/voc_eval.py eval/result.pkl ${CONFIG}
