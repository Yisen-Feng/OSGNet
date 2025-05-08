#### training
config_file=$1
resume_path=$2
exp_id=$3
device_id=$4
mode=$5
echo ${device_id}
NUM_GPUS=$(echo $device_id | tr ',' ' ' | wc -w)
CUDA_VISIBLE_DEVICES=${device_id} torchrun --standalone --nproc_per_node=${NUM_GPUS} --master_port=55557 train.py ${config_file} \
--output ${exp_id} --resume ${resume_path} --mode=${mode}\
${@:6}