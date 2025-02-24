nvidia-smi

model_name=opt-125m
percentage=0.05 # {0.025, 0.05, 0.1, 0.25, 0.50, 1.0}%
number_of_gpus=1
data_seed=3 # Seed for shuffling data: 3, 6, 9, ...
validation_task=mmlu
include_validation=false
pipeline_parallel=false
batch_size_per_device=1
method=random
data_shuffle=true 
learning_rate=2e-05
num_train_epochs=4
save_steps_per_epoch=1
val_only=false
accelerate_launch=false
out_dir=out_7b_mmlu
deepspeed_config=zero1


for learning_rate in 4e-05; do
    for percentage in 0.025 0.05; do
        for data_seed in 3; do

            #* Training 
            # export CUDA_VISIBLE_DEVICES=0,1 # changed to 0,1 bc only 2 GPUs

            cd /users/inguyen4/code/task-difficulty/ccds/CCDS && \
            conda run -p /users/inguyen4/.conda/envs/ccds \
            bash ccds_scripts/train_random.sh "$percentage" "$number_of_gpus" "$data_seed" "$validation_task" \
            "$include_validation" "$pipeline_parallel" "$batch_size_per_device" \
            "$method" "$data_shuffle" "$learning_rate" "$num_train_epochs" \
            "$save_steps_per_epoch" "$val_only" "$model_name" "$accelerate_launch" "$out_dir" \
            "$deepspeed_config"

            #* Evaluation
            export CUDA_VISIBLE_DEVICES=0

            model_path="/users/inguyen4/code/task-difficulty/ccds/CCDS/${out_dir}/${model_name}-{$validation_task}-{$method}-p${percentage}-lora-seed${data_seed}-lr${learning_rate}-${num_train_epochs}epoch-$number_of_gpus"
            cd /users/inguyen4/code/task-difficulty/ccds/CCDS/ccds/evaluation && \
            conda run -p /users/inguyen4/.conda/envs/ccds \
            bash eval_scripts/run_eval_mmlu.sh $model_path
        
        done
    done
done


for learning_rate in 2e-05; do
    for percentage in 0.10 0.25 0.50; do
        for data_seed in 3; do

            #* Training 
            export CUDA_VISIBLE_DEVICES=0,1,2,3

            cd /users/inguyen4/code/task-difficulty/ccds/CCDS && \
            conda run -p /users/inguyen4/.conda/envs/ccds \
            bash ccds_scripts/train_random.sh "$percentage" "$number_of_gpus" "$data_seed" "$validation_task" \
            "$include_validation" "$pipeline_parallel" "$batch_size_per_device" \
            "$method" "$data_shuffle" "$learning_rate" "$num_train_epochs" \
            "$save_steps_per_epoch" "$val_only" "$model_name" "$accelerate_launch" "$out_dir" \
            "$deepspeed_config"

            #* Evaluation
            export CUDA_VISIBLE_DEVICES=0

            model_path="/users/inguyen4/code/task-difficulty/ccds/CCDS/${out_dir}/${model_name}-{$validation_task}-{$method}-p${percentage}-lora-seed${data_seed}-lr${learning_rate}-${num_train_epochs}epoch-$number_of_gpus"
            cd /users/inguyen4/code/task-difficulty/ccds/CCDS/ccds/evaluation && \
            conda run -p /users/inguyen4/.conda/envs/ccds \
            bash eval_scripts/run_eval_mmlu.sh $model_path
        
        done
    done
done