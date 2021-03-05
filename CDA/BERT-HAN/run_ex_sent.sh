export GLUE_DIR=MDA_data/AAN/
export TASK_NAME=cite

python run_ex_sent.py
  --model_name_or_path './CDLM' \
  --task_name $TASK_NAME \
  --do_lower_case \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --save_steps 500 \
  --logging_steps 500 \
  --overwrite_output_dir \
  --data_dir $GLUE_DIR \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 8.0 \
  --output_dir ./cite_model_sent
