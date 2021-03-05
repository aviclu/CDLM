export GLUE_DIR=data/cite_acl
export TASK_NAME=cite

python run_ex_sent.py \
  --model_type bert-han \
  --sub_model_type han-sg \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_lower_case \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --save_steps 13324 \
  --logging_steps 13324 \
  --overwrite_output_dir \
  --data_dir $GLUE_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size 2 \
  --learning_rate 2e-5 \
  --num_train_epochs 8.0 \
  --output_dir ./cite_model_sg_sent_wlinear
