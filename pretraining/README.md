# Pretraining CDLM

We heavily relied on the code from the [language modeling script](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py) from the [Huggingface Transformers](https://github.com/huggingface/transformers) repository.

## Getting started - Packages and Data

* Install python3 requirements `pip install -r requirements.txt` 
* Download he Multi-News corpus from [here](https://drive.google.com/drive/folders/1qZ3zJBv0zrUy4HVWxnx33IsrHGimXLPy).
 Then, create the `multinews` directory and place there the downloaded `*.src` files.
 
## Pretraining CDLM
Just run the script `run_language_modeling.py`. We used the following command-line (using 8 GPUs):

```python
python run_language_modeling.py \
--output_dir outputs \
--logging_dir logs \
--fp16 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--num_train_epochs 1
```


Note that the effective batch-size should be 64. Keep in mind that `total_batch_size`x`gradient_accumulation_steps`=`effective_batch_size`.