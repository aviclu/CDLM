# Document classification tasks
We heavily relied on the code of the EMNLP2020 paper: [Multilevel Text Alignment with Cross-Document Attention](https://arxiv.org/abs/2010.01263), which is forked from [the official repository](https://github.com/XuhuiZhou/CDA).


Please follow the instructions from the repository link, and download all the requested datasets from [here](https://xuhuizhou.github.io/Multilevel-Text-Alignment/).

### Working with other versions of packages:
Please use a `transformers` version that supports the Longformer model (we used `4.1.1`). 

### Finetuning CDLM on tasks
We used 8 Quadro RTX 8000 GPUs (32GB each).
The following command will finetune our CDLM on the AAN corpus. Please modify the relative path accordingly to run the .sh script.
```bash
./BERT-HAN/run_ex_sent.sh
```

Note: Make sure to place the tokenizer files inside the model directory.