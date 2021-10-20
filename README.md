# Cross-Document Language Modeling

This repository contains the accompanying code for the paper:

**"CDLM: Cross-Document Language Modeling."** Avi Caciularu, Arman Cohan, Iz Beltagy, Matthew E Peters, Arie Cattan 
and Ido Dagan. *In EMNLP Findings, 2021*.
[[PDF]](https://arxiv.org/pdf/2101.00406.pdf)


## Structure
The repository contains:
* Implementation of the CDMLM pretraining, based on the *Huggingface* code (in `pretraining` dir).
* Code for finetuning over cross-document coreference resolution (in `cross_encoder` dir).
* Code for finetuning over multi-document classification tasks (in `CDA` dir).
* Code for the attention analysis over the sampled ECB+ dataset (in `attention_analysis` dir).
* Code for finetuning over the multi-hop question answering task, using the *HotpotQA* dataset, including instructions, appears [here](https://github.com/armancohan/longformer/tree/hotpotqa).

---
## Pretrained Model Usage

Our model is available on HuggingFace: https://huggingface.co/biu-nlp/cdlm

```python
from transformers import AutoTokenizer, AutoModel
# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('biu-nlp/cdlm')
model = AutoModel.from_pretrained('biu-nlp/cdlm')
```

Please note that during our pretraining we used the document and sentence separators, which you might want to add to your data. The document and sentence separators are `<doc-s>`, `</doc-s>` (the last two tokens in the vocabulary), and `<s>`, `</s>`, respectively.



---
## Citation:
If you find our work useful, please cite the paper as:

```bibtex
@article{caciularu2021cross,
  title={Cross-Document Language Modeling},
  author={Caciularu, Avi and Cohan, Arman and Beltagy, Iz and Peters, Matthew E and Cattan, Arie and Dagan, Ido},
  journal={Findings of the Association for Computational Linguistics: EMNLP 2021},
  year={2021}
}
```
