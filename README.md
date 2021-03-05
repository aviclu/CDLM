# Cross-Document Language Modeling

This repository contains the accompanying code for the paper:

**"Cross-Document Language Modeling."** Avi Caciularu, Arman Cohan, Iz Beltagy, Matthew E Peters, Arie Cattan 
and Ido Dagan. *Appears on arXiv*.
[[PDF]](https://arxiv.org/pdf/2101.00406.pdf)


## Structure
The repository contains:
* Implementation of the CDMLM pretraining, based on the *Huggingface* code (in `pretraining` dir).
* Code for finetuning over cross-document coreference resolution (in `cross_encoder` dir).
* Code for finetuning over multi-document classification tasks (in `CDA` dir).
* Note that we executed experiments also over the multi-hop question answering task, using the *HotpotQA* dataset. 
The code that we used for this simulation, including the rest of the instractions, appears [here](https://github.com/armancohan/longformer/tree/hotpotqa).

---
## Pretrained Model Usage

You can either pretrain or download the pretrained CDLM model weights and tokenizer files, which are available [here](https://drive.google.com/drive/folders/1txXAZbt-C53FcgtbL9DNvUCacZV9xxdC?usp=sharing). 

Then, create the directory `CDLM` and place there all the weights and tokenizer files. For loading the model and tokenizer, use
```python
from transformers import AutoTokenizer, AutoModel
# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('CDLM')
model = AutoModel.from_pretrained('CDLM')
```

Please note that during our pretraining we used the document and sentence separators, which you might want to add to your data. The document and sentence separators are `<doc-s>`, `</doc-s>` (the last two tokens in the vocabulary), and `<s>`, `</s>`, respectively.



---
## Citation:
If you find our work useful, please cite the paper as:

```bibtex
@article{caciularu2021cross,
  title={Cross-Document Language Modeling},
  author={Caciularu, Avi and Cohan, Arman and Beltagy, Iz and Peters, Matthew E and Cattan, Arie and Dagan, Ido},
  journal={arXiv preprint arXiv:2101.00406},
  year={2021}
}
```