# Shared-private Memory Networks for Multimodal Sentiment Analysis (SPMN)

<img src="https://github.com/xiaobaicaihhh/SPAMN/blob/main/img/model.png" width="800px" div align=center />
## Getting started

1. Dependences
   `pip install transformers=4.15`
   `pip install torch=1.10`
   `cuda version=10.2`

2. Download datasets
   Inside `./datasets` folder, run `./download_datasets.sh` to download MOSI and MOSEI datasets

3. Training SPAMN on MOSI
   **Training scripts:**

   - SPAMN `python main.py --model [pretrained model]`
   - text pretrained model:bert, xlnet, albert, roberta, electra
   - speech pretrained model: wavlm


4. Model usage

   We would like to thank [huggingface](https://huggingface.co/) and [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer) for providing and open-sourcing transformer-based pretrained model code for developing our models.

