# Multimodal Residual Learning for Visual QA (NIPS 2016)

Multimodal residual networks three-block layered model. GRUs initialized with Skip-Thought Vectors for question embedding and ResNet-152 for extracting visual feature vectors are used. Joint representations are learned by element-wise multiplication, which leads to implicit attentional model without attentional parameters. 

This current code can get **61.84** on Open-Ended and **66.33** on Multiple-Choice on **test-standard** split.

Notice that this code is based on Lu et al (2015)'s [VQA_LSTM_CNN](https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/readme.md). Also, you need to use this base code for preprocessing.

Our latest work can be found in [Hadamard Product for Low-rank Bilinear Pooling](https://arxiv.org/abs/1610.04325), which is the state-of-the-art (Single: *65.07/68.89*, Ensemble: *66.89/70.29* for *test-standard*) as of Dec 1st 2016. The code for this will be released in [Github](https://github.com/jnhwkim/MulLowBiVQA).

### Dependencies

* [rnn](https://github.com/Element-Research/rnn)

You can install the dependencies:

```bash
luarocks install rnn
```

### Training

Please follow the instruction from [VQA_LSTM_CNN](https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/readme.md) for preprocessing. `--split 2` option allows to use train+val set to train, and test-dev or test-standard set to evaluate. Set `--num_ans` to `2000` to reproduce the result.

For question features, you need to use this:

* [skip-thoughts](https://github.com/ryankiros/skip-thoughts)
* [DPPnet](https://github.com/HyeonwooNoh/DPPnet) (see 003_skipthoughts_porting)
* `make_lookuptable.lua`

for image features,

```
$ th prepro_res.lua -input_json data_train-val_test-dev_2k/data_prepro.json -image_root path_to_image_root -cnn_model path to cnn_model
```

The pretrained ResNet-152 model and related scripts can be found in [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua).

```
$ th train_residual.lua
``` 

With the default parameter, this will take around twenty hours on a sinlge NVIDIA Titan X GPU, and will generate the model under `model/`. 

Notice that for the exact reproduction, ResNet-152 features by Caffe are needed. 

### Evaluation

```
$ th eval_residual.lua
```

In evaluation, you can use generated image captions to improve accuracies (for test-dev; overall +0.08%, others +0.17%) with option `-priming` (default=false). We used [NeuralTalk2](https://github.com/karpathy/neuraltalk2) to generate `captions_test2015.json`. This is only used for evaluation.

### References

If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:

```
@inproceedings{kim2016b,
author = {Kim, Jin-Hwa and Lee, Sang-Woo and Kwak, Donghyun and Heo, Min-Oh and Kim, Jeonghee and Ha, Jung-Woo and Zhang, Byoung-Tak},
booktitle = {Advances In Neural Information Processing Systems 29},
pages = {361--369},
title = {{Multimodal Residual Learning for Visual QA}},
year = {2016}
}
```

This code uses Torch7 `rnn` package and its `TrimZero` module for question embeddings. Notice that following papers:

```
@article{Leonard2015a,
author = {L{\'{e}}onard, Nicholas and Waghmare, Sagar and Wang, Yang and Kim, Jin-Hwa},
journal = {arXiv preprint arXiv:1511.07889},
title = {{rnn : Recurrent Library for Torch}},
year = {2015}
}
@inproceedings{Kim2016a,
author = {Kim, Jin-Hwa and Kim, Jeonghee and Ha, Jung-Woo and Zhang, Byoung-Tak},
booktitle = {Proceedings of KIIS Spring Conference},
isbn = {2093-4025},
number = {1},
pages = {165--166},
title = {{TrimZero: A Torch Recurrent Module for Efficient Natural Language Processing}},
volume = {26},
year = {2016}
}
```

### License

BSD 3-Clause License.

### Patent (Pending)

METHOD AND SYSTEM FOR PROCESSING DATA USING ELEMENT-WISE MULTIPLICATION AND MULTIMODAL RESIDUAL LEARNING FOR VISUAL QUESTION-ANSWERING
