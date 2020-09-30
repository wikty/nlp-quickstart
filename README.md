# Language Model

单词级 n-gram 前向神经网络语言模型 [A Neural Probabilistic Language Model](http://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (Bengio et al., 2001; 2003)

字符级 RNN 语言模型介绍 [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

字符级 n-gram 语言模型跟 RNN 对比 [The unreasonable effectiveness of Character-level Language Models](https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139)

[好玩的文本生成](https://www.msra.cn/zh-cn/news/features/ruihua-song-20161226)

# Word Embedding

review:

- https://ruder.io/word-embeddings-1/
- https://ruder.io/word-embeddings-softmax/index.html

word2vec tutorial:

- [skip-gram](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) by Chris McCormick
- [negative sample](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) by Chris McCormick

word2vec paper:

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)

embedding for downstream tasks:

- [A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning](https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf)(Collobert and Weston 2008)

- [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398)
- http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/

distributional semantic model:

- [From frequency to meaning: Vector space models of semantics](https://www.aaai.org/Papers/JAIR/Vol37/JAIR-3705.pdf)
- [A Brief History of Word Embeddings](https://www.gavagai.io/text-analytics/a-brief-history-of-word-embeddings/)
- https://rare-technologies.com/making-sense-of-word2vec/
- https://ruder.io/secret-word2vec/index.html
- [Glove: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://transacl.org/ojs/index.php/tacl/article/viewFile/570/124)

sentence embedding:

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

[A Convolutional Neural Network for Modelling Sentences](https://www.aclweb.org/anthology/P14-1062)

[paragraph vector](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

[Skip-Thought Vectors](https://arxiv.org/abs/1506.06726)

[Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432)

[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364)

https://github.com/Separius/awesome-sentence-embedding

https://www.dataiku.com/product/plugins/sentence-embedding/

[Are distributional representations ready for the real world?](https://arxiv.org/abs/1705.11168)

more aboue embedding:

[Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606v1.pdf)

https://ruder.io/word-embeddings-2017/

[A Survey of Cross-lingual Word Embedding Models](https://arxiv.org/abs/1706.04902)

visual embedding:

- https://distill.pub/2016/misread-tsne/

# Machine Learning

[Linear Regression](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/)

[Logistic Regression](http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/)

[Softmax Regression](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)

https://cs231n.github.io/linear-classify/

[Linear model by Andrew Ng](http://cs229.stanford.edu/notes/cs229-notes1.pdf)

[An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/)



# Neural Network

神经网络入门介绍 [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Nielsen

RNN:

- LSTM 介绍 [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Colah
- https://distill.pub/2016/augmented-rnns/

CNN:

- http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
- http://colah.github.io/posts/2014-07-Understanding-Convolutions/
- http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
- [Dilated CNN](https://arxiv.org/abs/1610.10099)

RecNN:

- [RecNN](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)

# Seq2Seq

From SMT to NMT

- [Statistical phrase-based translation](https://www.aclweb.org/anthology/N03-1017/)
- [Recurrent continuous translation models](https://www.aclweb.org/anthology/D13-1176.pdf)
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [On the properties of neural machine translation: Encoder–Decoder approaches](https://www.aclweb.org/anthology/W14-4012.pdf)

Attention

- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473v7)
- [Effective Approaches to Attention-based Neural Machine Translation](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
- https://distill.pub/2016/augmented-rnns/

Transformer:

- Transformer 介绍 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- Transformer 论文 [Attention Is All You Need]()
- Transformer [Pytorch 实现](http://nlp.seas.harvard.edu/2018/04/03/attention.html)



# Natural Language Processing

基于 NLTK 库的自然语言处理实践教程 [The NLTK Book](http://www.nltk.org/book/)

基于神经网络的自然语言处理方法的历史演进 [A Review of the Neural History of Natural Language Processing](http://ruder.io/a-review-of-the-recent-history-of-nlp/) by Sebastian Ruder

基于神经网络的自然语言处理常用方法简介 [A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf) by Yoav Goldberg

基于深度学习的自然语言处理技术最佳实践 [Deep Learning for NLP Best Practices](http://ruder.io/deep-learning-nlp-best-practices/index.html#attentionhttp://ruder.io/deep-learning-nlp-best-practices/index.html)

[基于 Pytorch 深度学习和自然语言处理入门](https://nlp-pt.apachecn.org/)



# Pretrained language models

[NLP's ImageNet moment has arrived](http://ruder.io/nlp-imagenet/)

[The Illustrated BERT, ELMo, and others](http://jalammar.github.io/illustrated-bert/)

[Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432)

[ELMo](https://arxiv.org/abs/1802.05365)

[ULMFiT](https://arxiv.org/abs/1801.06146)

[OpenAI-Transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

[BERT](https://arxiv.org/abs/1810.04805)

[BERT Word Embeddings](http://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)

http://ruder.io/multi-task/

[auxiliary task](http://ruder.io/multi-task-learning-nlp/)

https://github.com/huggingface/transformers

https://github.com/hanxiao/bert-as-service

# Text Classification

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

[A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820)

http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

https://github.com/yoonkim/CNN_sentence

http://albertxiebnu.github.io/fasttext/

# Text embedding search

https://github.com/facebookresearch/faiss

https://engineering.fb.com/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

https://www.elastic.co/blog/text-similarity-search-with-vectors-in-elasticsearch

https://hanxiao.io/2019/11/22/Video-Semantic-Search-in-Large-Scale-using-GNES-and-TF-2-0/

# Tools

fasttext: https://github.com/facebookresearch/fastText

textcnn: https://github.com/dennybritz/cnn-text-classification-tf



# Resources

[深度学习 500 问](https://github.com/scutan90/DeepLearning-500-questions)

[中文自然语言处理相关资料](https://github.com/crownpku/awesome-chinese-nlp)

[中文自然语言处理各任务最新进展](https://chinesenlp.xyz/#/zh/) by 滴滴人工智能实验室

http://ruder.io/

http://www.hankcs.com/

[liuhuanyong 常见自然语言处理任务项目](https://liuhuanyong.github.io/)