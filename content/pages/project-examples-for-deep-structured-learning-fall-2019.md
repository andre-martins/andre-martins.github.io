Title: Project Examples for Deep Structured Learning (Fall 2018)

We suggest below some project ideas. Feel free to use this as inspiration for your project. Talk to us for more details.

---

# Emergent Communication

- **Problem:** Agents need to communicate to solve problems that require collaboration. The goal of this project is to apply techniques (for example using sparsemax or reinforcement learning) to induce communication among agents. 
- **Data:** See references below.
- **Evaluation:** See references below.
- **References:**

    1. [Serhii Havrylov and Ivan Titov. Emergence of language with multi-agent games: Learning to communicate with sequences of symbols. NeurIPS 2017.](https://papers.nips.cc/paper/6810-emergence-of-language-with-multi-agent-games-learning-to-communicate-with-sequences-of-symbols.pdf)
    2. [Kris Cao, Angeliki Lazaridou, Marc Lanctot, Joel Z Leibo, Karl Tuyls, Stephen Clark. Emergent Communication through Negotiation. In ICLR 2018.](https://openreview.net/pdf?id=Hk6WhagRW)
    3. [Satwik Kottur, José Moura, Stefan Lee, Dhruv Batra. Natural Language Does Not Emerge Naturally in Multi-Agent Dialog. In EMNLP 2017.](https://www.aclweb.org/anthology/D17-1321.pdf)

---

# Explainability of Neural Networks

- **Problem:** Neural networks are black boxes and not amenable to interpretation. The goal of this project is to develop and study methods that lead to explainability of neural network model's predictons (for example using sparsemax attention).
-- **Method:** For example, sparse attention, gradient-based measures of feature importance, LIME (see below).
-- **Data:** Stanford Sentiment Treebank, IMDB Large Movie Reviews Corpus, etc. See references below.
-- **References:**

    1. [Marco T Ribeiro, Sammer Singh, and Carlos Guestrin. Why Should I Trust You? Explaining the Predictions of Any Classifier. KDD 2016.](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)
    2. [Sarthak Jain and Byron C Wallace. Attention is not explanation. NAACL 2019.](https://www.aclweb.org/anthology/N19-1357.pdf)
    3. [Zachary  C  Lipton. The  mythos  of  model  interpretability. ICML 2016 Workshop on Human Interpretability in Machine Learning.](https://arxiv.org/pdf/1606.03490.pdf)
    4. [Sofia Serrano and Noah A Smith. Is attention interpretable? ACL 2019.](https://www.aclweb.org/anthology/P19-1282.pdf)
    5. [Sarah  Wiegreffe  and  Yuval  Pinter. Attention  is  not  not  explanation. EMNLP 2019.](https://arxiv.org/pdf/1908.04626.pdf)

---

# Generative Adversarial Networks for Discrete Data

- **Problem:** Compare different deep generative models' ability to generate discrete data (such as text).
- **Methods:** Generative Adversarial Networks.
- **Data:** [SNLI](https://nlp.stanford.edu/projects/snli/) (just the text), [Yelp/Yahoo datasets for unaligned sentiment/topic transfer](https://www.yelp.com/dataset/challenge), other text data.
- **Evaluation:** Some of the metrics in [4].
- **References:**

    1. [Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville, Bengio. Generative Adversarial Networks. NIPS 2014.](https://arxiv.org/abs/1406.2661)  
    2. [Zhao, Kim, Zhang, Rush, LeCun. Adversarially Regularized Autoencoders. ICML 2018.](http://proceedings.mlr.press/v80/zhao18b/zhao18b.pdf)  
    3. [Semeniuta, Severyn, Gelly. On Accurate Evaluation of GANs for Language Generation. 2018.](https://arxiv.org/abs/1806.04936)

---

# Object detection with weak supervision

- **Problem:** Given images with object tags, can we accurately predict object location by training only with coarse, weak supervision about which objects are present in the data?
- **Method:** Latent Potts model; Belief Propagation
- **Data:** COCO-Stuff, Pascal VOC 
- **References:** See https://github.com/kazuto1011/deeplab-pytorch

---

# Sparse transformers

- **Problem:** Transformers and BERT models are extremely large and expensive to train and keep in memory. The goal of this project is to distill or induce sparser and smaller Transformer models without losing accuracy, applying them to machine translation or language modeling.
- **Method:** See references below.
- **Data:** WMT datasets, WikiText, etc. See references below.
- **References:**

    1. [Gonçalo M. Correia, Vlad Niculae, André F.T. Martins. Adaptively Sparse Transformers. EMNLP 2019.](https://arxiv.org/pdf/1909.00015.pdf)
    2. [Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, Armand Joulin. Adaptive Attention Span in Transformers. ACL 2019.](https://www.aclweb.org/anthology/P19-1032.pdf)
    3. [Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever. Generating Long Sequences with Sparse Transformers. Arxiv 2019.](https://arxiv.org/pdf/1904.10509.pdf)
    4. [Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. TinyBERT: Distilling BERT for Natural Language Understanding. Arxiv 2019.](https://arxiv.org/pdf/1909.10351.pdf)

---

# Contextual Probabilistic Embeddings / Language Modeling

-- **Problem:** Embedding words as vectors (aka point masses) cannot distinguish between more vague or more specific concepts. One solution is to embed words as a mean vector μ and a covariance Σ. Muzellec & Cuturi have a nice framework for this, tested for learning non-contextualized embeddings. Can we extend it to contextualized embeddings via language modelling? E.g. a model that reads an entire sentence and predicts a context-dependent pair (μ, Σ) for each word (perhaps left-to-right or masked). What likelihood to use? How can we evaluate the learned embeddings downstream?
-- **Method:** See reference below.
-- **References:**

    1. [Boris Muzellec, Marco Cuturi. Generalizing Point Embeddings using the Wasserstein Space of Elliptical Distributions. Arxiv 2018.](https://arxiv.org/abs/1805.07594)

---

# Constrained Structured Classification with AD3
- **Problem:** Use AD3 and dual decomposition techniques to impose logic/budget/structured constraints in structured problems. Possible tasks could involve generating diverse output, forbidding certain configurations, etc.
- **Data:**  
    - ["weasel words": detecting hedges/uncertainty in writing in order to improve clarity](http://rgai.inf.u-szeged.hu/conll2010st/download.html)  
    - [Coreference in quizbowl](https://www.cs.umd.edu/~aguha/qbcoreference)
- **Evaluation:** task-dependent
- **References:**

    1. [Martins, Figueiredo, Aguiar, Smith, Xing. AD3: Alternating Directions Dual Decomposition for MAP Inference in Graphical Models. JMLR 2015.](http://jmlr.org/papers/volume16/martins15a/martins15a.pdf)  
    2. [Niculae, Park, Cardie. Argument Mining with Structured SVMs and RNNs. ACL 2017.](http://aclweb.org/anthology/P17-1091)
    3. [AD3 toolkit.](https://github.com/andre-martins/AD3)  

---

# Structured multi-label classification

- **Problem:** Multi-label classification is a learning setting where every sample can be assigned zero, one or more labels.
- **Method:** Correlations between labels can be exploited by learning an affinity matrix of label correlation. Inference in a fully-connected correlation graph is hard; approximating the graph by a tree makes inference fast (Viterbi can be used.)
- **Data:**  [Multi-label datasets](http://mulan.sourceforge.net/datasets-mlc.html)
- **Evaluation:** see [here](http://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics)
- **References:**

    1. [Sorower. A Literature Survey on Algorithms for Multi-label Learning.](https://pdfs.semanticscholar.org/6b56/91db1e3a79af5e3c136d2dd322016a687a0b.pdf)  
    2. [Thomas Finley and Thorsten Joachims. 2008. Training structural SVMs when exact inference is intractable.](https://www.cs.cornell.edu/people/tj/publications/finley_joachims_08a.pdf)  
    3. [Pystruct](https://pystruct.github.io/user_guide.html#multi-label-svm)  
    4. [Scikit.ml](http://scikit.ml/) (very strong methods not based on structured prediction)

---

# Hierarchical sparsemax attention

- **Problem:** Performing neural attention over very long sequences (e.g. for document-level classification, translation, ...)
- **Method:** sparse hierarchical attention with product of sparsemaxes.
- **Data:** [text classification datasets](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
- **Evaluation:** Accuracy; empirical analysis of where the models attend to.
- **Notes:** If the top-level sparsemax gives zero probability to some paragraphs, those can be pruned from the computation graph. Can this lead to speedups?
- **References:**

    1. [Yang, Yang, Dyer, He, Smola, Hovy. Hierarchical Attention Networks for Document Classification. NAACL 2016.](http://www.aclweb.org/anthology/N16-1174)  
    2. [Martins and Astudillo. From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification. ICML 2016.](http://proceedings.mlr.press/v48/martins16.html)

---

# Sparse group lasso attention mechanism

- **Problem:** For structured data segmented into given "groups" (e.g. fields in a form, regions in an image, sentences in a paragraph), design a "group-sparse" attention mechanism that tends to give zero weight to entire groups when deemed not relevant enough.
- **Method:** a Sparse Group-Lasso penalty in a generalized structured attention framework [2]
- **Notes:** the L1 term is redundant when optimizing over the simplex; regular group lasso will be sparse!
- **References:**

    1. [A note on the group lasso and a sparse group lasso. J. Friedman, T. Hastie, R. Tibshirani. 2010.](https://arxiv.org/abs/1001.0736)  
    2. [A Regularized Framework for Sparse and Structured Neural Attention. Vlad Niculae, Mathieu Blondel. NIPS 2017.](https://papers.nips.cc/paper/6926-a-regularized-framework-for-sparse-and-structured-neural-attention.pdf)  
    3. [Model selection and estimation in regression with grouped variables. Yuan and Lin. 2006.](https://www.stat.wisc.edu/~myuan/papers/glasso.final.pdf)

---

# Sparse link prediction

- **Problem:** Predicting links in a large structured graph. For instance: predict co-authorship, movie recommendation, coreference resolution, discourse relations between sentences in a document.
- **Method:** The simplest approach is independent binary classification: for every node pair (i, j), predict whether there is a link or not. Issues: Very high imbalance: most nodes are not linked.
Structure and higher-order correlations are ignored in independent approach. Develop a method that can address the issues: incorporate structural correlations (e.g. with combinatorial inference, constraints, latent variables) and account for imbalance (ideally via pairwise ranking losses: learn a scorer such that S(i, j) > S(k, l) if there is an edge (i, j) but no edge (k, l).
- **Data:** [arXiv macro usage](https://github.com/CornellNLP/Macros), [Coreference in quizbowl](https://www.cs.umd.edu/~aguha/qbcoreference)
- **Notes:** Can graph-CNNs (previous idea) be useful here?
- **References:**

    1. [Rosenfeld, Meshi, Tarlow, Globerson. Learning Structured Models with the AUC Loss and Its Generalizations. AISTATS 2014.](http://ttic.uchicago.edu/~meshi/papers/structAUC_aistats14.pdf)

---

# Structured Prediction Energy Networks

- **Problem:** Structured output prediction with energy networks: replace discrete structured inference with continuous optimization in a neural net. Applications: multi-label classification; simple structured problems: sequence tagging, arc-factored parsing?
- **Method:** Learn a neural network E(x, y; w) to model the energy of an output configuration y (relaxed to be a continuous variable). Inference becomes min_y E(x, y; w). How far can this relaxation take us? Can it be better/faster than global combinatorial optimization approaches?
- **Data:** Sequence tagging, parsing, optimal matching?
- **Notes:** When E is a neural network, min_y E(x, y; w) is a non-convex optimization problem (possibly with mild constraints such as y in [0, 1]. Amos et al. have an approach that allows E to be a complicated neural net but remain convex in y. Is this beneficial? Are some kinds of structured data better suited for SPENs than others? E.g. sequence labelling seems "less structured" than dependency parsing.
- **References:**

    1. [Belanger and McCallum. Structured Prediction Energy Networks. ICML 2016.](http://proceedings.mlr.press/v48/belanger16.pdf)  
    2. [Belager, Yang, McCallum. End-to-End Learning for Structured Prediction Energy Networks. ICML 2017.](https://arxiv.org/abs/1703.05667)  
    3. [Amos, Xu, Kolter. Input Convex Neural Networks. ICML 2017](http://proceedings.mlr.press/v70/amos17b/amos17b.pdf)
