Title: Project Examples for Deep Structured Learning (Fall 2018)

We suggest below some project ideas. Feel free to use this as inspiration for your project. Talk to us for more details.

---

# Sparse Classification with Sparsemax

- **Problem:** Apply sparsemax and/or sparsemax loss to a problem that requires outputting sparse label probabilities or sparse latent variables (attention).
- **Data:** [Multi-label datasets](http://mulan.sourceforge.net/datasets-mlc.html), [SNLI](https://nlp.stanford.edu/projects/snli/), [WMT](http://www.statmt.org/wmt18/translation-task.html), any data containing many labels for which only a few are plausible for each example.
- **Evaluation:** F1, accuracy, inspection of where the model learns to attend to.
- **References:**

    1. [Martins and Astudillo. From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification. ICML 2016.](http://proceedings.mlr.press/v48/martins16.html)

---

# Deep Generative Models for Discrete Data

- **Problem:** Compare different deep generative models' ability to generate discrete data (such as text).
- **Methods:** Generative Adversarial Networks, Variational Auto-Encoders.
- **Data:** [SNLI](https://nlp.stanford.edu/projects/snli/) (just the text), [Yelp/Yahoo datasets for unaligned sentiment/topic transfer](https://www.yelp.com/dataset/challenge), other text data.
- **Evaluation:** Some of the metrics in [4].
- **References:**

    1. [Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville, Bengio. Generative Adversarial Networks. NIPS 2014.](https://arxiv.org/abs/1406.2661)  
    2. [Kingma and Wellington. Auto-Encoding Variational Bayes. NIPS 2013.](https://arxiv.org/pdf/1312.6114.pdf)  
    3. [Zhao, Kim, Zhang, Rush, LeCun. Adversarially Regularized Autoencoders. ICML 2018.](http://proceedings.mlr.press/v80/zhao18b/zhao18b.pdf)  
    4. [Semeniuta, Severyn, Gelly. On Accurate Evaluation of GANs for Language Generation. 2018.](https://arxiv.org/abs/1806.04936)

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

# Geometrical structure: embedding ellipses instead of points

- **Problem:** Go beyond vector (point) embeddings: embed objects as ellipses instead of points; capture notions of inclusion/overlap.
- **References:**

    1. [Generalizing Point Embeddings using the Wasserstein Space of Elliptical Distributions. Muzellec & Cuturi. 2018.](https://arxiv.org/abs/1805.07594)

---

# Embed structured input data with graph-CNNs

- **Problem:** learn good fixed-size hidden representations for data that comes in graph format with different shapes and sizes.
- **Method:** [Graph convolutional networks](http://tkipf.github.io/graph-convolutional-networks/)
- **Data:** [arXiv macro usage](https://github.com/CornellNLP/Macros), [annotated semantic relationships datasets](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets), [paralex](http://knowitall.cs.washington.edu/paralex/)
- **References:**

    1. [Kipf and Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.](https://openreview.net/pdf?id=SJU4ayYgl)  
    2. [Defferrard et al. Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. NIPS 2016.](https://arxiv.org/abs/1606.09375)

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
