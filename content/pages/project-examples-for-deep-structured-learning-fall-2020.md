Title: Project Examples for Deep Structured Learning (Fall 2019)

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

- **Problem:** Neural networks are black boxes and not amenable to interpretation. The goal of this project is to develop and study methods that lead to explainability of neural network model's predictons (for example using sparsemax attention). This project can be either a survey about recent work in this area or it can explore some practical applications. 
- **Method:** For example, sparse attention, rationalizers, gradient-based measures of feature importance, LIME, influence functions, etc.
- **Data:** BEER dataset, Stanford Sentiment Treebank, IMDB Large Movie Reviews Corpus, etc. See references below.
- **References:**

    1. [Marco T Ribeiro, Sammer Singh, and Carlos Guestrin. Why Should I Trust You? Explaining the Predictions of Any Classifier. KDD 2016.](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf). 
    2. [Tao Lei, Regina Barzilay and Tommi Jaakkola. Rationalizing Neural Predictions. EMNLP 2016.(https://people.csail.mit.edu/taolei/papers/emnlp16_rationale.pdf). 
    3. [Marcos V. Treviso, André F. T. Martins. Towards Prediction Explainability through Sparse Communication. Blackbox Workshop 2020.](https://arxiv.org/abs/2004.13876). 
    4. [Zachary  C  Lipton. The  mythos  of  model  interpretability. ICML 2016 Workshop on Human Interpretability in Machine Learning.](https://arxiv.org/pdf/1606.03490.pdf)  
    5. [Sarah  Wiegreffe  and  Yuval  Pinter. Attention  is  not  not  explanation. EMNLP 2019.](https://arxiv.org/pdf/1908.04626.pdf). 

---

# Generative Adversarial Networks for Discrete Data

- **Problem:** Compare different deep generative models' ability to generate discrete data (such as text).
- **Methods:** Generative Adversarial Networks.
- **Data:** [SNLI](https://nlp.stanford.edu/projects/snli/) (just the text), [Yelp/Yahoo datasets for unaligned sentiment/topic transfer](https://www.yelp.com/dataset/challenge), other text data.
- **Evaluation:** Some of the metrics in [3].
- **References:**

    1. [Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville, Bengio. Generative Adversarial Networks. NIPS 2014.](https://arxiv.org/abs/1406.2661)  
    2. [Zhao, Kim, Zhang, Rush, LeCun. Adversarially Regularized Autoencoders. ICML 2018.](http://proceedings.mlr.press/v80/zhao18b/zhao18b.pdf)  
    3. [Semeniuta, Severyn, Gelly. On Accurate Evaluation of GANs for Language Generation. 2018.](https://arxiv.org/abs/1806.04936)


---



# Sub-quadratic Transformers
- **Problem:** Transformers and BERT models are extremely large and expensive to train and keep in memory. The goal of this project is to make Transformers more efficient in terms of time and memory complexity by reducing the quadratic cost of self-attention or by inducing a sparser and smaller model.
- **Method:** See references below.
- **Data:** WMT datasets, WikiText, etc. See in the [original Transformer paper](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) and references below.
- **References:** 
    1. [Tay, Yi, Mostafa Dehghani, Dara Bahri, and Donald Metzler. "Efficient Transformers: A Survey." ArXiv 2020](https://arxiv.org/pdf/2009.06732.pdf) (e.g. Transformer-XL, Reformer, Linformer, Linear transformer, Compressive Transformer, etc.)
    2. [Gonçalo M. Correia, Vlad Niculae, André F.T. Martins. Adaptively Sparse Transformers. EMNLP 2019.](https://arxiv.org/pdf/1909.00015.pdf)
    3. [Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, Armand Joulin. Adaptive Attention Span in Transformers. ACL 2019.](https://www.aclweb.org/anthology/P19-1032.pdf)
    4. [Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever. Generating Long Sequences with Sparse Transformers. Arxiv 2019.](https://arxiv.org/pdf/1904.10509.pdf)

---

# Contextual Probabilistic Embeddings / Language Modeling

- **Problem:** Embedding words as vectors (aka point masses) cannot distinguish between more vague or more specific concepts. One solution is to embed words as a mean vector μ and a covariance Σ. Muzellec & Cuturi have a nice framework for this, tested for learning non-contextualized embeddings. Can we extend it to contextualized embeddings via language modelling? E.g. a model that reads an entire sentence and predicts a context-dependent pair (μ, Σ) for each word (perhaps left-to-right or masked). What likelihood to use? How can we evaluate the learned embeddings downstream?
- **Method:** See reference below.
- **References:**

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

# Sparse link prediction

- **Problem:** Predicting links in a large structured graph. For instance: predict co-authorship, movie recommendation, coreference resolution, discourse relations between sentences in a document.
- **Method:** The simplest approach is independent binary classification: for every node pair (i, j), predict whether there is a link or not. Issues: Very high imbalance: most nodes are not linked.
Structure and higher-order correlations are ignored in independent approach. Develop a method that can address the issues: incorporate structural correlations (e.g. with combinatorial inference, constraints, latent variables) and account for imbalance (ideally via pairwise ranking losses: learn a scorer such that S(i, j) > S(k, l) if there is an edge (i, j) but no edge (k, l).
- **Data:** [arXiv macro usage](https://github.com/CornellNLP/Macros), [Coreference in quizbowl](https://www.cs.umd.edu/~aguha/qbcoreference)
- **Notes:** Can graph-CNNs (previous idea) be useful here?
- **References:**

    1. [Rosenfeld, Meshi, Tarlow, Globerson. Learning Structured Models with the AUC Loss and Its Generalizations. AISTATS 2014.](http://ttic.uchicago.edu/~meshi/papers/structAUC_aistats14.pdf)

---

# Energy Networks

- **Problem:** Energy networks can be used for density estimation (estimating p(x)) and structured prediction (estimating p(y|x)) when y is structured. Both cases pose challenges due to intractability of computing the partition function and sampling. In structured output prediction with energy networks, the idea is to replace discrete structured inference with continuous optimization in a neural net. This project can be either a survey about recent work in this area or it can explore some practical applications. Applications: multi-label classification and sequence tagging. 
- **Method:** Learn a neural network E(x; w) to model the energy of x or E(x, y; w) to model the energy of an output configuration y (relaxed to be a continuous variable). Inference becomes min_y E(x, y; w). How far can this relaxation take us? Can it be better/faster than global combinatorial optimization approaches?
- **Data:** MNIST, multi-label classification, sequence tagging.
- **References:**

    1. [LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning. Predicting structured data, 1(0).](http://yann.lecun.com/exdb/publis/orig/lecun-06.pdf).  
    2. [Will Grathwohl, Kuan-Chieh Wang, Joern-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, Kevin Swersky (2020). Your classifier is secretly an energy based model and you should treat it like one. ICLR 2020.](https://openreview.net/pdf?id=Hkxzx0NtDB). 
    2. [Belanger and McCallum. Structured Prediction Energy Networks. ICML 2016.](http://proceedings.mlr.press/v48/belanger16.pdf)  
    3. [Belager, Yang, McCallum. End-to-End Learning for Structured Prediction Energy Networks. ICML 2017.](https://arxiv.org/abs/1703.05667)  
    
    
---

# Memory-augmented Neural Networks 
- **Problem:** Improve the generalization of neural nets by searching similar examples in the training set.
- **Method:** kNN + NN, fast search + NN, prototype attention (efficient attention over the dataset)
- **Data:** See in references below. 
- **References:**
    1. [Khandelwal, Urvashi, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through Memorization: Nearest Neighbor Language Models. ICLR 2020](https://openreview.net/pdf?id=HklBjCEKvH)
    2. [Wiseman, Sam, and Karl Stratos. Label-Agnostic Sequence Labeling by Copying Nearest Neighbors. ACL 2019](https://www.aclweb.org/anthology/P19-1533.pdf)
    3. [Lample, Guillaume, Alexandre Sablayrolles, Marc'Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. Large Memory Layers with Product Keys. NeurIPS 2019](http://papers.nips.cc/paper/9061-large-memory-layers-with-product-keys.pdf)
    4. [Hashimoto, Tatsunori B., Kelvin Guu, Yonatan Oren, and Percy S. Liang. A retrieve-and-edit framework for predicting structured outputs. NeurIPS 2018](https://scholar.google.com/scholar?hl=en&as_sdt=0,5&q=A+Retrieve-and-Edit+Framework+for+Predicting+Structured+Outputs&btnG=)
    5. [Guu, Kelvin, Tatsunori B. Hashimoto, Yonatan Oren, and Percy Liang.  Generating Sentences by Editing Prototypes. TACL 2018](https://www.mitpressjournals.org/doi/pdfplus/10.1162/tacl_a_00030)
    6. [Tu, Zhaopeng, Yang Liu, Shuming Shi, and Tong Zhang. Learning to Remember Translation History with a Continuous Cache. TACL 2018](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00029)
    7. [Weston, Jason, Emily Dinan, and Alexander H. Miller. Retrieve and Refine- Improved Sequence Generation Models For Dialogue. EMNLP - WSCAI 2018 ](https://www.aclweb.org/anthology/W18-5713.pdf)
    8. [Gu, Jiatao, Yong Wang, Kyunghyun Cho, and Victor OK Li. Search Engine Guided Neural Machine Translation. AAAI 2018](https://www.aaai.org/GuideBook2018/17282-74380-GB.pdf)

---

# Quality Estimation and Uncertainty Estimation
- **Problem:** Estimate the quality of a translation hypothesis without access to reference translations.
- **Method:** See [OpenKiwi](https://github.com/Unbabel/OpenKiwi): Neural Nets, Transfer Learning, BERT, XLM, etc.
- **Data:** in [WMT2020](http://www.statmt.org/wmt20/) page
- **References:** 
    1. [Kreutzer, Julia, Shigehiko Schamoni, and Stefan Riezler. QUality Estimation from ScraTCH (QUETCH): Deep Learning for Word-level Translation Quality Estimation. WMT 2015](https://www.aclweb.org/anthology/W15-3037.pdf)
    2. [Kim, Hyun, Jong-Hyeok Lee, and Seung-Hoon Na. Predictor-Estimator using Multilevel Task Learning with Stack Propagation for Neural Quality Estimation. WMT 2017](https://www.aclweb.org/anthology/W17-4763.pdf)
    3. [Wang, Jiayi, Kai Fan, Bo Li, Fengming Zhou, Boxing Chen, Yangbin Shi, and Luo Si. Alibaba Submission for WMT18 Quality Estimation Task. WMT 2018](http://statmt.org/wmt18/pdf/WMT093.pdf). 
    4. [Fabio Kepler, Jonay Trénous, Marcos Treviso, Miguel Vera, André F. T. Martins. OpenKiwi: An Open Source Framework for Quality Estimation. ACL 2019.](https://www.aclweb.org/anthology/P19-3020.pdf). 
    5. [Fomicheva, Marina, Shuo Sun, Lisa Yankovskaya, Frédéric Blain, Francisco Guzmán, Mark Fishel, Nikolaos Aletras, Vishrav Chaudhary, and Lucia Specia.  Unsupervised Quality Estimation for Neural Machine Translation. arXiv preprint 2020](https://arxiv.org/pdf/2005.10608.pdf)

---

# Causality and Disentanglement
- **Problem:** Causal inference and discovery is a area of growing interest in machine learning and statistics, with numerous applications and connections to confounding removal, reinforcement learning, and disentanglement of factors of variation. This project can be either a survey about the area or it can explore some practical applications.   
- **Method:** Plenty to choose from!
- **Data:** See the references below.  
- **References:** 
    1. [Elias Bareimboim. Causal Reinforcement Learning. Tutorial in ICML 2020.](https://icml.cc/Conferences/2020/ScheduleMultitrack?event=5752). 
    2. [Katherine A. Keith, David Jensen, and Brendan O'Connor. Text and Causal Inference: A Review of Using Text to Remove Confounding from Causal Estimates.](https://www.aclweb.org/anthology/2020.acl-main.474.pdf). 
    3. [Bengio, Yoshua, Tristan Deleu, Nasim Rahaman, Rosemary Ke, Sébastien Lachapelle, Olexa Bilaniuk, Anirudh Goyal, and Christopher Pal. A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms. ICLR 2020 ](https://openreview.net/forum?id=ryxWIgBFPS)
    4. [Zhu, Shengyu, Ignavier Ng, and Zhitang Chen. Causal Discovery with Reinforcement Learning. ICLR 2020](https://openreview.net/forum?id=S1g2skStPB)
    5. [Schölkopf, Bernhard. Causality for Machine Learning](https://arxiv.org/pdf/1911.10500.pdf)
    6. [Alvarez-Melis, David, and Tommi S. Jaakkola. A causal framework for explaining the predictions of black-box sequence-to-sequence models. EMNLP 2017](https://www.aclweb.org/anthology/D17-1042.pdf)

