Title: Project Examples for Deep Structured Learning (Spring 2023)

We suggest below some project ideas. Feel free to use this as inspiration for your project. Talk to us for more details.

---

# Multimodal Deep Learning 
- **Type:** Survey.
- **Problem:** Many tasks require combining different modalities, such as language and vision, language and speech, etc. This project will survey some of these approaches.
- **References:**

    1. [Visual Question Answering](https://visualqa.org/)
    2. [Tutorial on Multimodal Machine Learning](https://cmu-multicomp-lab.github.io/mmml-tutorial/cvpr2022/)
    3. [Alayrac et al. 2022. "Flamingo: a Visual Language Model for Few-Shot Learning". NeurIPS 2022.](https://openreview.net/pdf?id=EbMuimAbPbs)
    4. [S. Frank, E. Bugliarello, and D. Elliott. "Vision-and-Language or Vision-for-Language? On Cross-Modal Influence in Multimodal Transformers." EMNLP 2021.](https://arxiv.org/abs/2109.04448)

---

# Global Workspace Theory
- **Problem:** Current deep learning models lack higher order cognition capabilities. Global Workspace Theory is a cognitive science theory of consciousness that serves as inspiration to endow higher	order cognition	capabilities to neural networks.
- **References:**

    1. [Anirudh Goyal, Aniket Didolkar, Alex Lamb, Kartikeya Badola, Nan Rosemary Ke, Nasim Rahaman, Jonathan Binas, Charles Blundell, Michael Mozer, Yoshua Bengio. "Coordination Among Neural Modules Through a Shared Global Workspace."](https://arxiv.org/abs/2103.01197)  
    2. [Anirudh Goyal, Yoshua Bengio. "Inductive Biases for Deep Learning of Higher-Level Cognition."](https://arxiv.org/abs/2011.15091)  

---

# Reinforcement Learning with Human Feedback / Constitutional AI
- **Type:** Practical project or survey.  
- **Problem:** Current large language models such as ChatGPT include a final step of reinforcement learning with human feedback, where human preferences are taken into account to produce a reward model which is then used to fine-tune a pretrained model.  
- **References:**

    1. [Stiennon et al. "Learning to summarize from human feedback". NeurIPS 2020.](https://arxiv.org/abs/2009.01325)  
    2. [Bai et al. "Constitutional AI: Harmlessness from AI Feedback". 2022.](https://arxiv.org/abs/2212.08073)  

---

# Reinforcement Learning for Text Generation
- **Type:** Practical project or survey.  
- **Problem:** Current models for text generation trained with maximum likelihood estimation often suffer from exposure bias. New metrics for text generation (such as COMET, BLEURT, BARTSCORE) offer new strategies to train systems by maximizing a better reward function, using reinformement learning techniques.  
- **References:**

    1. [Han Guo, Bowen Tan, Zhengzhong Liu, Eric P. Xing, Zhiting Hu. "Text Generation with Efficient (Soft) Q-Learning"](https://arxiv.org/abs/2106.07704)  
    2. [Rémi Leblond, Jean-Baptiste Alayrac, Laurent Sifre, Miruna Pislar, Jean-Baptiste Lespiau, Ioannis Antonoglou, Karen Simonyan, Oriol Vinyals. "Machine Translation Decoding beyond Beam Search"](https://arxiv.org/abs/2104.05336)  
    3. [Richard Yuanzhe Pang, He He, Kyunghyun Cho. "Amortized Noisy Channel Neural Machine Translation."](https://arxiv.org/abs/2112.08670)  

---

# GFlowNets
- **Type:** Survey.  
- **Problem:** Survey Generative Flow Networks (GFlowNets), a recent method that allows samples a diverse set of candidates in an active learning context, with a training objective that makes them approximately sample in proportion to a given reward function.
- **References:**
    1. [Yoshua Bengio, Tristan Deleu, Edward J. Hu, Salem Lahlou, Mo Tiwari, Emmanuel Bengio. "GFlowNet Foundations."](https://arxiv.org/abs/2111.09266)  
    2. [Yoshua Bengio. Generative Flow Networks.](https://yoshuabengio.org/2022/03/05/generative-flow-networks/)  

---

# Memory consolidation and continual learning
- **Type:** Survey.  
- **Problem:** Current deep learning models store their long-term memory in the model parameters or learn new tasks on the fly through in-context learning, but they lack a middle ground so that they can assimilate new information on the fly based on their interactions and experience. The ability to learn on a continual basis is highly desirable in many applications, mimicking what humans do via memory consolidation. This project will survey existing work in this area, potentially exploring connections with neuroscience.
- **References:**
    1. [Lange et al. "A continual learning survey: Defying forgetting in classification tasks" TPAMI 2021.](https://arxiv.org/pdf/1909.08383.pdf)  
    2. [Biesialska et al. "Continual Lifelong Learning in Natural Language Processing: A Survey", COLING 2020.](https://aclanthology.org/2020.coling-main.574/)

---

# Prompting / Adaptors / Retrieval for NLP Tasks
- **Type:** Practical project.  
- **Problem:** Large pretrained language models can be expensive to fine-tune. Lightweight strategies include adaptors and prompting methods, as well as retrieval-based techniques. The goal is to experiment with (or combine) some of these techniques. For example, can we use retrieval (with a similarity search engine like FAISS) to generate good examples in a few-shot learning setting?
- **Data:** See papers.
- **References:**

    1. [Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig. "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing"](https://arxiv.org/abs/2107.13586) and references therein.
    2. [Khandelwal, Urvashi, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through Memorization: Nearest Neighbor Language Models. ICLR 2020](https://openreview.net/pdf?id=HklBjCEKvH)
    3. [Guu, Kelvin, Tatsunori B. Hashimoto, Yonatan Oren, and Percy Liang.  Generating Sentences by Editing Prototypes. TACL 2018](https://www.mitpressjournals.org/doi/pdfplus/10.1162/tacl_a_00030)

---

# Uncertainty Quantification / Conformal Prediction 

- **Type:** Practical project.  
- **Problem:** How to estimate the uncertainty of a structured classifier or regressor?
- **Method:** Monte Carlo dropout, deep ensembles, heteroscedastic regression, direct epistemic uncertainty prediction, etc.
- **References:**

    1. [Chrysoula Zerva, Taisiya Glushkova, Ricardo Rei, Andre F. T. Martins. "Disentangling Uncertainty in Machine Translation Evaluation. EMNLP 2022.](https://aclanthology.org/2022.emnlp-main.591/)  
    2. [Alex Kendall, Yarin Gal. "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NeurIPS 2017.](https://arxiv.org/abs/1703.04977)  
    3. [Anastasios N. Angelopoulos, Stephen Bates. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification". 2022.](https://arxiv.org/abs/2107.07511)  
    4. [Angelopoulos et al. "Conformal Risk Control". 2022.](https://arxiv.org/pdf/2208.02814.pdf)

    
---

# Sub-quadratic Sequence Models with S4

- **Type:** Practical project.  
- **Problem:** Transformers and BERT models are extremely large and expensive to train and keep in memory. The goal of this project is to survey or to make Transformers more efficient in terms of time and memory complexity by reducing the quadratic cost of self-attention or by inducing a sparser and smaller model. One possibility is to experiment with the recently proposed S4 model [1,2].
- **Method:** See references below.
- **Data:** Wikitext, Long Range Arena, etc. See in the [original Transformer paper](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) and references below.
- **References:**

    1. [Albert Gu, Karan Goel, Christopher Ré. "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022](https://arxiv.org/abs/2111.00396)  
    2. [Sasha Rush. "The Annotated S4."](https://srush.github.io/annotated-s4/)  
    3. [Tay, Yi, Mostafa Dehghani, Dara Bahri, and Donald Metzler. "Efficient Transformers: A Survey." ArXiv 2020](https://arxiv.org/pdf/2009.06732.pdf) (e.g. Transformer-XL, Reformer, Linformer, Linear transformer, Compressive Transformer, etc.)  
    4. [Gonçalo M. Correia, Vlad Niculae, André F.T. Martins. Adaptively Sparse Transformers. EMNLP 2019.](https://arxiv.org/pdf/1909.00015.pdf)  
    5. [Choromanski et al. "Rethinking Attention with Performers". ICLR 2021](https://arxiv.org/abs/2009.14794)  


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

# Energy-Based Models / Diffusion Models

- **Problem:** Energy networks can be used for density estimation (estimating p(x)) and structured prediction (estimating p(y|x)) when y is structured. Both cases pose challenges due to intractability of computing the partition function and sampling. In structured output prediction with energy networks, the idea is to replace discrete structured inference with continuous optimization in a neural net. This project can be either a survey about recent work in this area or it can explore some practical applications. Applications: multi-label classification and sequence tagging. 
- **Method:** Learn a neural network E(x; w) to model the energy of x or E(x, y; w) to model the energy of an output configuration y (relaxed to be a continuous variable). Inference becomes min_y E(x, y; w). How far can this relaxation take us? Can it be better/faster than global combinatorial optimization approaches?
- **Data:** MNIST, multi-label classification, sequence tagging.
- **References:**

    1. [LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning. Predicting structured data, 1(0).](http://yann.lecun.com/exdb/publis/orig/lecun-06.pdf).
    2. [Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." ICML 2021](https://arxiv.org/abs/2103.03230).
    3. [Will Grathwohl, Kuan-Chieh Wang, Joern-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, Kevin Swersky (2020). Your classifier is secretly an energy based model and you should treat it like one. ICLR 2020.](https://openreview.net/pdf?id=Hkxzx0NtDB). 
    4. [Belanger and McCallum. Structured Prediction Energy Networks. ICML 2016.](http://proceedings.mlr.press/v48/belanger16.pdf)  
    5. [Yang Song, Diederik P. Kingma. "How to Train Your Energy-Based Models."](https://arxiv.org/abs/2101.03288)  
    6. [Yang et al. 2022. "Diffusion Models: A Comprehensive Survey of Methods and Applications"](https://arxiv.org/abs/2209.00796)

---

# Causal Representation Learning / Causal Structure Models

- **Type:** Survey or practical project.  
- **Problem:** Causal inference and discovery is a area of growing interest in machine learning and statistics, with numerous applications and connections to confounding removal, reinforcement learning, and disentanglement of factors of variation. This project can be either a survey about the area or it can explore some practical applications.   
- **Method:** Plenty to choose from!
- **Data:** See the references below.  
- **References:** 
    1. [Elias Bareimboim. Causal Reinforcement Learning. Tutorial in ICML 2020.](https://icml.cc/Conferences/2020/ScheduleMultitrack?event=5752). 
    2. [Katherine A. Keith, David Jensen, and Brendan O'Connor. Text and Causal Inference: A Review of Using Text to Remove Confounding from Causal Estimates.](https://www.aclweb.org/anthology/2020.acl-main.474.pdf). 
    3. [Bengio, Yoshua, Tristan Deleu, Nasim Rahaman, Rosemary Ke, Sébastien Lachapelle, Olexa Bilaniuk, Anirudh Goyal, and Christopher Pal. A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms. ICLR 2020 ](https://openreview.net/forum?id=ryxWIgBFPS)
    4. [Zhu, Shengyu, Ignavier Ng, and Zhitang Chen. Causal Discovery with Reinforcement Learning. ICLR 2020](https://openreview.net/forum?id=S1g2skStPB)
    5. [Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, Yoshua Bengio. "Towards Causal Representation Learning". 2021.](https://arxiv.org/abs/2102.11107)
    6. [Alvarez-Melis, David, and Tommi S. Jaakkola. A causal framework for explaining the predictions of black-box sequence-to-sequence models. EMNLP 2017](https://www.aclweb.org/anthology/D17-1042.pdf)
    7. [Gonçalo R. A. Faria, André F. T. Martins, Mário A. T. Figueiredo. Differentiable Causal Discovery Under Latent Interventions. CLEAR 2022.](https://arxiv.org/abs/2203.02336)

---

# Associative Memory with Modern Hopfield networks  

- **Type:** Survey or practical project.
- **Project:** Hopfield networks [1] are recurrent neural networks with dynamical trajectories converging to fixed point attractor states dictated by the minimization of an energy function. These networks can be regarded as models of associative memory. Recently, a modern family of Hopfield networks has been proposed and studied with very interesting properties.  
- **References:**

   1. [Hopfield, John (1982). "Neural networks and physical systems with emergent collective computational abilities"](https://www.pnas.org/doi/pdf/10.1073/pnas.79.8.2554)  
   2. [Ramsauer, Hubert; et al. (2021). "Hopfield Networks is All You Need". ICLR 2020.](https://arxiv.org/abs/2008.02217)

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

