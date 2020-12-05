Title: Deep Structured Learning (IST, Fall 2020)

# Summary

**Structured prediction** is a framework in machine learning which deals with structured and highly interdependent output variables, with applications in natural language processing, computer vision, computational biology, and signal processing.
In the last 5 years, several applications in these areas achieved new breakthroughs by replacing the traditional feature-based linear models by more powerful deep learning models based on neural networks, capable of learning internal representations.

In this course, we will describe methods, models, and algorithms for structured prediction, ranging from "shallow" **linear models** (hidden Markov models, conditional random fields, structured support vector machines) to modern **deep learning models** (convolutional networks, recurrent neural networks, attention mechanisms, etc.), passing through shallow and deep methods akin to reinforcement learning. Representation learning will also be discussed (PCA, auto-encoders, and various deep generative models).
The theoretical concepts taught in this course will be complemented by a strong practical component, by letting students work in group projects where they can solve practical problems by using software suitable for deep learning (e.g., Pytorch, TensorFlow, DyNet).

---

# Course Information

- **Instructor:** [André Martins](http://andre-martins.github.io)
- **TA:** [Marcos Treviso](http://mtreviso.github.io)
- **Schedule:** The classes are held on Wednesdays 14:00-17:00 remotely (Zoom link provided in Piazza)
- **Communication**: [piazza.com/tecnico.ulisboa.pt/fall2020/pdeecdsl](http://piazza.com/tecnico.ulisboa.pt/fall2020/pdeecdsl)

---

# Grading

- Homework assignments (60%)
- Final project (40%)

---

# Project Examples

The course project is an opportunity for you to explore an interesting problem using a real-world dataset. You can either choose one of [our suggested projects](/pages/project-examples-for-deep-structured-learning-fall-2020.html) or pick your own topic (the latter is encouraged). We encourage you to discuss your project with TAs/instructor to get feedback on your ideas.

**Team:** Projects can be done by a team of 2-4 students. You may use Piazza to find potential team mates.

**Milestones:** There are 3 deliverables:

- Proposal: A 1-page description of the project. Do not forget to include a title, the team members, and a short description of the problem, methodology, data, and evaluation metrics. **Due on 21/10.**
- Midway report: Introduction, related work, details of the proposed method, and preliminary results if available (4-5 pages). **Due on 25/11.**
- Final report: A full report written as a conference paper, including all the above in full detail, finished experiments and results, conclusion and future work (8 pages excluding references). **Due on 8/1.**

All reports should be in [NeurIPS format](https://nips.cc/Conferences/2018/PaperInformation/StyleFiles). There will be a class presentation and (tentatively) a poster session, where you can present your work to the peers, instructors, and other community members who will stop by.

See [here](/pages/project-examples-for-deep-structured-learning-fall-2020.html) for a list of project ideas.

---

# Recommended Bibliography

- [Deep Learning.](http://www.deeplearningbook.org) Ian Goodfellow and Yoshua Bengio and Aaron Courville. MIT Press, 2016.
- Machine Learning: a Probabilistic Perspective. Kevin P. Murphy. MIT Press, 2013.
- Introduction to Natural Language Processing. Jacob Einsenstein. MIT Press, 2019.
- Linguistic Structured Prediction. Noah A. Smith. Morgan & Claypool Synthesis Lectures on Human Language Technologies. 2011.

---

# Schedule

<table class="table table-condensed table-bordered table-hover">
<colgroup>
  <col span="1" style="width: 10%;">
  <col span="1" style="width: 45%;">
  <col span="1" style="width: 30%;">
  <col span="1" style="width: 15%;">
</colgroup>

<tr>
<th>Date</th>
<th>Topic</th>
<th>Optional Reading</th>
<th></th>
</tr>

<tr>
<td><b>Sep 23</b></td>
<td>
<a href="../docs/dsl2020/lecture_01.pdf">Introduction and Course Description</a>
</td>
<td>
<!--a href="http://lxmls.it.pt/2018/Figueiredo_LxMLS2018.pdf">Mário Figueiredo's LxMLS intro lecture</a><br/>
<a href="https://github.com/luispedro/talk-python-intro">Luis Pedro Coelho's intro to Python</a><br/-->
Goodfellow et al. Ch. 1-5<br/>
Murphy Ch. 1-2
</td>
<td></td>
</tr>

<tr>
<td><b>Sep 30</b></td>
<td>
<!--Linear Classifiers-->
<a href="../docs/dsl2020/lecture_02.pdf">Linear Classifiers</a>
</td>
<td>
Murphy Ch. 3, 6, 8-9, 14<br/>
Eisenstein Ch. 2
</td>
<td></td>
</tr>

<tr>
<td><b>Oct 7</b></td>
<td>
<!--Feedforward Neural Networks-->
<a href="../docs/dsl2020/lecture_03.pdf">Feedforward Neural Networks</a>
</td>
<td>
Goodfellow et al. Ch. 6
</td>
<td>
<a href=../docs/dsl2020/homework1.pdf>HW1 is out!</a> Skeleton code: <a href=../docs/dsl2020/hw1-q2.py>Q2</a>,  <a href=../docs/dsl2020/hw1-q3.py>Q3</a>.
</td>
</tr>

<tr>
<td><b>Oct 14</b></td>
<td>
<!--Neural Network Toolkits-->
<a href="https://github.com/mtreviso/pytorch-lecture">Neural Network Toolkits (Marcos Treviso)</a>
</td>
<td>
Goodfellow et al. Ch. 9, 14-15
</td>
<td></td>
</tr>

<tr>
<td><b>Oct 21</b></td>
<td>
<a href="../docs/dsl2020/lecture_04.pdf">Representation Learning and Convolutional Neural Networks</a>
</td>
<td>
Goodfellow et al. Ch. 7-8
</td>
<td>Project proposal is due.</td>
</tr>

<tr>
<td><b>Oct 28</b></td>
<td>
<a href="../docs/dsl2020/lecture_05.pdf">Linear Sequence Models</a>
</td>
<td>
Eisenstein, Ch. 6-8<br/>
Smith, Ch. 3-4<br/>
Murphy Ch. 17, 19
</td>
<td>
HW1 is due.<br/>
<a href=../docs/dsl2020/homework2.pdf>HW2 is out!</a> Skeleton code: <a href=../docs/dsl2020/hw2-q1.py>hw2-q1.py</a>, <a href=../docs/dsl2020/hw2-q2.py>hw2-q2.py</a>, <a href=../docs/dsl2020/hw2-q3.py>hw2-q3.py</a>, <a href=../docs/dsl2020/hw2_decoder.py>hw2_decoder.py</a>, <a href=../docs/dsl2020/hw2_linear_crf.py>hw2_linear_crf.py</a>, <a href=../docs/dsl2020/utils.py>utils.py</a>.
</td>
</tr>

<tr>
<td><b>Nov 4</b></td>
<td>
<a href="../docs/dsl2020/DeepRL.pdf">Deep Reinforcement Learning (Francisco Melo)</a><br/>
<a href="../docs/dsl2020/taxi.py">Game of Taxi</a><br/>
</td>
<td>
</td>
<td></td>
</tr>

<tr>
<td><b>Nov 11</b></td>
<td>
<a href="../docs/dsl2020/lecture_06.pdf">Recurrent Neural Networks</a>
</td>
<td>
Goodfellow et al. Ch. 10
</td>
<td></td>
</tr>

<tr>
<td><b>Nov 18</b></td>
<td>
<a href="../docs/dsl2020/lecture_07.pdf">Probabilistic Graphical Models</a>
</td>
<td>
Murphy Ch. 10, 19-22<br/>
Goodfellow et al. Ch. 16<br/>
<a href="http://www.inference.org.uk/itprnn/book.pdf">David MacKay's book, Ch. 16, 25-26</a><br/>
<a href="https://sailinglab.github.io/pgm-spring-2019/notes/lecture-04">Eric Xing's CMU lecture</a><br/>
<a href="https://ermongroup.github.io/cs228-notes/inference/ve">Stefano Ermon's notes on variable elimination</a>
<a href="https://www.bradyneal.com/causal-inference-course">Brady Neal's course on Causal Inference</a>
</td>
<td>
HW2 is due.<br/>
<a href=../docs/dsl2020/homework3.pdf>HW3 is out!</a>
</td>
</tr>

<tr>
<td><b>Nov 25</b></td>
<td>
<a href="../docs/dsl2020/lecture_08.pdf">Sequence-to-Sequence Learning</a>
</td>
<td>
Eisenstein, Ch. 18<br/>
<a href="https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf">Sutskever et al.</a>, 
<a href="https://arxiv.org/pdf/1409.0473.pdf">Bahdanau et al.</a>,
<a href="https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf">Vaswani et al.</a>
</td>
<td>
Midterm report is due.
</td>
</tr>

<tr>
<td><b>Dec 2</b></td>
<td>
<a href="../docs/dsl2020/attention-mechanisms.pdf">Attention Mechanisms (Marcos Treviso)</a>
</td>
<td>
<a href="https://vene.ro/talks/18-sparsemap-amsterdam.pdf">Learning with Sparse Latent Structure</a><br/> 
<a href="http://jalammar.github.io/illustrated-transformer">Illustrated Transformer</a>
</td>
<td></td>
</tr>

<tr>
<td><b>Dec 16</b></td>
<td>
<a href="../docs/dsl2020/lecture_11.pdf">Deep Generative Models</a><br/>
</td>
<td>
Goodfellow et al. Ch. 20<br/>
Murphy, Ch. 28<br/>
<a href="http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf">NeurIPS16 tutorial on GANs</a><br/>
<a href="https://arxiv.org/abs/1312.6114">Kingma and Welling, 2014</a><br/>
</td>
<td>HW3 is due (Dec 9).</td>
</tr>

<tr>
<td><b>Jan 8</b></td>
<td></td>
<td>
</td>
<td>
Final report is due.
</td>
</tr>

<tr>
<td><b>Jan 15, 22</b></td>
<td>Final Projects</td>
<td>
</td>
<td></td>
</tr>

</table>
