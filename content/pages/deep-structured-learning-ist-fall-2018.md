Title: Deep Structured Learning (IST, Fall 2018)

# Summary

**Structured prediction** is a framework in machine learning which deals with structured and highly interdependent output variables, with applications in natural language processing, computer vision, computational biology, and signal processing.
In the last 5 years, several applications in these areas achieved new breakthroughs by replacing the traditional feature-based linear models by more powerful deep learning models based on neural networks, capable of learning internal representations.

In this course, I will describe methods, models, and algorithms for structured prediction, ranging from "shallow" **linear models** (hidden Markov models, conditional random fields, structured support vector machines) to modern **deep learning models** (convolutional networks, recurrent neural networks, attention mechanisms, etc.), passing through shallow and deep methods akin to reinforcement learning. Representation learning will also be discussed (PCA, auto-encoders, and various deep generative models).
The theoretical concepts taught in this course will be complemented by a strong practical component, by letting students work in group projects where they can solve practical problems by using software suitable for deep learning (e.g., Pytorch, TensorFlow, DyNet).

---

# Course Information

- **Instructor:** [André Martins](http://andre-martins.github.io)
- **TAs:** [Vlad Niculae](http://vene.ro/), Erick Fonseca
- **Schedule:** Wednesdays 14:30-18:00, Room LT2 North Tower Level 4 (tentative)
- **Communication**: [http://piazza.com/tecnico.ulisboa.pt/fall2018/pdeecdsl](http://piazza.com/tecnico.ulisboa.pt/fall2018/pdeecdsl)

---

# Grading

- Homework assignments (60%)
- Final project (40%)

---

# Project Examples

The course project is an opportunity for you to explore an interesting problem using a real-world dataset. You can either choose one of [our suggested projects](/pages/project-examples-for-deep-structured-learning-fall-2018.html) or pick your own topic (the latter is encouraged). We encourage you to discuss your project with TAs/instructors to get feedback on your ideas.

**Team:** Projects can be done by a team of 2-4 students. You may use Piazza to find potential team mates.

**Milestones:** There are 3 deliverables:

- Proposal: A 1-page description of the project. Do not forget to include a title, the team members, and a short description of the problem, methodology, data, and evaluation metrics. **Due on 17/10.**
- Midway report: Introduction, related work, details of the proposed method, and preliminary results if available (4-5 pages). **Due on 14/11.**
- Final report: A full report written as a conference paper, including all the above in full detail, finished experiments and results, conclusion and future work (8 pages excluding references). **Due on 12/12.**

All reports should be in [NIPS format](https://nips.cc/Conferences/2018/PaperInformation/StyleFiles). There will be a class presentation and (tentatively) a poster session, where you can present your work to the peers, instructors, and other community members who will stop by.

See [here](/pages/project-examples-for-deep-structured-learning-fall-2018.html) for a list of project ideas.

---

# Recommended Bibliography

- [Deep Learning.](http://www.deeplearningbook.org) Ian Goodfellow and Yoshua Bengio and Aaron Courville. MIT Press, 2016.
- Machine Learning: a Probabilistic Perspective. Kevin P. Murphy. MIT Press, 2013.
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
<td><b>Sep 19</b></td>
<td>
<a href="../docs/dsl2018/lecture_01.pdf">Introduction and Course Description</a>
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
<td><b>Sep 26</b></td>
<td><a href="../docs/dsl2018/lecture_02.pdf">Linear Classifiers</a></td>
<td>
Murphy Ch. 3, 6, 8-9, 14
</td>
<td>
<a href=../docs/dsl2018/homework1.pdf>HW1 is out!</a>
</td>
</tr>

<tr>
<td><b>Oct 3</b></td>
<td><a href="../docs/dsl2018/lecture_03.pdf">Feedforward Neural Networks</a></td>
<td>
Goodfellow et al. Ch. 6
</td>
<td></td>
</tr>

<tr>
<td><b>Oct 10</b></td>
<td>
<a href="../docs/dsl2018/lecture_04.pdf">Neural Network Toolkits</a><br/>
<a href="https://github.com/erickrf/pytorch-lecture">Guest lecture: Erick Fonseca</a>
</td>
<td>
Goodfellow et al. Ch. 7-8
</td>
<td>
HW1 is due.<br/>
<a href=../docs/dsl2018/homework2.pdf>HW2 is out!</a>
</td>
</tr>

<tr>
<td><b>Oct 17</b></td>
<td>
<a href="../docs/dsl2018/lecture_05.pdf">Linear Sequence Models</a>
</td>
<td>
Smith, Ch. 3-4<br/>
Murphy Ch. 17, 19
</td>
<td>Project proposal is due.</td>
</tr>

<tr>
<td><b>Oct 24</b></td>
<td>
<a href="../docs/dsl2018/lecture_06.pdf">Representation Learning and Convolutional Neural Networks</a>
</td>
<td>
Goodfellow et al. Ch. 9, 14-15
</td>
<td></td>
</tr>

<tr>
<td><b>Oct 31 (rescheduled to Oct 29!)</b></td>
<td>Structured Prediction and Graphical Models</td>
<td>
Murphy Ch. 10, 19-22<br/>
Goodfellow et al. Ch. 16
</td>
<td>
HW2 is due.
</td>
</tr>

<tr>
<td><b>Nov 7</b></td>
<td>Recurrent Neural Networks</td>
<td>
Goodfellow et al. Ch. 10
</td>
<td></td>
</tr>

<tr>
<td><b>Nov 14</b></td>
<td>Sequence-to-Sequence Learning</td>
<td>
</td>
<td></td>
</tr>

<tr>
<td><b>Nov 21</b></td>
<td>Attention Mechanisms and Neural Memories</td>
<td>
</td>
<td></td>
</tr>

<tr>
<td><b>Nov 28</b></td>
<td>Deep Reinforcement Learning</td>
<td>
</td>
<td></td>
</tr>

<tr>
<td><b>Dec 5</b></td>
<td>Deep Generative Models (Variational Auto-Encoders and Generative Adversarial Networks)</td>
<td>
Goodfellow et al. Ch. 20<br/>
Murphy, Ch. 28
</td>
<td></td>
</tr>

<tr>
<td><b>Dec 12</b></td>
<td>Final Projects I</td>
<td>
</td>
<td></td>
</tr>

<tr>
<td><b>Dec 19</b></td>
<td>Final Projects II</td>
<td>
</td>
<td></td>
</tr>


</table>
