Title: Deep Structured Learning (IST, Fall 2019)

# Summary

**Structured prediction** is a framework in machine learning which deals with structured and highly interdependent output variables, with applications in natural language processing, computer vision, computational biology, and signal processing.
In the last 5 years, several applications in these areas achieved new breakthroughs by replacing the traditional feature-based linear models by more powerful deep learning models based on neural networks, capable of learning internal representations.

In this course, we will describe methods, models, and algorithms for structured prediction, ranging from "shallow" **linear models** (hidden Markov models, conditional random fields, structured support vector machines) to modern **deep learning models** (convolutional networks, recurrent neural networks, attention mechanisms, etc.), passing through shallow and deep methods akin to reinforcement learning. Representation learning will also be discussed (PCA, auto-encoders, and various deep generative models).
The theoretical concepts taught in this course will be complemented by a strong practical component, by letting students work in group projects where they can solve practical problems by using software suitable for deep learning (e.g., Pytorch, TensorFlow, DyNet).

---

# Course Information

- **Instructors:** [André Martins](http://andre-martins.github.io) and [Vlad Niculae](http://vene.ro/)
- **Schedule:** The classes are held on Mondays 9:30-11:00 and Fridays 15:00-16:30 in Room LT2 (North Tower, 4th floor)
- **Communication**: [piazza.com/tecnico.ulisboa.pt/fall2019/pdeecdsl](http://piazza.com/tecnico.ulisboa.pt/fall2019/pdeecdsl)

---

# Grading

- Homework assignments (60%)
- Final project (40%)

---

# Project Examples

The course project is an opportunity for you to explore an interesting problem using a real-world dataset. You can either choose one of [our suggested projects](/pages/project-examples-for-deep-structured-learning-fall-2019.html) or pick your own topic (the latter is encouraged). We encourage you to discuss your project with TAs/instructors to get feedback on your ideas.

**Team:** Projects can be done by a team of 2-4 students. You may use Piazza to find potential team mates.

**Milestones:** There are 3 deliverables:

- Proposal: A 1-page description of the project. Do not forget to include a title, the team members, and a short description of the problem, methodology, data, and evaluation metrics. **Due on 17/10.**
- Midway report: Introduction, related work, details of the proposed method, and preliminary results if available (4-5 pages). **Due on 14/11.**
- Final report: A full report written as a conference paper, including all the above in full detail, finished experiments and results, conclusion and future work (8 pages excluding references). **Due on 12/12.**

All reports should be in [NIPS format](https://nips.cc/Conferences/2018/PaperInformation/StyleFiles). There will be a class presentation and (tentatively) a poster session, where you can present your work to the peers, instructors, and other community members who will stop by.

See [here](/pages/project-examples-for-deep-structured-learning-fall-2019.html) for a list of project ideas.

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
<td><b>Sep 18</b></td>
<td>
<a href="../docs/dsl2019/lecture_01.pdf">Introduction and Course Description</a>
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
<td><b>Sep 23, 27</b></td>
<td>
<!--Linear Classifiers<-->
<a href="../docs/dsl2019/lecture_02.pdf">Linear Classifiers</a>
</td>
<td>
Murphy Ch. 3, 6, 8-9, 14
</td>
<td>
<a href=../docs/dsl2019/homework1.pdf>HW1 is out!</a> <a href=../docs/dsl2019/hw1.py>Skeleton code.</a> 
</td>
</tr>

<tr>
<td><b>Sep 30, Oct 4</b></td>
<td>
<!-->Feedforward Neural Networks<-->
<a href="../docs/dsl2019/lecture_03.pdf">Feedforward Neural Networks</a>
</td>
<td>
Goodfellow et al. Ch. 6
</td>
<td></td>
</tr>

<tr>
<td><b>Oct 7, 11</b></td>
<td>
Neural Network Toolkits
<!--a href="../docs/dsl2018/lecture_04.pdf">Neural Network Toolkits</a-->
<br/>
<!--a href="https://github.com/erickrf/pytorch-lecture">Guest lecture: Erick Fonseca</a-->
</td>
<td>
Goodfellow et al. Ch. 7-8
</td>
<td>
HW1 is due.<br/>
<!--a href=../docs/dsl2018/homework2.pdf>HW2 is out!</a-->
</td>
</tr>

<!--tr>
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
<td><b>Oct 31 (rescheduled to Oct 29, rooms V1.17/V1.11!)</b></td>
<td>
<a href="../docs/dsl2018/lecture_07.pdf">Structured Prediction and Graphical Models</a>
</td>
<td>
Murphy Ch. 10, 19-22<br/>
Goodfellow et al. Ch. 16<br/>
<a href="http://www.inference.org.uk/itprnn/book.pdf">David MacKay's book, Ch. 16, 25-26</a>
</td>
<td>
HW2 is due.<br/>
<a href=../docs/dsl2018/homework3.pdf>HW3 is out!</a>
</td>
</tr>

<tr>
<td><b>Nov 7</b></td>
<td>
<a href="../docs/dsl2018/lecture_08.pdf">Recurrent Neural Networks</a>
</td>
<td>
Goodfellow et al. Ch. 10
</td>
<td></td>
</tr>

<tr>
<td><b>Nov 14 (room E5)</b></td>
<td>
<a href="../docs/dsl2018/lecture_09.pdf">Sequence-to-Sequence Learning</a>
</td>
<td>
<a href="https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf">Sutskever et al.</a>, 
<a href="https://arxiv.org/pdf/1409.0473.pdf">Bahdanau et al.</a>,
<a href="https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf">Vaswani et al.</a>
</td>
<td></td>
</tr>

<tr>
<td><b>Nov 21 (room F8)</b></td>
<td>
<a href="../docs/dsl2018/lecture_10.pdf">Attention Mechanisms and Neural Memories</a><br/>
<a href="../docs/dsl2018/attention.pdf">Guest lecture: Vlad Niculae</a>
</td>
<td>
<a href="https://vene.ro/talks/18-sparsemap-amsterdam.pdf">Learning with Sparse Latent Structure</a>
</td>
<td>HW3 is due.<br/>
<a href=../docs/dsl2018/homework4.pdf>HW4 is out!</a>
</td>
</tr>

<tr>
<td><b>Nov 28</b></td>
<td>
<a href="../docs/dsl2018/DeepRL.pdf">Deep Reinforcement Learning</a><br/>
<a href="../docs/dsl2018/taxi.py">Game of Taxi</a><br/>
Guest lecture: Francisco Melo
</td>
<td>
</td>
<td>
Midterm report is due.
</td>
</tr>

<tr>
<td><b>Dec 5</b></td>
<td>
<a href="../docs/dsl2018/lecture_12.pdf">Deep Generative Models</a><br/>
</td>
<td>
Goodfellow et al. Ch. 20<br/>
Murphy, Ch. 28<br/>
<a href="http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf">NIPS16 tutorial on GANs</a><br/>
<a href="https://arxiv.org/abs/1312.6114">Kingma and Welling, 2014</a><br/>
</td>
<td></td>
</tr>

<tr>
<td><b>Jan 9</b></td>
<td></td>
<td>
</td>
<td>
Final report is due.
</td>
</tr>

<tr>
<td><b>Jan 16</b></td>
<td>Final Projects I</td>
<td>
</td>
<td></td>
</tr>

<tr>
<td><b>Jan 23</b></td>
<td>Final Projects II</td>
<td>
</td>
<td></td>
</tr-->


</table>
