# L07 — Neural Networks & Deep Learning

> *Sarah Chen's eighth day at NorthStar Retail. Marcus's question from L06: "Can you predict checkout completion FROM SEQUENTIAL CUSTOMER BEHAVIOUR while they're shopping?" Today she trains her first neural network.*
> By the end of this lesson you will know what a perceptron is, what a multi-layer network does, how gradient descent + backprop make learning work, and how to write a PyTorch training loop end-to-end.

---

## Before you start — environment setup

> **If this is your first time, follow the [Setup Guide →](./setup.md)** (installs PyTorch).
>
> **Already set up from L01–L06?** Run `pip install torch` once — L07 introduces PyTorch as a new dependency.

---

## Where L07 fits

| Lesson | What you build |
|---|---|
| **L01–L05** | Classical ML — supervised + unsupervised on tabular data |
| **L06 — Time Series** | Forecasting on temporal data |
| **L07 — Neural Networks** *(you are here)* | First neural network: MLP for tabular classification + PyTorch training loop |
| **L08 — Computer Vision** | CNN — neural networks for images |
| **L09 — NLP** | Neural networks for text |
| **L10 — Transformers & GenAI** | The architecture behind ChatGPT, BERT, etc. |

**The narrative thread:** L07 is the **bridge** from classical ML to deep learning. The goal is intuition — what a neural network IS, how it LEARNS, and how to train one in PyTorch — not production engineering. The next three lessons specialise this toolkit to images (L08), text (L09), and transformers (L10).

---

## What you will be able to do by the end

- **Describe** what a perceptron is, why adding multiple layers + non-linearity is needed for non-linear problems
- **Explain** gradient descent + backpropagation in plain English (no chain-rule derivation required)
- **Write** a PyTorch training loop end-to-end — `Dataset` → `DataLoader` → model → loss → optimiser → forward → backward → step
- **Train** a simple multi-layer perceptron and reach a non-trivial accuracy on Sarah's session-completion data
- **Compare** the MLP to L03/L04 baselines (logistic regression, gradient boosting) — and have an opinion on when each is appropriate

---

## How class time is structured (~3 hrs)

| Phase | Time | Format |
|---|---|---|
| Concept recap (slides) | ~45–60 min | Instructor recaps the key concepts with the lesson slides — you already explored the [interactive key-concepts page](https://su-ntu-ctp.github.io/6m-data-3.7-Neural-Networks-Deep-Learning/) pre-class |
| Hands-on code-alongs | ~90 min | Three notebooks (~30 min each, including Pause-and-Predict discussion) |
| Class exit survey | ~15 min | Quick survey to capture what clicked and what didn't — helps shape the next session |
| (Self-study after class) | self-paced | Each notebook has 🟡 Extension + the assignment |

---

## Your learning path

### Phase 1 — Before class: self-study (~30 min)

**Start here →** [**pre-class.md**](./pre-class.md)

You'll run `01_morning_briefing.ipynb` to see the customer-session data + watch a video on neural networks + try three mini-exercises. After the video, explore the [**interactive key-concepts page**](https://su-ntu-ctp.github.io/6m-data-3.7-Neural-Networks-Deep-Learning/) (GitHub Pages).

---

### Phase 2 — In class: hands-on (~3 hrs)

**Short reference & review →** [**lesson.md**](./lesson.md) (overview, key takeaways, training-run checklist, 9-question review, L08→L10 course map)

**Need a recap? →** [**Key Concepts — interactive page**](https://su-ntu-ctp.github.io/6m-data-3.7-Neural-Networks-Deep-Learning/) (the page you explored pre-class — revisit any time)

**Notebooks — run in order:**

| # | Notebook | Sarah's day | What you explore |
|---|---|---|---|
| 02 | [`02_perceptron_to_mlp.ipynb`](./notebooks/02_perceptron_to_mlp.ipynb) | Late morning | Perceptron · why non-linearity is needed · sklearn MLPClassifier baseline |
| 03 | [`03_gradient_descent.ipynb`](./notebooks/03_gradient_descent.ipynb) | Early afternoon | Gradient descent intuition · backprop mental model · simple PyTorch model |
| 04 | [`04_pytorch_training_loop.ipynb`](./notebooks/04_pytorch_training_loop.ipynb) | Late afternoon | Full PyTorch training loop · Adam optimiser · compare to L04 baselines |

---

### Phase 3 — After class: assignment (self-paced)

**Assignment →** [`notebooks/assignment.ipynb`](./notebooks/assignment.ipynb)

Apply the MLP toolkit to a new tabular domain (loan-default prediction).

**Further reading →** [**reference.md**](./reference.md)

---

## Core vs Optional — what this lesson teaches

**🟢 Core (taught in class):**
- Perceptron → multi-layer mental model · activations · loss
- Gradient descent + backpropagation intuition
- PyTorch training loop end-to-end

**🟡 Optional (self-study, not assessed):**
- NumPy-from-scratch neural net (manual forward + backward pass)
- Weight initialisation strategies (Xavier, He)
- Adam vs SGD vs RMSprop comparison
- Dropout + batch normalisation

Optional material lives in [`notebooks/optional_extensions.ipynb`](./notebooks/optional_extensions.ipynb).

---

## File map

```
README.md                              ← You are here
setup.md                               ← One-time environment setup
pre-class.md                           ← Phase 1: ~30-min self-study
lesson.md                              ← Short reference: overview, takeaways, training-run checklist, review Q&A, course map
reference.md                           ← Phase 3: Further reading + glossary
environment.yml                        ← Conda environment spec
docs/
  index.html                           ← Interactive key-concepts page — explore during pre-class (GitHub Pages)
notebooks/
  data/
    northstar_sessions.csv             ← 8,000 customer sessions
  01_morning_briefing.ipynb              ← Pre-class hook
  02_perceptron_to_mlp.ipynb           ← Part 1: Perceptron → MLP
  03_gradient_descent.ipynb            ← Part 2: GD + PyTorch basics
  04_pytorch_training_loop.ipynb       ← Part 3: Full PyTorch loop + comparison
  assignment.ipynb                     ← After class: loan-default prediction
  optional_extensions.ipynb            ← 🟡 NumPy from scratch · weight init · Adam variants · dropout · batch norm
```
