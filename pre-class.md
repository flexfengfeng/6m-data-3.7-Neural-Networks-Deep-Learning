# Before Class — L07 — Neural Networks & Deep Learning

**Estimated time: ~30 minutes.** Complete this before class.

This is the simplest version of "show up prepared": watch a short intro, run one notebook, answer a few questions. You'll come to class having seen the idea in action.

| Step | Time | What you do |
|---|---|---|
| **0. Watch** | ~5 min  | Watch the [lesson intro video](https://youtu.be/WESoo1cLmLw) |
| **1. Try it** | ~20 min | Open and run `notebooks/01_morning_briefing.ipynb` |
| **2. Reflect** | ~5 min  | Three short questions below |

---

## Step 0 — Watch the intro video (~5 min)

▶️ **[L07 Intro — Neural Networks & Deep Learning](https://youtu.be/WESoo1cLmLw)**

A short orientation to the day: why L07 is the bridge from classical ML to deep learning, and what Sarah is about to build for Marcus's checkout-completion question. Watch it before opening the notebook.

🕹️ **After the video:** open the [interactive key-concepts page](https://su-ntu-ctp.github.io/6m-data-3.7-Neural-Networks-Deep-Learning/) and play with it for 10–15 minutes. Drag the sliders, click the buttons — you can't break anything. Arriving in class having *seen* these ideas move makes the session far easier.

---

## Step 1 — Try it (~20 min)

Open **`notebooks/01_morning_briefing.ipynb`** in VS Code with the `dsai-m3` kernel. Run every cell top to bottom. Read the markdown between cells. Don't skip any cell.

Marcus's next ask: predict checkout completion from a customer's session behaviour. Sarah builds her first MLP in PyTorch. The notebook walks through tabular session data, baselines (LR, GB), and a small PyTorch network — and discovers all four models land within 0.015 AUC. Neural nets aren't magic; the lesson is the PyTorch training loop you'll reuse in L08-L10.

If this is your first time running a notebook in this repo, see [setup.md](./setup.md) once — you only need to do this for the first lesson.

---

## Step 2 — Quick reflection (~5 min)

Write a sentence or two for each. You can scribble in a notebook, in a journal, or just hold the answer in your head — what matters is that you *tried*.

**Q1. Why does the MLP tie LR on this tabular task?**

What kind of data makes neural networks shine instead? (One sentence — L08 will show you.)

**Q2. Identify each line of the PyTorch training loop.**

`zero_grad → forward → loss → backward → step`. What does each one do, in plain English?

**Q3. XOR by hand.**

Can a single-layer perceptron learn the XOR function? Why or why not? (You don't need to compute it — a sentence is fine.)

> *New to XOR?* It's the "exclusive or" rule: output 1 when **exactly one** of two inputs is 1, and 0 otherwise (so 0,0→0 · 1,0→1 · 0,1→1 · 1,1→0). We'll meet it properly in class — for now just think about whether one straight line could separate those answers.

---

## Bring to class

- Your answer to Q2 — the five-line training loop, named.
- One question about backprop or gradient descent that you'd like the class to discuss.

---

**Want to go deeper before class?** See **[reference.md](./reference.md) → *Further reading & watching*** at the bottom — videos and recommended readings that used to live in this guide.

**Ran out of time?** Doing just Step 1 (running the notebook) is enough. The class builds on having felt what the lesson teaches; the reflection questions get re-asked live.
