# Lesson — L07 Neural Networks & Deep Learning

> **Chapter 7 of the NorthStar Retail story.** *Sarah Chen · Customer Experience Analyst · Day 8.*
> Marcus's brief from end-of-L06: *"Predict checkout completion from sequential customer behaviour — can a neural network do better than the L03/L04 baselines?"* Sarah opens `northstar_sessions.csv`: 8,000 customer sessions, 9 behavioural features, target = did they complete checkout.
> Today she builds her first neural network — and writes a PyTorch training loop she will reuse for the rest of the course.

This document is a **short reference** — the lesson itself is taught in the notebooks. Read it for orientation before class, then come back to it for the takeaways, the training-run checklist, the review questions, and the course map.

---

## How L07 is taught

| Stage | Where to go |
|---|---|
| **Pre-class** | `pre-class.md` + `notebooks/01_morning_briefing.ipynb` |
| **In-class — Part 1: Perceptron → MLP** | `notebooks/02_perceptron_to_mlp.ipynb` |
| **In-class — Part 2: Gradient descent + PyTorch basics** | `notebooks/03_gradient_descent.ipynb` |
| **In-class — Part 3: Full PyTorch training loop** | `notebooks/04_pytorch_training_loop.ipynb` |
| **Self-study** | `notebooks/assignment.ipynb` + `notebooks/optional_extensions.ipynb` |
| **Reference & review** | This document |

The notebooks are the spine. Run them in order. Come back here for the consolidated takeaways and the review questions.

---

## Overview

L07 is a single arc in three parts. Part 1 starts with the **perceptron** — a single neuron that is just logistic regression in disguise — and shows why it cannot learn XOR; the fix is stacking layers with non-linear activations between them, giving the multi-layer perceptron (MLP). Part 2 opens the hood on the `.fit()` call: **gradient descent** on a 1D loss surface, then PyTorch's autograd computing the same gradient automatically, then the five-line training step that powers every deep-learning model in the rest of the course. Part 3 promotes that loop to production shape — `DataLoader`, mini-batches, Adam, early stopping — and benchmarks the result against L03 logistic regression and L04 gradient boosting on Sarah's session data. The goal is **intuition and a reusable training-loop template**, not state-of-the-art tabular performance.

---

## Key takeaways

1. **A neural network is just a flexible non-linear function whose parameters are learned by gradient descent on a loss.** No mystery, no magic. Forward pass computes the output; the loss measures wrong; the gradient tells you which way to nudge the parameters.
2. **A single perceptron can only learn linearly separable patterns — XOR is the canonical counterexample.** Stack perceptrons in layers *with a non-linearity between them* and you can approximate any function. Two stacked linear layers with nothing in between is still one linear layer.
3. **Activation choice is mostly mechanical.** ReLU (`max(0, x)`) for hidden layers, sigmoid for binary-classification output, softmax for multi-class output. The non-linearity is what unlocks expressive power.
4. **Backpropagation is the chain rule applied efficiently to a computation graph.** You will never derive it by hand. PyTorch's autograd records every operation on a `requires_grad=True` tensor during the forward pass; `loss.backward()` walks the graph backwards to fill in `.grad` on every parameter.
5. **The learning rate is the single most important hyperparameter.** Too high and the loss oscillates or diverges; too low and training crawls. `lr=1e-3` with Adam is the safe default for tabular problems; tune from there. Adam adapts the per-parameter step size, removing most of the manual scheduling burden SGD demands.
6. **Memorise the five-line PyTorch training step.** `optimizer.zero_grad()` → forward pass → `loss = criterion(...)` → `loss.backward()` → `optimizer.step()`. This is identical for an MLP, a CNN, or a transformer — only the model class and loss change.
7. **Batch size trades stability for speed.** Full-batch GD gives the smoothest gradient but is slow and memory-hungry. Mini-batch SGD/Adam (typically 32–128) injects useful noise that helps escape shallow minima and lets you stream data that doesn't fit in memory.
8. **On well-preprocessed tabular data, an MLP rarely beats a strong linear or gradient-boosted baseline.** On Sarah's session data, LR / sklearn-MLP / PyTorch-MLP / GB all land within 0.015 AUC of each other. L07's payoff is not this scoreboard — it is the toolkit that makes L08 (images), L09 (text), and L10 (transformers) possible.

---

## Reading a training run — a checklist

Before you accept a trained model's number — or change a hyperparameter in frustration — run the loss curves through this three-step check:

1. **Did it converge?** Plot train loss and validation loss against epoch. Train loss should fall and then plateau. If it is still falling sharply at the last epoch, you stopped too early. If it is bouncing wildly, the learning rate is too high. If it never moved, the learning rate is too low or your data isn't reaching the model.
2. **Is it overfitting?** Train loss keeps falling while validation loss bottoms out and starts rising — that gap is overfitting. Fix it with early stopping (restore the weights from the best validation epoch), more regularisation, more data, or a smaller model. Do not look at the test set while tuning.
3. **Is the learning rate right?** A healthy loss curve drops fast at first, then bends and plateaus smoothly. A jagged or exploding curve means lr too high; a near-flat curve means lr too low. Try a 10× sweep (`1e-2`, `1e-3`, `1e-4`) before tuning anything else.

Skip any of these and you'll either ship an undertrained model, an overfit one, or burn an afternoon tuning architecture when the real bug was the learning rate.

---

## Key concepts — plain-English review

Use this as a self-check before the review questions: read each concept, and if any feels fuzzy, jump back to the notebook or section that teaches it.

**Neuron / perceptron** — The smallest building block: it multiplies each input by a weight, adds them up, and passes the total through a simple yes/no-ish function. One neuron alone can only draw a straight-line boundary.
*Real-world use:* An email filter's simplest form — add up spammy-word scores and flag the message if the total crosses a threshold.

**Weights** — The numbers the network learns; each weight says how much one input matters. Training is just nudging weights until predictions improve.
*Real-world use:* A bank's credit model learning that "missed payments" deserves a much bigger weight than "age of account".

**Activation function** — A simple bend (like ReLU: "keep positives, zero out negatives") applied between layers. Without it, stacking layers adds nothing — the bend is what lets networks learn curvy, complex patterns.
*Real-world use:* Recognising that risk isn't a straight line — a hospital readmission model where risk jumps sharply only past a certain age.

**Hidden layers / MLP** — Stack neurons in layers (a multi-layer perceptron) and each layer builds on the last, letting the network learn patterns no single straight line can capture.
*Real-world use:* Sarah's checkout-completion model at NorthStar, combining 9 session behaviours into one probability.

**Loss function** — A single number that measures "how wrong were we?" on the training data. Training means making this number smaller.
*Real-world use:* A ride-hailing app scoring its arrival-time predictions — every minute off adds to the loss it tries to shrink.

**Gradient descent** — The learning method: check which direction makes the loss smaller, take a small step that way, repeat. Like walking downhill in fog, one step at a time.
*Real-world use:* Any modern recommendation engine — Netflix's suggestions improve by millions of tiny downhill steps.

**Learning rate** — The step size. Too big and you overshoot the valley and bounce around; too small and you crawl forever. The single most important dial to tune.
*Real-world use:* Like adjusting a thermostat — turning it 10 degrees at a time overshoots; 0.1 degrees takes all day.

**Backpropagation** — The bookkeeping trick that works out, for every weight in the network, how much it contributed to the error — so all weights can be nudged at once. PyTorch does it automatically.
*Real-world use:* Post-mortem in a factory: tracing a defective product back through each station to see which step contributed how much to the fault.

**Training loop and epochs** — The repeated cycle: predict, measure loss, compute nudges, update weights. One full pass through the training data is an epoch; training runs many epochs and stops when validation performance stops improving.
*Real-world use:* A voice assistant retrained nightly runs this same loop; engineers watch the loss curve like a fitness tracker to know when to stop.

---

## Check your understanding

Work through these after finishing the three Part notebooks. Attempt each question on your own first.

### Part 1 — Perceptron → MLP

**Q1 — Why does XOR break a single perceptron?** A colleague says "we should just use a bigger perceptron with more inputs." Why doesn't that fix XOR?

> *Sample answer:* A perceptron computes `activation(w·x + b)` — its decision boundary is a single hyperplane (a straight line in 2D). The XOR pattern (1s in opposite corners) cannot be separated by *any* straight line, no matter how the weights are tuned. Adding more *inputs* keeps the boundary linear in those inputs. The fix is depth + a non-linearity between layers, not width of a single layer.

**Q2 — Why does the non-linearity matter?** What would happen if you built a 5-layer MLP but left every activation function as the identity (no ReLU, no sigmoid)?

> *Sample answer:* The whole network collapses to a single linear function. Composing linear maps gives another linear map, so the 5-layer network has exactly the same expressive power as one linear layer — it cannot learn XOR or any other non-linear pattern. The non-linearity between layers is what makes "deep" different from "wide-but-flat."

**Q3 — When does an MLP actually win on tabular data?** Sarah's session-completion benchmark shows LR, MLP, and gradient boosting all within 0.015 AUC. When *would* the MLP pull ahead?

> *Sample answer:* When the data is large (typically millions of rows), the features have rich non-linear interactions that aren't already captured by feature engineering, or when you need to ensemble with other neural models. On small, mostly-linear tabular sets — like 8,000 sessions with 9 engineered features — a well-tuned linear model is hard to beat. The MLP's real edge appears in L08–L10, where the input is pixels or tokens rather than tabular features.

### Part 2 — Gradient descent + backprop

**Q4 — What does `optimizer.zero_grad()` do, and what breaks if you forget it?** Walk through the consequence.

> *Sample answer:* PyTorch *accumulates* gradients in `.grad` on every `backward()` call rather than overwriting them. `optimizer.zero_grad()` clears the slate before the next forward/backward pass. If you forget it, each batch's gradient is added to the previous batches' gradients, so by step 10 your "gradient" is the sum of ten batches and `optimizer.step()` takes a wildly oversized step in a stale direction. Training will diverge or stall.

**Q5 — Learning rate too high vs too low.** You start gradient descent at `w=10` on `L(w)=(w-3)²`. With `lr=0.1` you converge smoothly. Describe what you'd see with `lr=1.5` and with `lr=0.001`.

> *Sample answer:* With `lr=1.5`, each step overshoots the minimum (3) and lands further on the other side; the next step overshoots back even further. The iterates oscillate with growing amplitude and the loss diverges. With `lr=0.001`, every step is tiny — the path crawls toward 3 and may not arrive within your epoch budget. Cost is wasted compute, not divergence. The sweet spot for this problem is roughly `0.01`–`0.1`; Adam removes most of this manual tuning by adapting the rate per parameter.

**Q6 — `BCELoss` vs `BCEWithLogitsLoss`.** A teammate's training run is silently producing garbage. Their model ends with `nn.Linear(hidden, 1)` (raw logits) and they use `criterion = nn.BCELoss()`. What's wrong?

> *Sample answer:* `BCELoss` expects probabilities in `[0, 1]` (post-sigmoid). Raw logits can be any real number; feeding them to `BCELoss` gives nonsensical (and numerically unstable) gradients. Two fixes: either apply `torch.sigmoid` inside `forward` and keep `BCELoss`, or — preferred — leave the model returning raw logits and switch to `nn.BCEWithLogitsLoss()`, which fuses sigmoid and BCE for numerical stability.

### Part 3 — PyTorch training loop

**Q7 — The five-line training step.** Without looking, write out the five lines inside a typical PyTorch training-batch loop, in order, and say what each one does.

> *Sample answer:*
> 1. `optimizer.zero_grad()` — clear stale gradients from the previous batch.
> 2. `logits = model(X_batch)` — forward pass; PyTorch records the operations for autograd.
> 3. `loss = criterion(logits, y_batch)` — compute the scalar loss.
> 4. `loss.backward()` — autograd walks the graph backwards and fills `.grad` on every parameter.
> 5. `optimizer.step()` — update each parameter using its `.grad` and the optimiser's rule (e.g. Adam).

**Q8 — Spotting overfitting in a training curve.** Train loss is 0.18 at epoch 30 and still falling; validation loss bottomed out at 0.31 around epoch 12 and has been creeping back up to 0.38. Is this model healthy? What do you do?

> *Sample answer:* No — this is textbook overfitting. The train/val gap is widening because the model is memorising training-set quirks that don't generalise. Use early stopping: restore the weights from epoch 12 (the best validation loss) and stop training. If you want better validation loss, try a smaller model, more data, dropout/weight decay, or stronger preprocessing — not more epochs.

**Q9 — `model.train()` vs `model.eval()`.** Why must you call `model.eval()` before evaluating on the validation set, even if your model has no dropout or batch-norm yet?

> *Sample answer:* For this lesson's tiny MLP it doesn't change the numbers — but it's a habit that prevents silent bugs the moment you add dropout or batch-norm. Dropout zeroes random activations only during training; batch-norm uses batch statistics in `train` mode and stored running statistics in `eval` mode. Forgetting the switch gives randomised or batch-dependent predictions at evaluation time. Always wrap evaluation in `model.eval()` plus `with torch.no_grad():`.

---

## Where L07 fits in the course

L07 introduces the neural-network playbook — model, loss, optimiser, training loop — that every remaining lesson specialises.

| Lesson | How L07 shows up |
|---|---|
| **L08 — Computer Vision** | Same five-line training step. The model becomes a CNN (convolutional layers, pooling) and the input is a tensor of pixels rather than a row of features. Transfer learning swaps in a pretrained backbone. |
| **L09 — NLP & Embeddings** | Same training loop again. The model becomes an embedding + sequence encoder; the input is tokenised text rather than a numeric vector. Loss and optimiser are unchanged. |
| **L10 — Transformers & GenAI** | The architecture is a transformer, the optimiser is AdamW, the loss is cross-entropy over the vocabulary — but the outer training loop is the same `zero_grad → forward → loss → backward → step` pattern from today. |

---

> *"OK. Now — we're starting an Instagram campaign. Can you automatically tag product PHOTOS so they auto-categorise as 'dress', 'jeans', 'jacket', etc.?"* — Marcus, after Sarah's end-of-day presentation.
>
> That question — *prediction from images* — is the engine of **L08 (Computer Vision)**.
