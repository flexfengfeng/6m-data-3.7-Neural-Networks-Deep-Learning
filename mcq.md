# L07 — Neural Networks & Deep Learning: 10 MCQs

Test your understanding of the key concepts. Answers and explanations are at the bottom.

---

**Q1. Why can a single perceptron not learn the XOR pattern?**

- A) XOR requires more input features than a perceptron can accept
- B) A perceptron's decision boundary is a single straight line (hyperplane), and XOR is not linearly separable
- C) The perceptron's learning rate is always too high for XOR
- D) XOR needs a softmax output, which a perceptron lacks

---

**Q2. You build a 5-layer MLP but set every activation function to the identity (no ReLU, no sigmoid). What is the result?**

- A) The network becomes more expressive because it is deeper
- B) The network collapses to the equivalent of a single linear layer
- C) Training will diverge immediately
- D) The network can now learn XOR but nothing else

---

**Q3. Which activation is the standard default for the *hidden* layers of a modern MLP?**

- A) Sigmoid
- B) Softmax
- C) ReLU (`max(0, x)`)
- D) Identity

---

**Q4. What does `optimizer.zero_grad()` do, and why is it required each batch?**

- A) Nothing important; it can be safely skipped
- B) It resets the model weights to zero before each batch
- C) It clears accumulated gradients, because PyTorch *adds* to `.grad` on every `backward()` rather than overwriting
- D) It sets the learning rate to zero to stabilise training

---

**Q5. Which line in the training step causes autograd to compute and fill in `.grad` on every parameter?**

- A) `optimizer.zero_grad()`
- B) `logits = model(X_batch)`
- C) `loss = criterion(logits, y_batch)`
- D) `loss.backward()`

---

**Q6. Your model ends with `nn.Linear(hidden, 1)` returning raw logits, and you use `nn.BCELoss()`. Why is this a bug?**

- A) `BCELoss` expects probabilities in `[0, 1]`, but raw logits are unbounded — the preferred fix is `nn.BCEWithLogitsLoss()`
- B) `BCELoss` only works for multi-class problems
- C) The final layer must output 2 units for binary classification
- D) There is no bug; this is the recommended setup

---

**Q7. With `L(w) = (w-3)²` starting at `w=10`, what happens if the learning rate is far too high (e.g. `lr=1.5`)?**

- A) Training crawls slowly but eventually converges
- B) The iterates oscillate with growing amplitude and the loss diverges
- C) The loss stays exactly constant
- D) The model converges faster than with a good learning rate

---

**Q8. Train loss keeps falling to 0.18 at epoch 30, but validation loss bottomed at 0.31 (epoch 12) and has crept up to 0.38. What is happening and what is the right fix?**

- A) Underfitting — train longer
- B) Overfitting — use early stopping and restore the weights from epoch 12
- C) The learning rate is too low — increase it
- D) Healthy training — ship it

---

**Q9. The lesson benchmarks LR, sklearn-MLP, PyTorch-MLP, and gradient boosting on Sarah's 8,000-session data. What is the key takeaway?**

- A) The MLP decisively beats every baseline
- B) All models land within ~0.015 AUC of each other; on small, mostly-linear tabular data an MLP rarely beats a strong baseline
- C) Gradient boosting is always strictly worse than neural networks
- D) The MLP fails completely on tabular data

---

**Q10. Why call `model.eval()` (and `torch.no_grad()`) before validation, even for a tiny MLP with no dropout or batch-norm?**

- A) It is required or the model will crash
- B) It speeds up training by 10×
- C) It is a habit that prevents silent bugs once dropout/batch-norm are added, since those behave differently in train vs eval mode
- D) It permanently freezes the model weights

---

## Answer key

| Q | Answer | Why |
|---|--------|-----|
| 1 | **B** | A perceptron computes `activation(w·x + b)` — one hyperplane. XOR's opposite-corner pattern can't be split by any line. Adding inputs keeps it linear; the fix is depth + non-linearity. |
| 2 | **B** | Composing linear maps yields another linear map. Without a non-linearity between layers, depth adds no expressive power. |
| 3 | **C** | ReLU is the default for hidden layers; sigmoid is for binary output, softmax for multi-class output. |
| 4 | **C** | PyTorch accumulates gradients; forgetting `zero_grad()` sums gradients across batches and `step()` takes a huge, stale step → divergence. |
| 5 | **D** | `loss.backward()` walks the computation graph backward and fills `.grad`. `optimizer.step()` then uses those grads to update parameters. |
| 6 | **A** | `BCELoss` needs post-sigmoid probabilities. Leave the model returning logits and switch to `BCEWithLogitsLoss`, which fuses sigmoid + BCE for numerical stability. |
| 7 | **B** | Too-high lr overshoots the minimum each step, oscillating with growing amplitude → divergence. Too-low lr crawls but doesn't diverge. |
| 8 | **B** | Widening train/val gap = overfitting. Early-stop and restore the best-validation weights (epoch 12); consider smaller model, more data, or regularisation. |
| 9 | **B** | The payoff of L07 isn't the scoreboard — it's the reusable training-loop toolkit for L08–L10 (images, text, transformers). |
| 10 | **C** | Dropout and batch-norm behave differently in train vs eval mode; `model.eval()` + `no_grad()` is a safe habit that prevents silent evaluation bugs. |
