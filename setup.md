# Setup — L07 — Neural Networks & Deep Learning

> One-time environment setup. **If you already set up the `dsai-m3` environment for L01–L06, you're almost done — L07 introduces one new dependency, PyTorch (`torch >= 2.2`). If you created the environment from any lesson's `environment.yml`, torch is already included; otherwise run `pip install torch` once inside the activated environment. Then just clone this repo and pick the `dsai-m3` kernel.**

This guide gets you from "fresh machine" to "running the L07 notebooks". It takes about 15 minutes the first time.

---

## 1. Prerequisites

You need:

- **Git** — to clone this repository
- **Miniconda** (or Anaconda) — to create the Python environment. [Install Miniconda →](https://docs.conda.io/projects/miniconda/en/latest/)
- **VS Code** — with the Python and Jupyter extensions. [Download VS Code →](https://code.visualstudio.com/)

Check git and conda are working:

```bash
git --version
conda --version
```

---

## 2. Clone this repository

```bash
git clone https://github.com/flexfengfeng/dsai-m3-l07-learner.git
cd dsai-m3-l07-learner
```

---

## 3. Create the conda environment

From inside the `dsai-m3-l07-learner` folder:

```bash
conda env create -f environment.yml
```

This creates an environment called **`dsai-m3`** with Python, pandas, numpy, matplotlib, scikit-learn, PyTorch, and Jupyter. It takes 5–10 minutes (torch is a large download).

> **Already have a `dsai-m3` environment from L01–L06?** You don't need to recreate it. Just make sure torch is present:
>
> ```bash
> conda activate dsai-m3
> python -c "import torch; print(torch.__version__)"
> ```
>
> If that errors, run `pip install "torch>=2.2"` once. No GPU needed — every network in L07 is small and trains from scratch on CPU. No pretrained model downloads either.

Activate the environment:

```bash
conda activate dsai-m3
```

---

## 4. Open the project in VS Code

```bash
code .
```

When VS Code opens, it may prompt you to install recommended extensions (Python, Jupyter). Accept.

---

## 5. Select the `dsai-m3` kernel

1. Open `notebooks/01_morning_briefing.ipynb`.
2. In the top-right of the notebook, click **Select Kernel**.
3. Choose **Python Environments → dsai-m3**.

If `dsai-m3` doesn't appear, restart VS Code and try again. If it still doesn't appear, run `conda env list` in a terminal to confirm the environment exists.

---

## 6. Smoke test

In `01_morning_briefing.ipynb`, run the first cell (the imports). If it completes without errors, you're set.

If you see `ModuleNotFoundError`, the wrong kernel is selected — go back to Step 5. If only `torch` is missing, see Step 3's note.

---

## Troubleshooting

**`conda: command not found`** → Miniconda isn't installed or isn't on your PATH. Reinstall Miniconda and restart your terminal.

**`environment.yml` solver hangs** → Try `conda env create -f environment.yml --solver=libmamba`. If that still hangs, delete the env (`conda env remove -n dsai-m3`) and retry.

**Kernel doesn't appear in VS Code** → Run `python -m ipykernel install --user --name dsai-m3` from inside the activated environment.

**Notebooks run but plots don't show** → Make sure you have `%matplotlib inline` at the top, or use the VS Code "Run Cell" button (not just `Shift+Enter` in a stale terminal).

---

Once setup is done, head back to **[README.md](./README.md)** → Phase 1.
