# Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data?

This repository contains all code for reproducing experiments from the paper [Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data?](https://arxiv.org/abs/2407.16607) Given a BPE tokenizer, our attack infers its training data distribution with high precision, recovering e.g., the proportion of different natural languages, code, and sources of data. In general, the attack will work for any set of data categories that are reasonably expected to cover the training data and have different "word" distributions. In controlled experiments, our attack performs 2 to 5 *orders of magnitude* better than baselines based on tokenizer encoding efficiency or analysis of the vocabulary. In robustness experiments (see §6), we show that the attack remains strong even when the attacker does not have access to the same data distribution, or when not all data categories are known to the attacker.

Please reach out to jhayase@cs.washington.edu and alisaliu@cs.washington.edu with any questions, including if you need help running the attack on new tokenizers.

# Download data
We use the [Oscar](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) dataset for mixtures of natural languages, [RedPajama-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) [Github split](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/blob/main/urls/github.txt) for mixtures of programming languages, and [RedPajama-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) for mixtures of domains. You can download a subset of the data from [this Google drive folder](https://drive.google.com/drive/folders/1jF12d_NsuNfVpC9JsAbK215gVYw72gXA?usp=sharing) for quick experimentation. To recreate the full dataset, please follow the instructions in `preprocessing/`.

# Setting up the environment

**Important note**: our project depends on a custom [fork](https://github.com/alisawuffles/tokenizers-bpe-attack) of [`huggingface/tokenizers`](https://github.com/huggingface/tokenizers) which conflicts with the original.
Because of this, we recommend *always installing this project in its own virtual environment*.

Our project depends on [Gurobi](https://www.gurobi.com/) and requires a rust compiler to build. You can obtain a free Gurobi academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Using Conda

```
conda create -n tokenizer-attack python=3.12 rust gurobi::gurobi
conda activate tokenizer-attack
# in the project root
pip install -r requirements.txt
```

## Using [PDM](https://pdm-project.org)

Install rust and Gurobi using your preferred method, then run `pdm install`.

# Reproducing controlled experiments
There are three steps to reproducing our main experiments, where we train tokenizers on known mixtures of data categories and evaluate our attack's effectiveness at reconstructing the true data mixture.

1. **Train tokenizers on mixtures of data categories.** See `script_examples/train_tokenizer.sh` for an example script. This will create a `merges.txt` and `meta.json` file in the specified output directory (e.g., `experiments/mixed_languages/n_112/0`), as well an empty directory for every data category in the mixture.
2. **Apply the learned merge list step-by-step to a given corpus.** This records the frequency of all possible merges at each step. This will create a `all_pair_counts.json` and `meta.json` inside the category subdir (e.g., `experiments/mixed_languages/n_112/0/en`). This needs to be run once for every possible data category; see `script_examples/get_merge_frequencies.sh` for an example script.
3. Apply the attack by running `python run_solver.py <output_dir>`. The final predictions can be found in `solution_[options].json` where `[options]` contains the solver parameters.

In `notebooks/experimental_results.ipynb`, you can find scripts for calculating the mean MSE over test trials and visualizing results.

# Apply our attack to a new, off-the-shelf tokenizer
Try applying our attack to new tokenizers! Skip step 1 if you are reproducing results on a tokenizer we included. You can also find detailed predictions for the tokenizers we considered in the `solution6_pairs_X.json` files (where `X` is ~30K) in `experiments/llm_tokenizers`.

1. Add the normalization and pretokenization configuration for your tokenizer of interest to `llm_tokenizer_configs.py`. If the tokenizer has a `tokenizer.json` available ([example for `GPT-4o`](https://huggingface.co/Xenova/gpt-4o/blob/main/tokenizer.json)), this information can usually be found in the `normalizer` and `pre_tokenizer` fields. Otherwise, you may need to reconstruct the pretokenization rules via manual inspection. Some common axes of variation include: (1) Is this a byte-level or character-level tokenizer? (2) Are digits merged with each other? (3) Are punctuation merged with neighboring punctuation, and with neighboring non-punctuation characters? Please see the [`tokenizers.pre_tokenizers`](https://huggingface.co/docs/tokenizers/en/api/pre-tokenizers) documentation to understand the available options. Reconstructing pretokenization rules perfectly is not necessary, but it is helpful so that the set of possible merges considered at every time step by our attack as close as possible to the true set considered during tokenizer training.
2. Create a directory for your tokenizer (e.g., `experiments/llm_tokenizers/gpt4o`) and add the corresponding `merges.txt`. For many tokenizers, this is directly available ([example for `GPT-4o`](https://huggingface.co/Xenova/gpt-4o/blob/main/merges.txt)). Some tokenizers trained with `sentencepiece` may have blocks of redundant merges, which is an artifact of the conversion from the `sentencepiece` to HuggingFace `tokenizers` format (see §C.3); these can be removed with the code in `notebooks/clean_merge_list.ipynb`. Additionally, if you have reason to believe that some merges are manually added by tokenizer creators and not learned organically by the BPE algorithm (e.g., the merges of spaces in `Gemma`'s merge list, see §C.4 for details), then you can remove them manually. 
3. Apply the learned merge list step-by-step to a given corpus, recording the frequency of all possible merges at each step. This will create a category subdir (e.g., `experiments/llm_tokenizers/gpt4o/en`) and `all_pair_counts.json` and `meta.json` inside the category subdir. This needs to be run once for every possible data category; see `script_examples/get_merge_frequencies_llm.sh` for an example script.
4. Apply the attack by running `python run_solver.py <output_dir>`. The final predictions can be found in `solution_[options].json` where `[options]` contains the solver parameters.

# Citation
```
@misc{hayase-etal-2024-data,
      title={Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data?}, 
      author={Jonathan Hayase and Alisa Liu and Yejin Choi and Sewoong Oh and Noah A. Smith},
      year={2024},
      eprint={2407.16607},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.16607}, 
}
```
