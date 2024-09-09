# Preparing data for controlled experiments

## Oscar data
Download [Oscar-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301). You can do this using the following command:

```
huggingface-cli download oscar-corpus/OSCAR-2301 --repo-type dataset --local-dir .
```

Then, run the preprocessing script

```
python -m preprocessing.preprocess_oscar
```

## RedPajama data
Download [RedPajama-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T). You can do this using the commands in `script_examples/download_rpj.sh`. Note that we only download a subset of the Common Crawl files to save space. The ones we used can be found in `preprocessing/cc_urls.txt`.

To preprocess each split, use `notebooks/preprocess_redpajama.ipynb`.
