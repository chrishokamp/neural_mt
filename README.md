** Setting up experiments **

```
git clone https://github.com/Unbabel/neural_mt.git
cd neural_mt

# downloading and preprocessing the default data with vocab size = 30000
python machine_translation/prepare_data.py -s es -t en --source-dev newstest2013.es --target-dev newstest2013.en --source-vocab 30000 --target-vocab 30000

# now edit machine_translation/configurations.py with your desired parameters

# run an experiment
export THEANO_FLAGS='device=gpu3, on_unused_input=warn'
python -m machine_translation 2&>1 | tee -a log.out 
```


**Here Define Image and Moduli instalation for AWS**

For AWS, we installed an image Vasco Pinho found, we need to ask about the details

TODO: add Anaconda, blocks+Fuel setup instructions and scripts

** byobu commands for monitoring your experiments **

- monitor GPU usage
`watch -d nvidia-smi`

** Notes and Gotchas **
- The `prepare_data.py` script tries to be smart about finding and extracting the data for your language pair, but it's not smart enough, because it doesn't find reverse pairs. I.e. if `es-en` exists, it doesn't know that you implictly have `en-es` (debates on translation direction aside). Therefore, you may need to rename some files in data/tmp/train_data.tgz to make things work.

- the default configuration require too much memory for a 4GB GPU -- the params that you need to change are: 
```
    # Sequences longer than this will be discarded
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620

```

For example, a working configuration is:

```
    # Sequences longer than this will be discarded
    config['seq_len'] = 40

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 600
    config['dec_nhids'] = 600

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 400
    config['dec_embed'] = 400
```


