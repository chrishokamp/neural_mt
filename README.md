# Unbabel's version of Montreal's Theano Neural-MT Model  

**Install**

Follow the instructions matching your platform at

    install_instructions

**Training a Sample Model**

There is a sample experiment with very small data set up in `test_data/sample_experiment/tiny_demo_dataset`. The 
configuration file for this experiment is in `experiment/configs/sample_nmt_config.yaml`. If your environment is set
up correctly, you can run:
```
python -m machine_translation experiments/configs/sample_nmt_config.yaml
```
to make sure things are working. The paths in this demo are relative, so you need to run it from inside the
`neural_mt` directory.

**Training a Real Model**

There are two steps to the  process: (1) Prepare your training and validation experiments (2) train your model

(1) Preparing the data          
Download and preprocess the default data with vocab size = 30000. Make sure you
have the source and target language codes correct!

    python machine_translation/prepare_data.py -s es -t en --source-dev newstest2013.es --target-dev newstest2013.en --source-vocab 30000 --target-vocab 30000

Edit machine_translation/configurations.py with your desired parameters, again,
be careful about the language codes

(2) Training the model            

Be sure your ~/.theanorc is configured correctly (see install_instructions). 

If you are using a GPU, check the memory available with 

    watch -d nvidia-smi

The default gpu should be zero. To select another GPU e.g. number 3 pre-apend

    export THEANO_FLAGS='device=gpu3'

to your call

    python -m machine_translation <path_to_configuration_file.yaml> 2&>1 | tee -a log.out 

When using remote machines it is useful to launch the jobs from byobu, tmux or
similar

**Notes and Gotchas**

- The `prepare_data.py` script tries to be smart about finding and extracting
  the data for your language pair, but it's not smart enough, because it
  doesn't find reverse pairs. I.e. if `es-en` exists, it doesn't know that you
  implicitly have `en-es` (debates on translation direction aside). Therefore,
  you may need to rename some files in data/tmp/train_data.tgz to make things
  work.

- the default configuration requires too much memory for a 4GB GPU -- the
  params that you need to change are: 
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

- the `config['vocab_size']` parameter also impacts the memory requirements,
  but you need to make sure that it corresponds to your settings for the
  `prepare_data.py` script (see above).
  
- we have also added support for using custom vocabularies from another file or set of files. This is useful if 
you want to use the parameters of the trained NMT model to initialize another model, such as a Quality Estimation model, 
and you need the vocabularies to match. See `machine_translation/prepare_data.py` for the available dataset preparation
  commands.

**Commands for monitoring your experiments**

- access the shared monitoring screens (Google for basic byoubiu commands)
`byobu` 

- monitor GPU usage
`watch -d nvidia-smi`
