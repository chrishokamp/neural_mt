# Instructions to deploy and run Montreal's Theano Neural-MT Model  

**Image and Moduli Installation for AWS**

For AWS, we installed an image Vasco Pinho found

    spawned from : ubuntu14.04-mkl-cuda-dl (ami-03e67874) in region eu-west-1 

The default installation uses zsh. The old Montreal NMT system (groundhog) 

    https://github.com/lisa-groundhog/GroundHog

could be run directly with this config, although sizes have to be reduced to fit
into the GPU.

**Commands for monitoring your experiments**

- access the shared monitoring screens (Google for basic byoubiu commands)
`byobu` 

- monitor GPU usage
`watch -d nvidia-smi`

**Install Anaconda and cutting edge theano for Blocks+Fuel**

To run blocks+Fuel install Anaconda. 

**DANGER**: this will create a separate python install but still will spoil the
Groundhog installation. We are not sure why, but a possible explanation is that
both installations share the same .theanorc and .theano/ folders. Theano in
Groundhog will work again of the same fix in ~/.theanorc as for Blocks is
applied (see below). However, hd5 wont work anymore.

**Important**: Switch from zsh to bash! 

Copy the cuda paths from ~/.zshrc into ~/.bashrc

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cudnn
    export LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cudnn
    export C_INCLUDE_PATH=/usr/local/cuda/include:/usr/local/cudnn
    export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:/usr/local/cudnn
    export LLVM_CONFIG=/usr/bin/llvm-config-3.5
    source /opt/intel/bin/compilervars.sh intel64
    alias sudo='sudo -H'

From Chris Hokamp's github instructions

    https://github.com/chrishokamp/python_deep_learning_stack_vm_setup/blob/master/install_python_deep_learning_stack.sh

Start in line 43

    wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.3.0-Linux-x86_64.sh
    bash Anaconda*.sh
    cd
    source .bashrc

Install bleeding edge theano
    git clone git://github.com/Theano/Theano.git
    cd Theano
    python setup.py develop --user
    cd ..

**Blocks**

make sure you're using Anaconda or similar

    pip install git+git://github.com/mila-udem/blocks.git \
    -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt

    # now upgrade
    pip install git+git://github.com/mila-udem/blocks.git \
        -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt --upgrade

At the current version theano will die unless we use *fast_compile*. This has
to be added to ~/.theanorc as 

    [global]
    device = gpu0
    floatX = float32
    #optimizer_including = cudnn
    optimizer = fast_compile
    
    [blas]
    ldflags = -lmkl_rt
    
    [nvcc]
    fastmath = True

**Setting up Blocks+Fuel Neural MT experiments**

```
git clone https://github.com/Unbabel/neural_mt.git
cd neural_mt

# downloading and preprocessing the default data with vocab size = 30000 -- Make sure you have the source and target language codes correct!
python machine_translation/prepare_data.py -s es -t en --source-dev newstest2013.es --target-dev newstest2013.en --source-vocab 30000 --target-vocab 30000

# now edit machine_translation/configurations.py with your desired parameters, again, be careful about the language codes

# run an experiment
export THEANO_FLAGS='device=gpu3, on_unused_input=warn'
python -m machine_translation 2&>1 | tee -a log.out 
```

**Notes and Gotchas**
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

- the `config['vocab_size']` parameter also impacts the memory requirements, but you need to make sure that it corresponds to your settings for the `prepare_data.py` script (see above).
