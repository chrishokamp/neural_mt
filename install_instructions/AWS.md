# Installing from an AWS image  

For AWS, we installed an image Vasco Pinho found

    spawned from : ubuntu14.04-mkl-cuda-dl (ami-03e67874) in region eu-west-1 

The default installation uses zsh. The old Montreal NMT system (groundhog) 

    https://github.com/lisa-groundhog/GroundHog

could be run directly with this config, although sizes have to be reduced to fit
into the GPU.

**Install Anaconda and cutting edge theano for Blocks+Fuel**

To run blocks+Fuel install Anaconda. 

**DANGER**: this will create a separate python install but still will spoil the
Groundhog installation. We are not sure why, but a possible explanation is that
both installations share the same .theanorc and .theano/ folders. Theano in
Groundhog will work again of the same fix in ~/.theanorc as for Blocks is
applied (see below). However, hdl5 wont work anymore.

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

Blocks

    pip install git+git://github.com/mila-udem/blocks.git \
    -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt

Now upgrade

    pip install git+git://github.com/mila-udem/blocks.git \
        -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt --upgrade

At the current version theano will die unless we use *fast_compile*. Also Blocks leaves variables without use that will kill the eval process. The ~/.theanorc has to be therefore configured as follows 

    [global]
    device = gpu0
    floatX = float32
    #optimizer_including = cudnn
    # This is needed for this AWS install, otherwise theano dies 
    optimizer = fast_compile
    # This downgrades "unused input variable error" to a warning. Needed to run
    # eval on the MT training
    on_unused_input = warn
    
    [blas]
    ldflags = -lmkl_rt
    
    [nvcc]
    fastmath = True
