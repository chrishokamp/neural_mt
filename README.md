** Setting up experiments **

```
git clone https://github.com/Unbabel/neural_mt.git
cd machine_translation

# downloading and preprocessing the default data with vocab size = 30000
python prepare_data.py -s es -t en --source-dev newstest2013.es --target-dev newstest2013.en --source-vocab 30000 --target-vocab 30000

# run an experiment
export THEANO_FLAGS='device=gpu3, on_unused_input=warn'
python -m machine_translation
```


**Here Define Image and Moduli instalation for AWS**

    For AWS, we installed an image Vasco Pinho found, we need to ask about the details

