def get_config_es2en():
    config = {}

    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded
    config['seq_len'] = 30

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 500
    config['dec_nhids'] = 500

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 250 
    config['dec_embed'] = 250

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'search_model_es2en'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 100 

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 0.5

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = '/ichec/work/dcu01/chokamp/data/'

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    config['src_vocab'] = datadir + 'vocab.en-es.en.pkl'
    config['trg_vocab'] = datadir + 'vocab.en-es.es.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'news-commentary-v10.en-es.en.tok.shuf'
    config['trg_data'] = datadir + 'news-commentary-v10.en-es.es.tok.shuf'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 20000
    config['trg_vocab_size'] = 20000

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'

    # Early stopping based on bleu related ------------------------------------

    # Normalize cost according to sequence length after beam-search
    config['normalized_bleu'] = True

    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = datadir + 'multi-bleu.perl'

    # Validation set source file
    config['val_set'] = datadir + 'newstest2013.en.tok'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'newstest2013.es.tok'

    # Print validation output to file
    config['output_val_set'] = True

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Beam-size
    config['beam_size'] = 12

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 500

    # Show samples from model after this many updates
    config['sampling_freq'] = 500 

    # Show this many samples at each sampling
    config['hook_samples'] = 5

    # Validate bleu after this many updates
    config['bleu_val_freq'] = 5000

    # Start bleu validation after this many updates
    config['val_burn_in'] = 1000

    return config
