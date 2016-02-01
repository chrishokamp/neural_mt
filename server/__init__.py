import logging
import os

from flask import Flask, request, render_template
from wtforms import Form, TextAreaField, validators

from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

import threading

import json

app = Flask(__name__)
# this needs to be set before we actually run the server
app.predictor = None

path_to_this_dir = os.path.dirname(os.path.abspath(__file__))
# TODO: the language params of the tokenizer and detokenizer need to be configurable
app.tokenizer_cmd = [os.path.join(path_to_this_dir, 'tokenizer.perl'), '-l', 'en', '-q', '-', '-no-escape', '1']
app.detokenizer_cmd = [os.path.join(path_to_this_dir, 'detokenizer.perl'), '-l', 'es', '-q', '-']
app.template_folder = os.path.join(path_to_this_dir, 'templates')

lock = threading.Lock()

def map_idx_or_unk(sentence, index, unknown_token='<UNK>'):
    if type(sentence) is str:
        sentence = sentence.split()
    return [index.get(w, unknown_token) for w in sentence]

class NeuralMTDemoForm(Form):
    sentence = TextAreaField('Write the sentence here:',
                             [validators.Length(min=1, max=100000)])
    target_text = ''


@app.route('/neural_MT_demo', methods=['GET', 'POST'])
def neural_mt_demo():
    form = NeuralMTDemoForm(request.form)
    if request.method == 'POST' and form.validate():

        # source_text = form.sentence.data.encode('utf-8')
        source_text = form.sentence.data # Problems in Spanish with 'A~nos. E'
        logger.info('Acquired lock')
        lock.acquire()

        source_sentence = source_text.encode('utf-8')
        tokenizer = Popen(app.tokenizer_cmd, stdin=PIPE, stdout=PIPE)
        sentence, _ = tokenizer.communicate(source_sentence)
        logger.info('original source: {}'.format(source_sentence))
        logger.info('tokenized source: {}'.format(sentence))

        # now map into idxs
        mapped_sentence = map_idx_or_unk(sentence, app.predictor.src_vocab, app.predictor.unk_idx)
        # add EOS index
        mapped_sentence += [app.predictor.src_eos_idx]
        logger.info('mapped source: {}'.format(mapped_sentence))

        # WORKING: support n-best
        output_text = ''
        translations, costs = app.predictor.predict_segment(mapped_sentence, n_best=1)
        for hyp in translations:
            logger.info('raw translation: {}'.format(hyp))

            # detokenize the translation
            detokenizer = Popen(app.detokenizer_cmd, stdin=PIPE, stdout=PIPE)
            detokenized_sentence, _ = detokenizer.communicate(hyp)
            output_text += detokenized_sentence + '\n'

        form.target_text = output_text.decode('utf-8')
        logger.info('detokenized translations:\n {}'.format(output_text))

        print "Lock release"
        lock.release()

    return render_template('neural_MT_demo.html', form=form)


def run_nmt_server(predictor, port=5000):
    # TODO: make the indexing API visible via the predictor
    app.predictor = predictor

    logger.info('Server starting on port: {}'.format(port))
    logger.info('navigate to: http://localhost:{}/neural_MT_demo to see the system demo'.format(port))
    app.run(debug=True, port=port, host='127.0.0.1')

