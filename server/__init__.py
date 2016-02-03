import logging
import os
import threading

from flask import Flask, request, render_template
from wtforms import Form, TextAreaField, validators

logger = logging.getLogger(__name__)

app = Flask(__name__)
# this needs to be set before we actually run the server
app.predictor = None

path_to_this_dir = os.path.dirname(os.path.abspath(__file__))
app.template_folder = os.path.join(path_to_this_dir, 'templates')

lock = threading.Lock()


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

        output_text = ''
        translations, costs = app.predictor.predict_segment(source_sentence, tokenize=True, detokenize=True)
        for hyp in translations:
            output_text += hyp + '\n'

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

