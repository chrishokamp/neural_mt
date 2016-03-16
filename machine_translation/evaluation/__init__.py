import tempfile
import subprocess
import os

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")


def mteval_13(source_file, reference_file, hypothesis_file, src_lang='en', trg_lang='de', engine='nmt',
              script_dir='/home/chris/projects/neural_mt/scripts/scoring/'):
    '''
    Calling WMT Perl script for the evaluation
    :param source_file: source SGML file
    :param reference_file: reference SGML file
    :param decoded_file: mt-translated SGML file
    :return: list of sentence level BLEU for each segment
    '''

    # WORKING: delete the named temporary files created in this function

    wrap_xml_script = os.path.join(script_dir, 'wrap-xml-modified.perl')

    flags = ['src', 'ref', 'tst']
    inps_and_flags = zip([source_file, reference_file, hypothesis_file], flags)

    wrapped_files = []
    for filename, flag in inps_and_flags:
        # the following logic assumes 'src' is the first item in the list, otherwise it will break
        if flag == 'src':
            """For the source, just pass the source filename"""
            source_sgml = filename
            # create the named temporary file that will hold the output sgml file
            source_sgm_file = tempfile.NamedTemporaryFile(delete=False)
            source_sgm_name = source_sgm_file.name
            output_sgm_name = source_sgm_name
            language = src_lang

         # TODO: why were they providing different sgml sources for ref and target?
        elif flag == 'ref':
#             base_fname = '.'.join(fname.split('.')[:-1])
#         source_sgml = "%s.%s.sgm" % (base_fname, src_lang)
            source_sgml = source_sgm_name
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            output_sgm_name = temp_file.name
            ref_sgm_name = output_sgm_name
            language = trg_lang

        else:
            source_sgml = source_sgm_name
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            output_sgm_name = temp_file.name
            hyp_sgm_file = output_sgm_name
            language = trg_lang

#         write the new sgm file
        wrap_xml_cmd = ['perl', wrap_xml_script, language, source_sgml, engine, flag]
        logger.debug(' '.join(wrap_xml_cmd) + '< ' + filename)
        with open(filename, 'r+') as f:
            fname_stdin = f.read()

        with open(output_sgm_name, 'w+') as f:
            p = subprocess.Popen(wrap_xml_cmd, stdout=f, stdin=subprocess.PIPE)
            p.communicate(input=fname_stdin)[0]

    # now compute the segment-level BLEU scores
    mteval_2013_script = os.path.join(script_dir, 'mteval-v13a.pl')
#     mteval_2013_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'mteval-v13a.pl')
#     mteval_cmd = ['perl', mteval_2013_script, '-r', reference_file, '-s',
#                   source_file, '-t', decoded_file, '-b', '-d', '2']
    mteval_cmd = ['perl', mteval_2013_script, '-r', ref_sgm_name, '-s',
                  source_sgm_name, '-t', hyp_sgm_file, '-b', '-d', '2']
    logger.debug(' '.join(mteval_cmd))
    mteval_proc = subprocess.Popen(mteval_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
    stdout_data = mteval_proc.communicate()[0]

    # now parse stdout_data to get the scores and do asserts to make sure parsing is correct
    per_seg_lines = stdout_data.split('\n\n')[1]
    per_seg_lines = per_seg_lines.split('\n')

    # the last two lines are corpus-level info
    per_seg_lines = per_seg_lines[:-2]
    bleu_scores = [float(l.split()[5]) for l in per_seg_lines]

    # this is because we will use the scores in a func that we want to minimize
    one_minus_bleu_scores = [1.-v for v in bleu_scores]

    with open(hypothesis_file) as inp:
        num_segments = len(inp.read().strip().split('\n'))

    assert len(bleu_scores) == num_segments, "We must get one score for each segment"

    return one_minus_bleu_scores


# sentence level bleu scoring
# TODO: we really shouldn't need to pass the source here, this is because of the mteval_v13 interface
def sentence_level_bleu(src, ref, samples):
    # create temporary files
    num_samples = len(samples)
    src_file = tempfile.NamedTemporaryFile(delete=False)
    ref_file = tempfile.NamedTemporaryFile(delete=False)
    trg_file = tempfile.NamedTemporaryFile(delete=False)

    # Note that we need to map to strings here because the output of the model is ints
    with open(src_file.name, 'wb') as out:
        for _ in range(num_samples):
            out.write(' '.join([str(i) for i in src]) + '\n')
    with open(ref_file.name, 'wb') as out:
        for _ in range(num_samples):
            out.write(' '.join([str(i) for i in ref]) + '\n')
    with open(trg_file.name, 'wb') as out:
        for s in samples:
            out.write(' '.join([str(i) for i in s]) + '\n')

    scores = mteval_13(src_file.name, ref_file.name, trg_file.name)

    return scores
