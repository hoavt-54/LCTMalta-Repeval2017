# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import re
import numpy
import tensorflow as tf

from vocab_utils import Vocab
from dataStream.py import SentenceMatchDataStream
from graphModel import SentenceMatchModelGraph
import namespace_utils

FLAGS = None

def collect_vocabs(train_path, with_dep=True, with_pos=True):
    all_labels = set()
    all_words = set()
    all_deps = set()
    all_pos = set()
    infile = open(train_path, 'rt')
    count = 0
    for line in infile:
        try:    
            count +=1
            #if(count) > 10000: break
            line = line.decode('utf-8').strip()
            if line.startswith('-'): continue
            items = re.split("\t", line)
            label = items[0]
            if(len(label) < 4): continue
            sentence1 = re.split("\\s+",items[1].lower())
            sentence2 = re.split("\\s+",items[4].lower())
            all_labels.add(label)
            all_words.update(sentence1)
            all_words.update(sentence2)
        
            if with_dep:
                deps1 = [pair.split('-')[1] for pair in re.split("\\s+",items[2])]
                all_deps.update(deps1)
        
            if with_pos:
                pos1 = items[2]# pos1 is string
                all_pos.update(pos1.split())
        
            #print (deps1)
            #deps2 = [pair.split('-')[1] for pair in re.split("\\s+",items[4])]

        except:
            print (line)
    infile.close()

    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)

    return (all_words, all_chars, all_labels, all_deps, all_pos)

def evaluate(dataStream, valid_graph, sess, outpath=None, label_vocab=None, mode='prediction',char_vocab=None):
    if outpath is not None: 
        outfile = open(outpath, 'wt')
        outfile.write('pairID,gold_label\n')
    total_tags = 0.0
    correct_tags = 0.0
    dataStream.reset()
    idx=0
    for batch_index in xrange(dataStream.get_num_batch()):
        cur_dev_batch = dataStream.get_batch(batch_index)
        (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch, 
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, 
                                 sent2_length_batch, sent1_char_length_batch, sent2_char_length_batch, 
                                 dep_pos1_batch, dep_pos2_batch, dep_label1_batch, dep_label2_batch,
                                 idep_label1_batch, idep_label2_batch, idep_pos1_batch, idep_pos2_batch,
                                 #idep_pos1_batch, idep_pos2_batch, idep_label1_batch, idep_label2_batch,
                                 pos1_idx_batch, pos2_idx_batch, pairid_batch) = cur_dev_batch
        feed_dict = {
                    valid_graph.get_truth(): label_id_batch, 
                    valid_graph.get_question_lengths(): sent1_length_batch, 
                    valid_graph.get_passage_lengths(): sent2_length_batch, 
                    valid_graph.get_in_question_words(): word_idx_1_batch, 
                    valid_graph.get_in_passage_words(): word_idx_2_batch,
                    #valid_graph.get_in_question_dependency(): dependency1_batch,
                    #valid_graph.get_in_passage_dependency(): dependency2_batch,
#                     valid_graph.get_question_char_lengths(): sent1_char_length_batch, 
#                     valid_graph.get_passage_char_lengths(): sent2_char_length_batch, 
#                     valid_graph.get_in_question_chars(): char_matrix_idx_1_batch, 
#                     valid_graph.get_in_passage_chars(): char_matrix_idx_2_batch, 
                }
        if FLAGS.with_dep:
            feed_dict[valid_graph.get_in_question_dependency_pos()] = dep_pos1_batch
            feed_dict[valid_graph.get_in_passage_dependency_pos()] = dep_pos2_batch
            feed_dict[valid_graph.get_in_question_dependency_label()] = dep_label1_batch
            feed_dict[valid_graph.get_in_passage_dependency_label()] = dep_label2_batch
        
        if FLAGS.with_idep:
            feed_dict[valid_graph.get_in_question_idependency_pos()] = idep_pos1_batch
            feed_dict[valid_graph.get_in_passage_idependency_pos()] = idep_pos2_batch
            feed_dict[valid_graph.get_in_question_idependency_label()] = idep_label1_batch
            feed_dict[valid_graph.get_in_passage_idependency_label()] = idep_label2_batch
        
        if FLAGS.with_pos:
            feed_dict[valid_graph.get_in_pos1()] = pos1_idx_batch
            feed_dict[valid_graph.get_in_pos2()] = pos2_idx_batch 

        if char_vocab is not None:
            feed_dict[valid_graph.get_question_char_lengths()] = sent1_char_length_batch
            feed_dict[valid_graph.get_passage_char_lengths()] = sent2_char_length_batch
            feed_dict[valid_graph.get_in_question_chars()] = char_matrix_idx_1_batch
            feed_dict[valid_graph.get_in_passage_chars()] = char_matrix_idx_2_batch

        if mode is not 'test_prediction':
            total_tags += len(label_batch)
            correct_tags += sess.run(valid_graph.get_eval_correct(), feed_dict=feed_dict)
        if outpath is not None:
            if mode =='test_prediction':
                predictions = sess.run(valid_graph.get_predictions(), feed_dict=feed_dict)
                #for i in xrange(len(sent1_batch)):
                for idx, pairID in enumerate(pairid_batch):
                    outline = pairID + "," + label_vocab.getWord(predictions[idx]) + "\n"
                    outfile.write(outline.encode('utf-8'))
            else:
                probs = sess.run(valid_graph.get_prob(), feed_dict=feed_dict)
                for i in xrange(len(label_batch)):
                    outfile.write(label_batch[i] + "\t" + output_probs(probs[i], label_vocab) + "\n")

    if outpath is not None: outfile.close()
    if mode is not 'test_prediction':
        accuracy = correct_tags / total_tags * 100
        return accuracy

def output_probs(probs, label_vocab):
    out_string = ""
    for i in xrange(probs.size):
        out_string += " {}:{}".format(label_vocab.getWord(i), probs[i])
    return out_string.strip()

def main(_):
    print('Configurations:')
    print(FLAGS)

    train_path = FLAGS.train_path
    mis_dev_path = FLAGS.mis_dev_path
    matched_dev_path = FLAGS.matched_dev_path
    matched_test_path = FLAGS.matched_test_path
    mis_test_path = FLAGS.mis_test_path
    word_vec_path = FLAGS.word_vec_path
    char_vec_path = FLAGS.char_vec_path
    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    path_prefix = log_dir + "/SentenceMatch.{}".format(FLAGS.suffix)

    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    # build vocabs
    word_vocab = Vocab(word_vec_path, fileformat='txt3') #fileformat='txt3'
    matched_best_path = path_prefix + 'matched.best.model'
    mis_best_path = path_prefix + 'mis.best.model'
    char_path = path_prefix + ".char_vocab"
    label_path = path_prefix + ".label_vocab"
    DEP_path = path_prefix + ".DEP_vocab"
    has_pre_trained_model = False
    DEP_vocab, pos_vocab = None, None
    if os.path.exists(matched_best_path) and False: # no need best path
        has_pre_trained_model = True
        label_vocab = Vocab(label_path, fileformat='txt2')
        #char_vocab = Vocab(char_path, fileformat='txt2')
    else:
        print('Collect words, chars and labels ...')
        (all_words, all_chars, all_labels, all_deps, all_pos) = collect_vocabs(train_path, with_dep=FLAGS.with_dep, with_pos=FLAGS.with_pos)
        print('Number of words: {}'.format(len(all_words)))
        print('Number of labels: {}'.format(len(all_labels)))
        label_vocab = Vocab(fileformat='voc', voc=all_labels,dim=2)
        label_vocab.dump_to_txt2(label_path)

        print('Number of chars: {}'.format(len(all_chars)))
        char_vocab = Vocab(fileformat='voc', voc=all_chars,dim=FLAGS.char_emb_dim)
        char_vocab = Vocab(char_vec_path, fileformat='txt3') #fileformat='txt3'
        char_vocab.dump_to_txt2(char_path)
        
        if FLAGS.with_dep:
            print('Number of dependencys: {}'.format(len(all_deps)))
            DEP_vocab = Vocab(fileformat='voc', voc=all_deps, dim=FLAGS.dep_emb_dim)
            print('DEP vocab size: {}'.format(len(DEP_vocab.word_vecs)))
            DEP_vocab.word_vecs[DEP_vocab.vocab_size].fill(0)
            #NER_vocab.dump_to_txt2(DEP_path)
        if FLAGS.with_pos:
            print('Number of pos tags: {}'.format(len(all_pos)))
            pos_vocab = Vocab(fileformat='voc', voc=all_deps, dim=20)

    print('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
    print('tag_vocab shape is {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    print('Build SentenceMatchDataStream ... ')
    print('Reading trainDataStream')
    trainDataStream = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=label_vocab, 
                                                dep_vocab = DEP_vocab, pos_vocab=pos_vocab, batch_size=FLAGS.batch_size, isShuffle=True, 
                                                isLoop=True, isSort=True, max_char_per_word=FLAGS.max_char_per_word, 
                                                max_sent_length=FLAGS.max_sent_length, with_dep=FLAGS.with_dep, with_pos=FLAGS.with_pos)
                        
    print('Reading devDataStream')
    matched_devDataStream = SentenceMatchDataStream(matched_dev_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=label_vocab, 
                                                dep_vocab = DEP_vocab, pos_vocab=pos_vocab, batch_size=FLAGS.batch_size, 
                                                isShuffle=False, isLoop=True, isSort=True, max_char_per_word=FLAGS.max_char_per_word, 
                                                max_sent_length=FLAGS.max_sent_length, with_dep=FLAGS.with_dep, with_pos=FLAGS.with_pos)

    mis_devDataStream = SentenceMatchDataStream(mis_dev_path, word_vocab=word_vocab, char_vocab=char_vocab,label_vocab=label_vocab, 
                                                dep_vocab = DEP_vocab, pos_vocab=pos_vocab, batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, 
                                                isSort=True, max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length, 
                                                with_dep=FLAGS.with_dep, with_pos=FLAGS.with_pos)

    mis_testDataStream = SentenceMatchDataStream(mis_test_path, word_vocab=word_vocab, char_vocab=char_vocab,label_vocab=label_vocab, 
                                                dep_vocab = DEP_vocab, pos_vocab=pos_vocab, batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, 
                                                isSort=True, max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length, 
                                                with_dep=FLAGS.with_dep, with_pos=FLAGS.with_pos)

    print('Reading testDataStream')
    matched_testDataStream = SentenceMatchDataStream(matched_test_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=label_vocab, 
                                                dep_vocab = DEP_vocab, pos_vocab=pos_vocab, batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=True, 
                                                max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length, 
                                                with_dep=FLAGS.with_dep, with_pos=FLAGS.with_pos)

    print('save cache file')
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(matched_devDataStream.get_num_instance()))
    print('Number of instances in testDataStream: {}'.format(matched_testDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of batches in devDataStream: {}'.format(matched_devDataStream.get_num_batch()))
    print('Number of batches in testDataStream: {}'.format(matched_testDataStream.get_num_batch()))
    
    sys.stdout.flush()
    if FLAGS.wo_char: char_vocab = None

    matched_best_accuracy = 0.0
    mis_best_accuracy = 0.0
    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, 
                    dep_vocab = DEP_vocab, pos_vocab=pos_vocab, dropout_rate=FLAGS.dropout_rate, 
                    learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type, lambda_l2=FLAGS.lambda_l2, 
                    char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim, 
                    context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num, 
                    fix_word_vec=FLAGS.fix_word_vec, aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=True, 
                    with_dep=FLAGS.with_dep, share_params=FLAGS.share_params, dep_emb_dim=FLAGS.dep_emb_dim, with_pos=FLAGS.with_pos, with_idep=FLAGS.with_idep)

            tf.summary.scalar("Training Loss", train_graph.get_loss()) # Add a scalar summary for the snapshot loss.
        
#         with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, dep_vocab = DEP_vocab, 
                 pos_vocab=pos_vocab, dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                 lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim, 
                 aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False, context_layer_num=FLAGS.context_layer_num, 
                 aggregation_layer_num=FLAGS.aggregation_layer_num, fix_word_vec=FLAGS.fix_word_vec,  with_dep=FLAGS.with_dep, 
                 share_params=FLAGS.share_params, dep_emb_dim=FLAGS.dep_emb_dim, with_pos=FLAGS.with_pos, with_idep=FLAGS.with_idep)

                
        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.all_variables():
            print(var.name)
            if "word_embedding" in var.name: continue
#             if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        matched_saver = tf.train.Saver(vars_)
        mis_saver = tf.train.Saver(vars_)
        sess = tf.Session()
        sess.run(initializer)
        if has_pre_trained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")

        print('Start the training loop.')
        train_size = trainDataStream.get_num_batch()
        max_steps = train_size * FLAGS.max_epochs
        total_loss = 0.0
        start_time = time.time()
        for step in xrange(max_steps):
            # read data
            cur_batch = trainDataStream.nextBatch()
            (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch, 
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, 
                                 sent2_length_batch, sent1_char_length_batch, sent2_char_length_batch, 
                                 dep_pos1_batch, dep_pos2_batch, dep_label1_batch, dep_label2_batch, 
                                 #idep_pos1_batch, idep_pos2_batch, idep_label1_batch, idep_label2_batch,
                                 idep_label1_batch, idep_label2_batch, idep_pos1_batch, idep_pos2_batch,
                                 pos1_idx_batch, pos2_idx_batch, pairid_batch) = cur_batch
            feed_dict = {
                         train_graph.get_truth(): label_id_batch, 
                         train_graph.get_question_lengths(): sent1_length_batch, 
                         train_graph.get_passage_lengths(): sent2_length_batch, 
                         train_graph.get_in_question_words(): word_idx_1_batch, 
                         train_graph.get_in_passage_words(): word_idx_2_batch,
                         }

            if FLAGS.with_dep:
                feed_dict[train_graph.get_in_question_dependency_pos()] = dep_pos1_batch
                feed_dict[train_graph.get_in_passage_dependency_pos()] = dep_pos2_batch
                feed_dict[train_graph.get_in_question_dependency_label()] = dep_label1_batch 
                feed_dict[train_graph.get_in_passage_dependency_label()] = dep_label2_batch
            
            if FLAGS.with_idep:
                numpy.set_printoptions(threshold=numpy.nan)
                #print ('idep_pos1_batch', idep_pos1_batch)

                feed_dict[train_graph.get_in_question_idependency_pos()] = idep_pos1_batch
                feed_dict[train_graph.get_in_passage_idependency_pos()] = idep_pos2_batch
                feed_dict[train_graph.get_in_question_idependency_label()] = idep_label1_batch 
                feed_dict[train_graph.get_in_passage_idependency_label()] = idep_label2_batch

            if FLAGS.with_pos:
                feed_dict[train_graph.get_in_pos1()] = pos1_idx_batch
                feed_dict[train_graph.get_in_pos2()] = pos2_idx_batch 

            if char_vocab is not None:
                feed_dict[train_graph.get_question_char_lengths()] = sent1_char_length_batch
                feed_dict[train_graph.get_passage_char_lengths()] = sent2_char_length_batch
                feed_dict[train_graph.get_in_question_chars()] = char_matrix_idx_1_batch
                feed_dict[train_graph.get_in_passage_chars()] = char_matrix_idx_2_batch


            _, loss_value = sess.run([train_graph.get_train_op(), train_graph.get_loss()], feed_dict=feed_dict)
            total_loss += loss_value
            
            if step % 100==0: 
                print('{} '.format(step), end="")
                sys.stdout.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % trainDataStream.get_num_batch() == 0 or (step + 1) == max_steps:
                print()
                # Print status to stdout.
                duration = time.time() - start_time
                start_time = time.time()
                print('Step %d: loss = %.2f (%.3f sec)' % (step, total_loss, duration))
                total_loss = 0.0

                # Evaluate against the validation set.
                print('Validation Data Eval:')
                matched_accuracy = evaluate(matched_devDataStream, valid_graph, sess,char_vocab=char_vocab)
                print("Current accuracy on matched dev is %.2f" % matched_accuracy)
                mis_accuracy = evaluate(mis_devDataStream, valid_graph, sess,char_vocab=char_vocab)
                print("Current accuracy on mismatched dev is %.2f" % mis_accuracy)
                #accuracy_train = evaluate(trainDataStream, valid_graph, sess,char_vocab=char_vocab)
                #print("Current accuracy on train is %.2f" % accuracy_train)
                if matched_accuracy > matched_best_accuracy:
                    matched_best_accuracy = matched_accuracy
                    matched_saver.save(sess, matched_best_path)
                if mis_accuracy > mis_best_accuracy:
                    mis_best_accuracy = mis_accuracy
                    mis_saver.save(sess, mis_best_path)

    print("Best accuracy on matched dev set is %.2f" % matched_best_accuracy)
    print("Best accuracy on mis dev set is %.2f" % mis_best_accuracy)
    # decoding
    print('Decoding on the test set:')
    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, 
                    dep_vocab = DEP_vocab, pos_vocab=pos_vocab, dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, 
                    optimize_type=FLAGS.optimize_type, lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, 
                    context_lstm_dim=FLAGS.context_lstm_dim, aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, 
                    is_training=False, context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num, 
                    fix_word_vec=FLAGS.fix_word_vec, with_dep=FLAGS.with_dep, share_params=FLAGS.share_params, 
                    dep_emb_dim=FLAGS.dep_emb_dim, with_pos=FLAGS.with_pos, with_idep=FLAGS.with_idep)
        
        vars_ = {}
        for var in tf.all_variables():
            print(var.name)
            if "word_embedding" in var.name: continue
            #if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        

        saver = tf.train.Saver(vars_)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        saver.restore(sess, mis_best_path)
        accuracy = evaluate(mis_testDataStream, valid_graph, sess,char_vocab=char_vocab, label_vocab=label_vocab, outpath=FLAGS.suffix + '_mismatched_results.txt', mode='test_prediction')
        #print("Accuracy for test set is %.2f" % accuracy)
        print("predicting mismatched set")

        saver = tf.train.Saver(vars_)        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        saver.restore(sess, matched_best_path)

        accuracy = evaluate(matched_testDataStream, valid_graph, sess,char_vocab=char_vocab, label_vocab=label_vocab, outpath=FLAGS.suffix + '_matched_results.txt', mode='test_prediction')
        #print("Accuracy for test set is %.2f" % accuracy)
        print("predicting matched set")
        accuracy_train = evaluate(trainDataStream, valid_graph, sess,char_vocab=char_vocab)
        print("Accuracy for train set is %.2f" % accuracy_train)
def set_args(config_file, FLAGS):
    import json
    with open(config_file, 'r') as f:
        config = json.load(f)
        for key in config:
            setattr(FLAGS, key, config[key])
        return FLAGS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='Path to the train set.')
    parser.add_argument('--matched_dev_path', type=str, help='Path to the dev set.')
    parser.add_argument('--mis_dev_path', type=str, help='Path to the mismatched dev set.')
    parser.add_argument('--mis_test_path', type=str, help='Path to the test set.')
    parser.add_argument('--matched_test_path', type=str, help='Path to the test set.')
    parser.add_argument('--word_vec_path', type=str, help='Path the to pre-trained word vector model.')
    parser.add_argument('--char_vec_path', type=str, help='Path the to pre-trained char vector model.')
    parser.add_argument('--model_dir', type=str, help='Directory to save model files.')
    parser.add_argument('--batch_size', type=int, default=60, help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs for training.')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--char_lstm_dim', type=int, default=100, help='Number of dimension for character-composed embeddings.')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=100, help='Number of dimension for aggregation layer.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--aggregation_layer_num', type=int, default=1, help='Number of LSTM layers for aggregation layer.')
    parser.add_argument('--context_layer_num', type=int, default=1, help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--suffix', type=str, default='normal',  help='Suffix of the model name.')
    parser.add_argument('--fix_word_vec', default=False, help='Fix pre-trained word embeddings during training.', action='store_true')
    parser.add_argument('--wo_char', default=False, help='Without character-composed embeddings.', action='store_true')
    parser.add_argument('--config_file', default='bilstm_dep.config', help='path to config file')
    parser.add_argument('--with_dep', default=True, help='indicate whether we use dependency')
    parser.add_argument('--dep_emb_dim', type=int, default=200, help='Number of dimension for dependency embeddings.')
    parser.add_argument('--beginning', default=False, help='indicate whether we add 2 beginning tokens to connect dependency')
    parser.add_argument('--share_params', default=True, help='indicate whether we add 2 beginning tokens to connect dependency')
    #share_params
    #print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    sys.stdout.flush()
    FLAGS, unparsed = parser.parse_known_args()
    # read from config file
    FLAGS = set_args(FLAGS.config_file, FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

