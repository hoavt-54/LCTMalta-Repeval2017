import tensorflow as tf
import my_rnn
import match_utils
#from count_sketch import count_sketch, bilinear_pool
from dataStream import pad_3d_tensor
class SentenceMatchModelGraph(object):
    def __init__(self, num_classes, word_vocab=None, char_vocab=None, dep_vocab=None, pos_vocab=None, 
                 dropout_rate=0.5, learning_rate=0.001, optimize_type='adam',lambda_l2=1e-5, with_word=True, 
                 with_char=True, char_lstm_dim=20, context_lstm_dim=200, aggregation_lstm_dim=200, dep_emb_dim =200, is_training=True,
                 context_layer_num=1, aggregation_layer_num=1, fix_word_vec=False, with_dep=True, with_pos=True, share_params=True, with_idep = True):

        # ======word representation layer======
        in_question_repres = [] # premise
        in_question_dep_cons = [] # premise dependency connections
        in_passage_repres = [] # hypothesis
        in_passage_dep_cons = [] # hypothesis dependency connections
        self.question_lengths = tf.placeholder(tf.int32, [None])
        self.passage_lengths = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None]) # [batch_size]
        input_dim = 0
        # word embedding
        if with_word and word_vocab is not None: 
            self.in_question_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
            self.in_passage_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
#             self.word_embedding = tf.get_variable("word_embedding", shape=[word_vocab.size()+1, word_vocab.word_dim], initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if fix_word_vec: 
                word_vec_trainable = False
                cur_device = '/cpu:0'
            with tf.device(cur_device):
                self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable, 
                                                  initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)
            #
            in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words) # [batch_size, question_len, word_dim]
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words) # [batch_size, passage_len, word_dim]
            
            #print (in_question_word_repres)
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)
            
            input_shape = tf.shape(self.in_question_words)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]
            input_dim += word_vocab.word_dim

        
        if with_pos:
            self.in_prem_POSs = tf.placeholder(tf.int32, [None, None])
            self.in_hypo_POSs = tf.placeholder(tf.int32, [None, None])
            self.pos_embedding = tf.get_variable("pos_embedding", initializer=tf.constant(pos_vocab.word_vecs), dtype=tf.float32)
            in_prem_pos_repres = tf.nn.embedding_lookup(self.pos_embedding, self.in_prem_POSs) # [batch_size, question_len, pos_dim]
            in_hypo_pos_repres = tf.nn.embedding_lookup(self.pos_embedding, self.in_hypo_POSs) # [batch_size, question_len, pos_dim]
            in_question_repres.append(in_prem_pos_repres)
            in_passage_repres.append(in_hypo_pos_repres)
            input_dim += pos_vocab.word_dim


        if with_char and char_vocab is not None: 
            self.question_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, question_len]
            self.passage_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
            self.in_question_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, question_len, q_char_len]
            self.in_passage_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, passage_len, p_char_len]
            input_shape = tf.shape(self.in_question_chars)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            q_char_len = input_shape[2]
            input_shape = tf.shape(self.in_passage_chars)
            passage_len = input_shape[1]
            p_char_len = input_shape[2]
            char_dim = char_vocab.word_dim
#             self.char_embedding = tf.get_variable("char_embedding", shape=[char_vocab.size()+1, char_vocab.word_dim], initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)
            self.char_embedding = tf.get_variable("char_embedding", initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)

            in_question_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_question_chars) # [batch_size, question_len, q_char_len, char_dim]
            in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, q_char_len, char_dim])
            question_char_lengths = tf.reshape(self.question_char_lengths, [-1])
            in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_passage_chars) # [batch_size, passage_len, p_char_len, char_dim]
            in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
            passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
            with tf.variable_scope('char_lstm'):
                # lstm cell
                char_lstm_cell = tf.contrib.rnn.BasicLSTMCell(char_lstm_dim)
                # dropout
                if is_training: char_lstm_cell = tf.contrib.rnn.DropoutWrapper(char_lstm_cell, output_keep_prob=(1 - dropout_rate))
                char_lstm_cell = tf.contrib.rnn.MultiRNNCell([char_lstm_cell])

                # question_representation
                question_char_outputs = my_rnn.dynamic_rnn(char_lstm_cell, in_question_char_repres, 
                        sequence_length=question_char_lengths,dtype=tf.float32)[0] # [batch_size*question_len, q_char_len, char_lstm_dim]
                question_char_outputs = question_char_outputs[:,-1,:]
                question_char_outputs = tf.reshape(question_char_outputs, [batch_size, question_len, char_lstm_dim])
            
                #if share_params:
                tf.get_variable_scope().reuse_variables()
                    
                # passage representation
                passage_char_outputs = my_rnn.dynamic_rnn(char_lstm_cell, in_passage_char_repres, 
                        sequence_length=passage_char_lengths,dtype=tf.float32)[0] # [batch_size*question_len, q_char_len, char_lstm_dim]
                passage_char_outputs = passage_char_outputs[:,-1,:]
                passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, passage_len, char_lstm_dim])
                
            in_question_repres.append(question_char_outputs)
            in_passage_repres.append(passage_char_outputs)

            input_dim += char_lstm_dim
       
       
        in_question_repres = tf.concat(in_question_repres,2) # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(in_passage_repres,2) # [batch_size, passage_len, dim]

        if is_training:
            in_question_repres = tf.nn.dropout(in_question_repres, (1 - dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - dropout_rate))
        else:
            in_question_repres = tf.multiply(in_question_repres, (1 - dropout_rate))
            in_passage_repres = tf.multiply(in_passage_repres, (1 - dropout_rate))

        

        #if True:
            
         

        if with_dep:
            cur_word_dim = input_dim#300 + char_lstm_dim
            self.in_question_dependency_pos = tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
            self.in_passage_dependency_pos = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
            self.in_question_dependency_label = tf.placeholder(tf.int32, [None, None])
            self.in_passage_dependency_label = tf.placeholder(tf.int32, [None, None])


            with tf.variable_scope('dependency_embedding'):
                self.dependency_embedding = tf.get_variable("dependency_embedding", 
                                        initializer=dep_vocab.word_vecs, dtype=tf.float32)

                #squeze head word embedding to size of dep_dim
                w_d = tf.get_variable("w_d", [cur_word_dim, dep_emb_dim], dtype=tf.float32)
                b_d = tf.get_variable("b_d", [dep_emb_dim], dtype=tf.float32)
                # Ex:    The     black     cat     sees       us       .
                # deps:(2-det) (2-amod) (3-nsubj) (0-root) (3-dobj)

                question_shape=tf.shape(in_question_repres)
                passage_shape=tf.shape(in_passage_repres)
                #Now gather words at pos and get embbeding for types
                question_dependency_labels_repres = tf.nn.embedding_lookup(self.dependency_embedding, self.in_question_dependency_label) #[batch_size, passage_len, dep_dim]  
                question_head_repres = match_utils.gather_along_second_axis(in_question_repres, self.in_question_dependency_pos) # [batch_size, passage_len, word_dim]
                if is_training:
                    question_head_repres= tf.nn.dropout(question_head_repres, (1 - dropout_rate))
                else:
                    question_head_repres=tf.multiply(question_head_repres, (1 - dropout_rate))
                question_head_repres = tf.reshape(question_head_repres, [-1, cur_word_dim])
                question_head_repres = tf.matmul(question_head_repres, w_d) + b_d #[batch_size*passage_len, dep_dim]
                print ('dep_emb_dim:%d'%(dep_emb_dim))
                question_head_repres = tf.reshape(question_head_repres, [question_shape[0], question_shape[1], dep_emb_dim]) #[batch_size, passage_len, dep_dim]
                question_dependency_parts = tf.multiply(question_dependency_labels_repres, question_head_repres) #[batch_size, passage_len, dep_dim] 
               
                passage_dependency_labels_repres = tf.nn.embedding_lookup(self.dependency_embedding, self.in_passage_dependency_label) #[batch_size, passage_len, dep_dim]  
                passage_head_repres = match_utils.gather_along_second_axis(in_passage_repres, self.in_passage_dependency_pos) # [batch_size, passage_len, word_dim]
                passage_head_repres = tf.reshape(passage_head_repres, [-1, cur_word_dim])
                if is_training:
                    passage_head_repres= tf.nn.dropout(passage_head_repres, (1 - dropout_rate))
                else:
                    passage_head_repres=tf.multiply(passage_head_repres, (1 - dropout_rate)) 
                passage_head_repres = tf.matmul(passage_head_repres, w_d) + b_d #[batch_size*passage_len, dep_dim] 
                passage_head_repres = tf.reshape(passage_head_repres, [passage_shape[0], passage_shape[1], dep_emb_dim]) #[batch_size, passage_len, dep_dim]
                passage_dependency_parts = tf.multiply(passage_dependency_labels_repres, passage_head_repres) #[batch_size, passage_len, dep_dim] 
                 
                if is_training:
                    passage_dependency_parts = tf.nn.dropout(passage_dependency_parts, (1 - dropout_rate))
                    question_dependency_parts = tf.nn.dropout(question_dependency_parts, (1 - dropout_rate))
                else:
                    passage_dependency_parts = tf.multiply(passage_dependency_parts, (1 - dropout_rate))
                    question_dependency_parts = tf.multiply(question_dependency_parts, (1 - dropout_rate))

                
                #Now add this dependency parts to words representation
                in_question_repres = tf.concat([in_question_repres, question_dependency_parts],2)
                in_passage_repres = tf.concat([in_passage_repres, passage_dependency_parts],2)



        if True:
            with tf.variable_scope('context_represent'):
                # parameters
                context_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                context_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                if is_training:
                    context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                context_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_fw])
                context_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_bw])



                # question representation
                (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                            context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32,
                                            sequence_length=self.question_lengths) # [batch_size, question_len, context_lstm_dim]
                question_context_representation = tf.concat([question_context_representation_fw, question_context_representation_bw], 2)
            
                #if share_params: 
                tf.get_variable_scope().reuse_variables()
                (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32,
                                        sequence_length=self.passage_lengths) # [batch_size, passage_len, context_lstm_dim] 
                passage_context_representation = tf.concat([passage_context_representation_fw, passage_context_representation_bw], 2) # [batch_size, passage_len, 2*context_lstm_dim]
            
                
                in_question_repres = question_context_representation
                in_passage_repres = passage_context_representation
                input_dim = 2*context_lstm_dim

        

        if with_idep:#concatenate after context layer with inverse dependency info
            print('run the dependency from node to head')
            question_shape=tf.shape(in_question_repres)
            passage_shape=tf.shape(in_passage_repres)
            
            self.in_question_idependency_pos = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, question_len, num_dep]
            self.in_passage_idependency_pos = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, passage_len, num_dep]
            self.in_question_idependency_label = tf.placeholder(tf.int32, [None, None, None])
            self.in_passage_idependency_label = tf.placeholder(tf.int32, [None, None, None])
            with tf.variable_scope('dependency_layer'):
                self.idependency_embedding = tf.get_variable("dependency_layer_emb",
                                                initializer=dep_vocab.word_vecs, dtype=tf.float32)
                # gather dependency tokens
                question_idependency_tokens_repres = match_utils.gather_along_second_axis2(in_question_repres, self.in_question_idependency_pos)#[batch_size, question_len, num_dep, word_dim]
                passage_idependency_tokens_repres = match_utils.gather_along_second_axis2(in_passage_repres, self.in_passage_idependency_pos)#[batch_size, question_len, num_dep, word_dim]
            
                # get dependency embedding
                question_idependency_emb = tf.nn.embedding_lookup(self.idependency_embedding, self.in_question_idependency_label) #[batch_size, question_len, num_dep, dep_dim]
                passage_idependency_emb = tf.nn.embedding_lookup(self.idependency_embedding, self.in_passage_idependency_label) #[batch_size, question_len, num_dep, dep_dim]
            
                # reshape token, emb to matrices
                question_idependency_tokens_repres = tf.reshape(question_idependency_tokens_repres, [-1, question_shape[2]]) #TODO check input_dim
                passage_idependency_tokens_repres = tf.reshape(passage_idependency_tokens_repres, [-1, passage_shape[2]])
                
                question_emb_shape = tf.shape(question_idependency_emb) #save this shape
                passage_emb_shape = tf.shape(passage_idependency_emb) #save this shape
                question_idependency_emb = tf.reshape(question_idependency_emb, [-1, dep_emb_dim])
                passage_idependency_emb = tf.reshape(passage_idependency_emb, [-1, dep_emb_dim])

                # make token dim become the same dim as label
                w_d = tf.get_variable("w_d", [200, dep_emb_dim], dtype=tf.float32)
                b_d = tf.get_variable("b_d", [dep_emb_dim], dtype=tf.float32)
                        
                if is_training:
                    question_idependency_tokens_repres = tf.nn.dropout(question_idependency_tokens_repres, (1 - dropout_rate))
                    passage_idependency_emb = tf.nn.dropout(passage_idependency_emb, (1 - dropout_rate))
                else:
                    question_idependency_tokens_repres = tf.multiply(question_idependency_tokens_repres, (1 - dropout_rate))
                    passage_idependency_tokens_repres = tf.multiply(passage_idependency_tokens_repres, (1 - dropout_rate))

                question_idependency_tokens_repres = tf.matmul(question_idependency_tokens_repres, w_d) + b_d
                passage_idependency_tokens_repres = tf.matmul(passage_idependency_tokens_repres, w_d) + b_d
                
                # concat token_repres * dep_repres to current word_rep
                question_idependency_parts = tf.multiply(question_idependency_tokens_repres, question_idependency_emb)
                question_idependency_parts = tf.reshape(question_idependency_parts, question_emb_shape) #[batch_size, question_len, num_dep, dep_dim]
                question_idependency_parts = tf.reduce_sum(question_idependency_parts, 2) #[batch_size, question_len, dep_dim]
                
                passage_idependency_parts = tf.multiply(passage_idependency_tokens_repres, passage_idependency_emb)
                passage_idependency_parts = tf.reshape(passage_idependency_parts, passage_emb_shape) #[batch_size, question_len, num_dep, dep_dim]
                passage_idependency_parts = tf.reduce_sum(passage_idependency_parts, 2) #[batch_size, question_len, dep_dim]
                
                in_question_repres = tf.concat([in_question_repres, question_idependency_parts], 2)
                in_passage_repres = tf.concat([in_passage_repres, passage_idependency_parts], 2)
                question_context_representation = in_question_repres
                passage_context_representation = in_passage_repres
                input_dim += dep_emb_dim
                    

        
        if False: #bilinear pooling
                match_dim = 2048
                question_context_representation = tf.reduce_max(question_context_representation, 1) #[batch_size, context_lstm_dim]
                passage_context_representation = tf.reduce_max(passage_context_representation, 1) #[batch_size, context_lstm_dim]

                if is_training:
                    question_context_representation = tf.nn.dropout(question_context_representation, (1 - dropout_rate))
                    passage_context_representation = tf.nn.dropout(passage_context_representation, (1 - dropout_rate))
                else:
                    question_context_representation = tf.multiply(question_context_representation, (1 - dropout_rate))
                    passage_context_representation = tf.multiply(passage_context_representation, (1 - dropout_rate))

                match_representation = bilinear_pool(question_context_representation, passage_context_representation, match_dim)
            
        if True:
            ### Mean pooling
            premise_sum = tf.reduce_max(question_context_representation, 1) #[batch_size, context_lstm_dim]
            premise_ave = premise_sum

            hypothesis_sum = tf.reduce_max(passage_context_representation, 1) #[batch_size, context_lstm_dim]
            #hypo_lens = tf.expand_dims(tf.cast(self.passage_lengths, tf.float32), -1) # [batch_size] -> [batch_size, 1]
            #hypothesis_ave = tf.div(hypothesis_sum, hypo_lens) #[batch_size, context_lstm_dim]
            hypothesis_ave = hypothesis_sum

            match_dim = 4*input_dim
            print('input_dim: ', input_dim)
            print ('match_dim: ', match_dim)

            #if False:
            ### Mou et al. concat layer ###
            #diff = tf.subtract(premise_ave, hypothesis_ave)
            #add = tf.add(premise_ave, hypothesis_ave)
            mul = tf.multiply(premise_ave, hypothesis_ave)
            max_ = tf.maximum(premise_ave, hypothesis_ave)
            match_representation = tf.concat([premise_ave, hypothesis_ave, max_, mul],1)




        #========Prediction Layer=========
        w_0 = tf.get_variable("w_0", [match_dim, match_dim/4], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [match_dim/4], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [match_dim/4, num_classes],dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [num_classes],dtype=tf.float32)

        logits = tf.matmul(match_representation, w_0) + b_0
        logits = tf.tanh(logits)
        if is_training:
            logits = tf.nn.dropout(logits, (1 - dropout_rate))
        else:
            logits = tf.multiply(logits, (1 - dropout_rate))
        logits = tf.matmul(logits, w_1) + b_1

        self.prob = tf.nn.softmax(logits)
        
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.cast(self.truth, tf.int64), name='cross_entropy_per_example')
        #self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

        gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))

        correct = tf.nn.in_top_k(logits, self.truth, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        self.predictions = tf.arg_max(self.prob, 1)

        if optimize_type == 'adadelta':
            clipper = 50 
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars)) 
        elif optimize_type == 'sgd':
            self.global_step = tf.Variable(0, name='global_step', trainable=False) # Create a variable to track the global step.
            min_lr = 0.000001
            self._lr_rate = tf.maximum(min_lr, tf.train.exponential_decay(learning_rate, self.global_step, 30000, 0.98))
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self._lr_rate).minimize(self.loss)
        elif optimize_type == 'ema':
            tvars = tf.trainable_variables()
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            # Create an ExponentialMovingAverage object
            ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            # Create the shadow variables, and add ops to maintain moving averages # of var0 and var1.
            maintain_averages_op = ema.apply(tvars)
            # Create an op that will update the moving averages after each training
            # step.  This is what we will use in place of the usual training op.
            with tf.control_dependencies([train_op]):
                self.train_op = tf.group(maintain_averages_op)
        elif optimize_type == 'adam':
            clipper = 50 
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars)) 

        extra_train_ops = []
        train_ops = [self.train_op] + extra_train_ops
        self.train_op = tf.group(*train_ops)


    def conv_sent(self, filter_sizes, word_dim, num_filters, sequence_length, input_tensor):
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                filter_shape=[filter_size, word_dim, 1, num_filters]
                #w_0 = tf.get_variable("w_0", [match_dim, match_dim/4], dtype=tf.float32)
                W = tf.get_variable("W", filter_shape )
                b = tf.get_variable("b", shape=[num_filters])
                #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                #Convolution
                conv = tf.nn.conv2d(input_tensor, W,    # []
                    strides=[1,1,1,1], padding="VALID", name="conv")

                #Apply non-linearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                #Max pooling over the input
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
                #tf.get_variable_scope().reuse_variables()

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        return h_pool_flat


    def get_predictions(self):
        return self.__predictions


    def set_predictions(self, value):
        self.__predictions = value


    def del_predictions(self):
        del self.__predictions



    def get_eval_correct(self):
        return self.__eval_correct


    def set_eval_correct(self, value):
        self.__eval_correct = value


    def del_eval_correct(self):
        del self.__eval_correct


    def get_question_lengths(self):
        return self.__question_lengths


    def get_passage_lengths(self):
        return self.__passage_lengths


    def get_truth(self):
        return self.__truth


    def get_in_question_words(self):
        return self.__in_question_words

    def get_in_question_dep_con(self):
        return self.__in_question_dep_con

    def set_in_question_dep_con(self, value):
        self.__in_question_dep_con = value
    
    def del_in_question_dep_con(self):
        del self.__in_question_dep_con
    
    def get_in_passage_dep_con(self):
        return self.__in_passage_dep_con
    
    def set_in_passage_dep_con(self, value):
        self.__in_passage_dep_con = value
    
    def del_in_passage_dep_con(self):
        del self.__in_passage_dep_con

    def get_in_passage_words(self):
        return self.__in_passage_words
    
    def get_in_question_dependency_pos(self):
        return self.__in_question_dependency_pos
    
    def get_in_question_idependency_pos(self):
        return self.__in_question_idependency_pos
    
    def set_in_question_idependency_pos(self, value):
        self.__in_question_idependency_pos = value

    def del_in_question_idependency_pos(self):
        del self.__in_question_idependency_pos
  
    def get_in_question_idependency_label(self):
        return self.__in_question_idependency_label
 
    def set_in_question_idependency_label(self, value):
        self.__in_question_idependency_label = value

    def del_in_question_idependency_label(self):
        del self.__in_question_idependency_label
    
    def get_in_question_dependency_label(self):
        return self.__in_question_dependency_label

    def get_in_passage_idependency_pos(self):
        return self.__in_passage_idependency_pos
    
    def set_in_passage_idependency_pos(self, value):
        self.__in_passage_idependency_pos = value

    def del_in_passage_idependency_pos(self, value):
        del self.__in_passage_idependency_pos
    
    def get_in_passage_idependency_label(self):
        return self.__in_passage_idependency_label
    
    def set_in_passage_idependency_label(self, value):
        self.__in_passage_idependency_label = value
    
    def del_in_passage_idependency_label(self):
        del self.__in_passage_idependency_label

    def get_in_passage_dependency_pos(self):
        return self.__in_passage_dependency_pos

    def get_in_passage_dependency_label(self):
        return self.__in_passage_dependency_label

    def get_word_embedding(self):
        return self.__word_embedding


    #def get_in_question_poss(self):
    #    return self.__in_question_POSs


    #def get_in_passage_poss(self):
    #    return self.__in_passage_POSs
    def get_in_pos1(self):
        return self.__in_question_POSs


    def get_in_pos2(self):
        return self.__in_passage_POSs


    def get_pos_embedding(self):
        return self.__POS_embedding


    def get_in_question_ners(self):
        return self.__in_question_NERs


    def get_in_passage_ners(self):
        return self.__in_passage_NERs


    def get_ner_embedding(self):
        return self.__NER_embedding


    def get_question_char_lengths(self):
        return self.__question_char_lengths


    def get_passage_char_lengths(self):
        return self.__passage_char_lengths


    def get_in_question_chars(self):
        return self.__in_question_chars


    def get_in_passage_chars(self):
        return self.__in_passage_chars


    def get_char_embedding(self):
        return self.__char_embedding


    def get_prob(self):
        return self.__prob


    def get_prediction(self):
        return self.__prediction


    def get_loss(self):
        return self.__loss


    def get_train_op(self):
        return self.__train_op


    def get_global_step(self):
        return self.__global_step


    def get_lr_rate(self):
        return self.__lr_rate


    def set_question_lengths(self, value):
        self.__question_lengths = value


    def set_passage_lengths(self, value):
        self.__passage_lengths = value


    def set_truth(self, value):
        self.__truth = value


    def set_in_question_words(self, value):
        self.__in_question_words = value


    def set_in_passage_words(self, value):
        self.__in_passage_words = value
    
    def set_in_question_dependency_pos(self, value):
        self.__in_question_dependency_pos = value

    def set_in_question_dependency_label(self, value):
        self.__in_question_dependency_label = value

    def set_in_passage_dependency_pos(self, value):
        self.__in_passage_dependency_pos = value

    def set_in_passage_dependency_label(self, value):
        self.__in_passage_dependency_label = value

    def set_word_embedding(self, value):
        self.__word_embedding = value


    def set_in_question_poss(self, value):
        self.__in_question_POSs = value


    def set_in_passage_poss(self, value):
        self.__in_passage_POSs = value


    def set_pos_embedding(self, value):
        self.__POS_embedding = value


    def set_in_question_ners(self, value):
        self.__in_question_NERs = value


    def set_in_passage_ners(self, value):
        self.__in_passage_NERs = value


    def set_ner_embedding(self, value):
        self.__NER_embedding = value


    def set_question_char_lengths(self, value):
        self.__question_char_lengths = value


    def set_passage_char_lengths(self, value):
        self.__passage_char_lengths = value


    def set_in_question_chars(self, value):
        self.__in_question_chars = value


    def set_in_passage_chars(self, value):
        self.__in_passage_chars = value


    def set_char_embedding(self, value):
        self.__char_embedding = value


    def set_prob(self, value):
        self.__prob = value


    def set_prediction(self, value):
        self.__prediction = value


    def set_loss(self, value):
        self.__loss = value


    def set_train_op(self, value):
        self.__train_op = value


    def set_global_step(self, value):
        self.__global_step = value


    def set_lr_rate(self, value):
        self.__lr_rate = value


    def del_question_lengths(self):
        del self.__question_lengths


    def del_passage_lengths(self):
        del self.__passage_lengths


    def del_truth(self):
        del self.__truth


    def del_in_question_words(self):
        del self.__in_question_words


    def del_in_passage_words(self):
        del self.__in_passage_words

    def del_in_question_dependency_pos(self):
        del self.__in_question_dependency_pos

    def del_in_question_dependency_label(self):
        del self.__in_question_dependency_label

    def del_in_passage_dependency_pos(self):
        del self.__in_passage_dependency_pos

    def del_in_passage_dependency_label(self):
        del self.__in_passage_dependency_label

    def del_word_embedding(self):
        del self.__word_embedding


    def del_in_question_poss(self):
        del self.__in_question_POSs


    def del_in_passage_poss(self):
        del self.__in_passage_POSs
    def set_in_question_poss(self, value):
        self.__in_question_POSs = value


    def set_in_passage_poss(self, value):
        self.__in_passage_POSs = value


    def del_pos_embedding(self):
        del self.__POS_embedding


    def del_in_question_ners(self):
        del self.__in_question_NERs


    def del_in_passage_ners(self):
        del self.__in_passage_NERs


    def del_ner_embedding(self):
        del self.__NER_embedding


    def del_question_char_lengths(self):
        del self.__question_char_lengths


    def del_passage_char_lengths(self):
        del self.__passage_char_lengths


    def del_in_question_chars(self):
        del self.__in_question_chars


    def del_in_passage_chars(self):
        del self.__in_passage_chars


    def del_char_embedding(self):
        del self.__char_embedding


    def del_prob(self):
        del self.__prob


    def del_prediction(self):
        del self.__prediction


    def del_loss(self):
        del self.__loss


    def del_train_op(self):
        del self.__train_op


    def del_global_step(self):
        del self.__global_step


    def del_lr_rate(self):
        del self.__lr_rate

    def get_image_feats(self):
        return self.__image_feats

    def set_image_feats(self, value):
        self.__image_feats = value

    def del_image_feats(self):
        del self.__image_feats

    image_feats = property(get_image_feats, set_image_feats, del_image_feats, "image_features's docstring")
    question_lengths = property(get_question_lengths, set_question_lengths, del_question_lengths, "question_lengths's docstring")
    passage_lengths = property(get_passage_lengths, set_passage_lengths, del_passage_lengths, "passage_lengths's docstring")
    truth = property(get_truth, set_truth, del_truth, "truth's docstring")
    in_question_words = property(get_in_question_words, set_in_question_words, del_in_question_words, "in_question_words's docstring")
    in_passage_words = property(get_in_passage_words, set_in_passage_words, del_in_passage_words, "in_passage_words's docstring")
    in_question_dependency_pos = property(get_in_question_dependency_pos, set_in_question_dependency_pos, del_in_question_dependency_pos, "in_question_dependency's docstring")
    in_passage_dependency_pos = property(get_in_passage_dependency_pos, set_in_passage_dependency_pos, del_in_passage_dependency_pos, "in_passage_dependency's docstring")
    in_question_dependency_label = property(get_in_question_dependency_label, set_in_question_dependency_label, del_in_question_dependency_label, "in_question_dependency's docstring")
    in_passage_dependency_label = property(get_in_passage_dependency_label, set_in_passage_dependency_label, del_in_passage_dependency_label, "in_passage_dependency's docstring")
    
    in_question_idependency_pos = property(get_in_question_idependency_pos, set_in_question_idependency_pos, del_in_question_idependency_pos, "in_question_dependency's docstring")
    in_passage_idependency_pos = property(get_in_passage_idependency_pos, set_in_passage_idependency_pos, del_in_passage_idependency_pos,"in_passage_dependency's docstring")
    in_question_idependency_label = property(get_in_question_idependency_label, set_in_question_idependency_label, del_in_question_idependency_label, "in_question_dependency's docstring")
    in_passage_idependency_label = property(get_in_passage_idependency_label, set_in_passage_idependency_label, del_in_passage_idependency_label, "in_passage_dependency's docstring")
 
    word_embedding = property(get_word_embedding, set_word_embedding, del_word_embedding, "word_embedding's docstring")
    #in_question_POSs = property(get_in_question_poss, set_in_question_poss, del_in_question_poss, "in_question_POSs's docstring")
    #in_passage_POSs = property(get_in_passage_poss, set_in_passage_poss, del_in_passage_poss, "in_passage_POSs's docstring")
    in_prem_POSs = property(get_in_pos1, set_in_question_poss, del_in_question_poss, "in_question_POSs's docstring")
    in_hypo_POSs = property(get_in_pos2, set_in_passage_poss, del_in_passage_poss, "in_passage_POSs's docstring")
    POS_embedding = property(get_pos_embedding, set_pos_embedding, del_pos_embedding, "POS_embedding's docstring")
    in_question_NERs = property(get_in_question_ners, set_in_question_ners, del_in_question_ners, "in_question_NERs's docstring")
    in_passage_NERs = property(get_in_passage_ners, set_in_passage_ners, del_in_passage_ners, "in_passage_NERs's docstring")
    NER_embedding = property(get_ner_embedding, set_ner_embedding, del_ner_embedding, "NER_embedding's docstring")
    question_char_lengths = property(get_question_char_lengths, set_question_char_lengths, del_question_char_lengths, "question_char_lengths's docstring")
    passage_char_lengths = property(get_passage_char_lengths, set_passage_char_lengths, del_passage_char_lengths, "passage_char_lengths's docstring")
    in_question_chars = property(get_in_question_chars, set_in_question_chars, del_in_question_chars, "in_question_chars's docstring")
    in_passage_chars = property(get_in_passage_chars, set_in_passage_chars, del_in_passage_chars, "in_passage_chars's docstring")
    char_embedding = property(get_char_embedding, set_char_embedding, del_char_embedding, "char_embedding's docstring")
    prob = property(get_prob, set_prob, del_prob, "prob's docstring")
    prediction = property(get_prediction, set_prediction, del_prediction, "prediction's docstring")
    loss = property(get_loss, set_loss, del_loss, "loss's docstring")
    train_op = property(get_train_op, set_train_op, del_train_op, "train_op's docstring")
    global_step = property(get_global_step, set_global_step, del_global_step, "global_step's docstring")
    lr_rate = property(get_lr_rate, set_lr_rate, del_lr_rate, "lr_rate's docstring")
    eval_correct = property(get_eval_correct, set_eval_correct, del_eval_correct, "eval_correct's docstring")
    predictions = property(get_predictions, set_predictions, del_predictions, "predictions's docstring")
    
    
