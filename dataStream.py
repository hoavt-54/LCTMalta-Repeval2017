import numpy as np
import re

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)] # zgwang: starting point of each batch

def pad_2d_matrix(in_val, max_length=None, dtype=np.int32):
    if max_length is None: max_length = np.max([len(cur_in_val) for cur_in_val in in_val])
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length), dtype=dtype)
    for i in xrange(batch_size):
        cur_in_val = in_val[i]
        kept_length = len(cur_in_val)
        if kept_length>max_length: kept_length = max_length
        out_val[i,:kept_length] = cur_in_val[:kept_length]
    return out_val

def pad_3d_tensor(in_val, max_length1=None, max_length2=None, dtype=np.int32):
    if max_length1 is None: max_length1 = np.max([len(cur_in_val) for cur_in_val in in_val])
    if max_length2 is None: max_length2 = np.max([np.max([len(val) for val in cur_in_val]) for cur_in_val in in_val])
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length1, max_length2), dtype=dtype)
    for i in xrange(batch_size):
        cur_length1 = max_length1
        if len(in_val[i])<max_length1: cur_length1 = len(in_val[i])
        for j in xrange(cur_length1):
            cur_in_val = in_val[i][j]
            kept_length = len(cur_in_val)
            if kept_length>max_length2: kept_length = max_length2
            out_val[i, j, :kept_length] = cur_in_val[:kept_length]
    return out_val


def pad_dependency_3d_tensor(in_val,pad_value=None, max_length1=None, max_length2=None, dtype=np.int32):
    if max_length1 is None: max_length1 = np.max([len(cur_in_val) for cur_in_val in in_val])
    if max_length2 is None: max_length2 = np.max([np.max([len(val) for val in cur_in_val]) for cur_in_val in in_val])
    if max_length2 > 6: max_length2 = 6
    #print 'max_length1: ', max_length1
    #print('max_length2: ', max_length2)
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length1, max_length2), dtype=dtype)
    
    #padded token connect to itself
    for i in out_val: #batch_size
        for j in xrange(max_length1): # sentence_length
            for k in xrange(max_length2): # num_deps
                if not pad_value:
                    i[j][k] = j 
                else:
                    i[j][k] = pad_value
    #print 'out_val after padding: ', out_val

    for i in xrange(batch_size):
        cur_length1 = max_length1
        if len(in_val[i])<max_length1: cur_length1 = len(in_val[i])
        for j in xrange(cur_length1):
            cur_in_val = in_val[i][j]
            kept_length = len(cur_in_val)
            if kept_length>max_length2: kept_length = max_length2
            out_val[i, j, :kept_length] = cur_in_val[:kept_length]
    return out_val 



def pad_dependency_2d_matrix(in_val, pad_value,  max_length=None, dtype=np.int32):
    if max_length is None: max_length = np.max([len(cur_in_val) for cur_in_val in in_val])
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length), dtype=dtype)
    for i in xrange(batch_size):
        #first pad to pad_value
        if pad_value > 0:
            out_val[i] = [pad_value]*max_length
        else:
            for idx in range(len(out_val[i])):
                out_val[i][idx] = idx
        cur_in_val = in_val[i]
        kept_length = len(cur_in_val)
        if kept_length>max_length: kept_length = max_length
        out_val[i,:kept_length] = cur_in_val[:kept_length]
    return out_val




class SentenceMatchDataStream(object):
    def __init__(self, inpath, word_vocab=None, char_vocab=None, label_vocab=None, dep_vocab = None, pos_vocab=None, 
                batch_size=60, isShuffle=False, isLoop=False, isSort=True, max_char_per_word=10, 
                with_dep = False, with_pos=False, max_sent_length=100, with_dep_inverse=True):

        instances = []
        count_ins = 0
        infile = open(inpath, 'rt')
        for line in infile:
            #if(count_ins > 3000): break
            
            count_ins +=1
            if(count_ins == 1): continue
            line = line.decode('utf-8').strip()
            if len(line) < 7: continue
            #if line.startswith('-'): continue
            items = re.split("\t", line)
            label = items[0]
            pairID= items[7]
            if len(label) < 4: continue
            sentence1 = items[1].lower()
            sentence2 = items[4].lower()

            if label_vocab is not None: 
                label_id = label_vocab.getIndex(label)
                if label_id >= label_vocab.vocab_size: label_id = 0
            else: 
                label_id = int(label)
            word_idx_1 = word_vocab.to_index_sequence(sentence1)
            word_idx_2 = word_vocab.to_index_sequence(sentence2)
            depsl1, depsl2, depp1, depp2 = None, None, None, None
            idepsl1, idepsl2, idepp1, idepp2 = None, None, None, None 
            if with_dep:
                depsl1 = [pair.split('-')[1] for pair in re.split("\\s+",items[2])]
                depsl2 = [pair.split('-')[1] for pair in re.split("\\s+",items[5])]
                 
                depp1 = [int(pair.split('-')[0]) - 1 for pair in re.split("\\s+",items[2])]
                depp2 = [int(pair.split('-')[0]) - 1 for pair in re.split("\\s+",items[5])]
                #fix the root label, depend on itself
                for idx, dep in enumerate(depsl1):
                    if dep == 'root' or depp1[idx] < 0:
                        depp1[idx] = idx
   
                for idx, dep in enumerate(depsl2):
                    if dep == 'root' or depp2[idx] < 0:
                        depp2[idx] = idx 
                
                depsl1 = dep_vocab.to_index_sequence(' '.join(depsl1))
                depsl2 = dep_vocab.to_index_sequence(' '.join(depsl2))
                
                assert(len(depp1) == len(word_idx_1))
                assert(len(depp2) == len(word_idx_2))
                
                #inverse dep
                idepp1 = [[] for _ in range(len(depp1))]
                idepsl1 = [[] for _ in range(len(depp1))]
                idepp2 = [[] for _ in range(len(depp2))]
                idepsl2 = [[] for _ in range(len(depp2))]

                for idx, dep in enumerate(depp1):
                    if idx >= max_sent_length: break
                    idepp1[dep].append(idx)
                    idepsl1[dep].append(depsl1[idx])
                for idx, dep in enumerate(depp2):
                    if idx >= max_sent_length: break
                    idepp2[dep].append(idx)
                    idepsl2[dep].append(depsl2[idx])
                #print 'depp1', depp1
                #print 'depsl1', depsl1
                #print 'idepp1', idepp1
                #print 'idepsl1', idepsl1
                #print '\n\n' 
            #if with_dep_inverse:
                

            pos1, pos2, pos1_idx, pos2_idx = None, None, None, None
            if with_pos:
                pos1 = items[3]
                pos2 = items[6]
                pos1_idx = pos_vocab.to_index_sequence(pos1)# treat pos as word, but different encoder
                pos2_idx = pos_vocab.to_index_sequence(pos2)
                assert(len(pos1_idx) == len(word_idx_1))
                assert(len(pos2_idx) == len(word_idx_2))
            char_matrix_idx_1 = char_vocab.to_character_matrix(sentence1)
            char_matrix_idx_2 = char_vocab.to_character_matrix(sentence2)

       
            char_matrix_idx_1 = char_vocab.to_character_matrix(sentence1)
            char_matrix_idx_2 = char_vocab.to_character_matrix(sentence2)  
            
             
            if len(word_idx_1)>max_sent_length: 
                word_idx_1 = word_idx_1[:max_sent_length]
                char_matrix_idx_1 = char_matrix_idx_1[:max_sent_length]
                if with_dep:
                    depp1 = depp1[:max_sent_length]
                    depsl1 = depsl1[:max_sent_length]
                    #after cutting, some words may refer out of bound, change to refer to itself
                    for idx in range(max_sent_length):
                        if depp1[idx] >= max_sent_length: 
                            depp1[idx] = idx

                if with_pos:
                    pos1_idx = pos1_idx[:max_sent_length]

            if len(word_idx_2)>max_sent_length:
                word_idx_2 = word_idx_2[:max_sent_length]
                char_matrix_idx_1 = char_matrix_idx_1[:max_sent_length]
                if with_dep:
                    depp2 = depp2[:max_sent_length]
                    depsl2 = depsl2[:max_sent_length]
                    for idx in range(max_sent_length):
                        if depp2[idx] >= max_sent_length: 
                            depp2[idx] = idx
                if with_pos:
                    pos2_idx = pos2_idx[:max_sent_length]
            
            
            instances.append((label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2,
                              depp1, depp2, depsl1, depsl2, idepsl1, idepsl2, idepp1, idepp2, pos1_idx, pos2_idx, pairID))
        infile.close()

        # sort instances based on sentence length
        #if isSort: instances = sorted(instances, key=lambda instance: (len(instance[4]), len(instance[5]))) # sort instances based on length
        self.num_instances = len(instances)
        
        # distribute into different buckets
        batch_spans = make_batches(self.num_instances, batch_size) 
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            label_batch = []
            sent1_batch = []
            sent2_batch = []
            label_id_batch = []
            word_idx_1_batch = []
            word_idx_2_batch = []
            dep_pos1_batch = []
            dep_pos2_batch = []
            dep_label1_batch = []
            dep_label2_batch = []
            idep_pos1_batch = []
            idep_pos2_batch = []
            idep_label1_batch = []
            idep_label2_batch = []
            char_matrix_idx_1_batch = []
            char_matrix_idx_2_batch = []
            sent1_length_batch = []
            sent2_length_batch = []
            sent1_char_length_batch = []
            sent2_char_length_batch = []
            pos1_idx_batch = []
            pos2_idx_batch = []
            pairID_batch = []

            for i in xrange(batch_start, batch_end):
                (label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2,
                    depp1, depp2, depsl1, depsl2, idepsl1, idepsl2, idepp1, idepp2, pos1_idx, pos2_idx, pairID) = instances[i]
                label_batch.append(label)
                sent1_batch.append(sentence1)
                sent2_batch.append(sentence2)
                label_id_batch.append(label_id)
                word_idx_1_batch.append(word_idx_1)
                word_idx_2_batch.append(word_idx_2)
                # add sentence dependencies to batch
                if with_dep:
                    dep_pos1_batch.append(depp1)
                    dep_pos2_batch.append(depp2)
                    dep_label1_batch.append(depsl1)
                    dep_label2_batch.append(depsl2)
                    idep_pos1_batch.append(idepp1)
                    idep_pos2_batch.append(idepp2)
                    idep_label1_batch.append(idepsl1)
                    idep_label2_batch.append(idepsl2)
                # add pos tags
                if with_pos:
                    pos1_idx_batch.append(pos1_idx)
                    pos2_idx_batch.append(pos2_idx)
                
                char_matrix_idx_1_batch.append(char_matrix_idx_1)
                char_matrix_idx_2_batch.append(char_matrix_idx_2)
                sent1_length_batch.append(len(word_idx_1))
                sent2_length_batch.append(len(word_idx_2))
                sent1_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_1])
                sent2_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_2])
                pairID_batch.append(pairID)
                
                
            cur_batch_size = len(label_batch)
            if cur_batch_size ==0: continue

            # padding
            max_sent1_length = np.max(sent1_length_batch) #max_sent_length
            max_sent2_length = np.max(sent2_length_batch) #max_sent_length
            #print 'sent1_length_batch', sent1_length_batch
            #max_sent1_length = max_sent_length
            #max_sent2_length = max_sent_length
            #print('maxSentenceLength: ', max_sent1_length, max_sent2_length)
            max_char_length1 = np.max([np.max(aa) for aa in sent1_char_length_batch])
            if max_char_length1>max_char_per_word: max_char_length1=max_char_per_word

            max_char_length2 = np.max([np.max(aa) for aa in sent2_char_length_batch])
            if max_char_length2>max_char_per_word: max_char_length2=max_char_per_word
            
            label_id_batch = np.array(label_id_batch)
            word_idx_1_batch = pad_2d_matrix(word_idx_1_batch, max_length=max_sent1_length)
            word_idx_2_batch = pad_2d_matrix(word_idx_2_batch, max_length=max_sent2_length)
            
            if with_dep:
                dep_pos1_batch = pad_dependency_2d_matrix(dep_pos1_batch, 0, max_length=max_sent1_length)
                dep_pos2_batch = pad_dependency_2d_matrix(dep_pos2_batch, 0, max_length=max_sent2_length)
                dep_label1_batch = pad_dependency_2d_matrix(dep_label1_batch, dep_vocab.vocab_size, max_length=max_sent1_length) 
                dep_label2_batch = pad_dependency_2d_matrix(dep_label2_batch, dep_vocab.vocab_size, max_length=max_sent2_length)
                
                idep_label1_batch = pad_dependency_3d_tensor(idep_label1_batch, dep_vocab.vocab_size, max_length1=max_sent1_length)
                idep_label2_batch = pad_dependency_3d_tensor(idep_label2_batch, dep_vocab.vocab_size, max_length1=max_sent2_length)
                idep_pos1_batch = pad_dependency_3d_tensor(idep_pos1_batch, None, max_length1=max_sent1_length)
                idep_pos2_batch = pad_dependency_3d_tensor(idep_pos2_batch, None, max_length1=max_sent2_length)
                
            #if with_dep_inverse:
                #dep_pos1_batch, dep_pos2_batch, dep_label1_batch, dep_label2_batch = np.array([]),np.array([]),np.array([]),np.array([])
                #print 'max_sent1_length: ', max_sent1_length, 'max_sent2_length', max_sent2_length
                #print 'label_batch', idep_label1_batch, '\n'
                #print 'position batch', idep_pos1_batch, '\n'
                #print 'dep_vocab.vocab_size', dep_vocab.vocab_size

            if with_pos:
                pos1_idx_batch = np.array(pos1_idx_batch)
                pos2_idx_batch = np.array(pos2_idx_batch)
                pos1_idx_batch = pad_2d_matrix(pos1_idx_batch, max_length=max_sent1_length)
                pos2_idx_batch = pad_2d_matrix(pos2_idx_batch, max_length=max_sent2_length)

            char_matrix_idx_1_batch = pad_3d_tensor(char_matrix_idx_1_batch, max_length1=max_sent1_length, max_length2=max_char_length1)
            char_matrix_idx_2_batch = pad_3d_tensor(char_matrix_idx_2_batch, max_length1=max_sent2_length, max_length2=max_char_length2)
            

            sent1_length_batch = np.array(sent1_length_batch)
            sent2_length_batch = np.array(sent2_length_batch)

            sent1_char_length_batch = pad_2d_matrix(sent1_char_length_batch, max_length=max_sent1_length)
            sent2_char_length_batch = pad_2d_matrix(sent2_char_length_batch, max_length=max_sent2_length)
            

            self.batches.append((label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch, 
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch, 
                                 sent1_char_length_batch, sent2_char_length_batch, dep_pos1_batch, dep_pos2_batch, 
                                 dep_label1_batch, dep_label2_batch, idep_label1_batch, idep_label2_batch, idep_pos1_batch, 
                                 idep_pos2_batch, pos1_idx_batch, pos2_idx_batch, pairID_batch))
        
        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array) 
        self.isLoop = isLoop
        self.cur_pointer = 0




    
    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0 
            if self.isShuffle: np.random.shuffle(self.index_array) 
#         print('{} '.format(self.index_array[self.cur_pointer]))
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        self.cur_pointer = 0
    
    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]
        
