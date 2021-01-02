import queue
from threading import Thread
import numpy as np
from transformers import *
from openie import StanfordOpenIE
from utility.utility import *
from bert_serving.client import BertClient
from rouge import Rouge
from stanfordcorenlp import StanfordCoreNLP
import pickle
from data.raw_data_loader import *
'''
nlp = StanfordCoreNLP('/home/ziqiang/stanfordnlp_resources/stanford-corenlp-full-2018-10-05')
bc = BertClient(ip='localhost')
client = StanfordOpenIE()
rougex = Rouge()
'''
g_b=0


import threading  
  
class threadsafe_generator:  
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()
  


class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, article, abstract, tokenizer, rougex, nlp):

    """
    Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.
    Args:
      article: source text; list of strings. each token is separated by a single space.
      abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
      vocab: Vocabulary object
      hps: hyperparameters
    """
    self.rougex=rougex
    self.nlp=nlp
    self.tokenizer=tokenizer
    article=article[:60]
    self.article=article
    self.abstract=abstract
    # Process the article
    self.article_fact=[]
    self.article_sent=[]
    self.article_fact_tag=[]
    for count,sent in enumerate(article):
        
        self.article_sent.append(self.tokenizer.encode(sent))
        sent=sent.strip(',')
        sent=sent.strip(':')

        sentfact=sent_split(sent,self.nlp)
                
        sentfact_file=[]
        for i in sentfact:
            if word_len(i) >50:
                ii=i.split(' ')
                ii=ii[0:50]
                sentfact_file.append(' '.join(ii))
                continue
            if len(i) >= 20:
                sentfact_file.append(i)   
                
        self.article_fact_tag.append(len(sentfact_file))
        self.article_fact+=sentfact_file
    self.article_id=[]
    for fact in self.article_fact:
        self.article_id.append(self.tokenizer.encode(fact,add_special_tokens=False))
    self.article_len = len(self.article_id) # store the number of sentences of the article 
    # Process the abstract
    self.original_abstract=[]
    self.abstract_fact=[]
    self.abstract_fact_all=[]    
    for sent in abstract:
        
        self.original_abstract.append(self.tokenizer.encode(sent))
        
        if word_len(sent) > 20:
            sent=sent.strip(',')
            sent=sent.strip(':')
            sentfact=sent_split(sent,self.nlp)
        else:
            sentfact=[sent]
                
        self.abstract_fact_all+=sentfact
        
    for i in self.abstract_fact_all:
        if word_len(i) >50:
            ii=i.split(' ')
            ii=ii[0:50]
            self.abstract_fact.append(' '.join(ii))  
        elif len(i) < 15:
            continue            
        else:
            self.abstract_fact.append(i)
    self.abstract_id=[]
    for fact in self.abstract_fact:
        self.abstract_id.append(self.tokenizer.encode(fact,add_special_tokens=False))

    self.abstract_len = len(self.abstract_id) # store the number of sentences of the article

    

    self.enc_fact=[]
    self.enc_sent=[]
    self.dec_fact=[] 
    
    self.dec_label_bert=[]
    self.dec_label_rouge=[]
    self.dec_label_sent=[]
    
    self.grap_sim_bert=np.zeros((self.article_len, self.article_len), dtype=np.float16)
    self.grap_sim_rouge=np.zeros((self.article_len, self.article_len), dtype=np.float16)
    self.grap_entity=np.zeros((self.article_len, self.article_len), dtype=np.float16)
    self.grap_cosent=np.zeros((self.article_len, self.article_len), dtype=np.float16)    
    self.grap_sent=np.zeros((len(self.article), len(self.article)), dtype=np.float16)
  def get_enc_fact(self, max_len):   
      
    """Pad the encoder input sequence with pad_id up to max_len."""
    
    for i in self.article_id:
        if len(i) > max_len:
            self.enc_fact.append(i[0:max_len])
        else:
            self.enc_fact.append(i)
            
            
    for i in self.article_sent:
        if len(i) > max_len*2:
            self.enc_sent.append(i[0:max_len*2])
        else:
            self.enc_sent.append(i)        
        
  def get_dec_fact(self, max_len):   
      
    """Pad the encoder input sequence with pad_id up to max_len."""
    
    for i in self.abstract_id:
        if len(i) > max_len:
            self.dec_fact.append(i[0:max_len])
        else:
            self.dec_fact.append(i)         
      
  def get_grap(self):
      
      """Get the sim bert graph """
      
      """Get the sim rouge graph """     
      for i,facti in enumerate(self.article_fact):
          for j,factj in enumerate(self.article_fact):
              scores = self.rougex.get_scores(facti, factj)
              self.grap_sim_rouge[i][j]=(scores[0]['rouge-1']['f']+scores[0]['rouge-2']['f'])/2      

      """Get the sim sent graph """     
      for i,facti in enumerate(self.article):
          for j,factj in enumerate(self.article):
              scores = self.rougex.get_scores(facti, factj)
              self.grap_sent[i][j]=(scores[0]['rouge-1']['f']+scores[0]['rouge-2']['f'])/2    
            
      """Get the entity graph"""
          
      """Get the co-sent graph"""
      now=0
      for i in self.article_fact_tag:
          for x in range(now+i)[now:now+i]:
              for y in range(now+i)[now:now+i]:
                  self.grap_cosent[x][y]=1
          now=now+i
                  
          
                   
  def get_dec_label_bert(self):
      self.dec_label_bert=[]
      self.oral_score_bert=0

          
  def get_dec_label_rouge(self):
      rouge=[]
      score_rouge=[]
      index_rouge=[]
      for j in self.abstract_fact:
          score=[]
          for k in self.article_fact:
              scores = self.rougex.get_scores(j, k)
              score.append((scores[0]['rouge-1']['f']+scores[0]['rouge-2']['f'])/2)
          choose=score.index(max(score))
          index_rouge.append(choose)
          rouge.append(self.article_fact[choose])
          score_rouge.append(max(score))   
      for i in range(len(self.article_fact)):
          if i in index_rouge:
              self.dec_label_rouge.append(1)
          else:
              self.dec_label_rouge.append(0) 

      self.oral_score_rouge = self.rougex.get_scores(' . '.join(rouge), ' . '.join(self.abstract))
      
  def get_dec_label_rouge_sent(self):
    get_dec_label_sent=self.greedy_selection(self.article, self.abstract, 3, self.rougex)
    for i in range(len(self.article)):
      if i in get_dec_label_sent:
          self.dec_label_sent.append(1)
      else:
          self.dec_label_sent.append(0)         
        
  def greedy_selection(self, doc_sent_list, abstract_sent_list, summary_size, rougex):
    selected = []
    max_rouge = 0.0
    reference=''
    for i in abstract_sent_list:
        reference+=i
        reference+=' . '
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(doc_sent_list)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates = ''
            for j in c:
                candidates+=doc_sent_list[j]
                candidates+=' . '
            scores = rougex.get_scores(candidates, reference)
            rouge_score = (scores[0]['rouge-1']['f']+scores[0]['rouge-2']['f'])/2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge
    return sorted(selected)


class Batch(object):

  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, max_len):

    """
    Turns the example_list into a Batch object.
    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """

    self.init_encoder(example_list, max_len) # initialize the input to the encoder
    self.init_decoder(example_list, max_len) # initialize the input and targets for the decoder
    self.init_result(example_list)
       
        
  def init_result(self, example_list):
    self.original_article=[]
    self.original_abstract=[]
    self.original_sent=[]
    self.sent_to_fact=[]
    for ex in example_list:
        self.original_sent.append(ex.article)
        self.original_article.append(ex.article_fact)
        self.original_abstract.append(ex.abstract)
        self.sent_to_fact.append(ex.article_fact_tag)
        
  def init_encoder(self, example_list, max_len):
    self.enc_fact=[]
    self.enc_sent=[]
    self.grap_sim_bert=[]
    self.grap_sim_rouge=[]
    self.grap_entity=[]
    self.grap_cosent=[]
    self.grap_sent=[]
    
    for ex in example_list:
      ex.get_enc_fact(max_len)
      ex.get_grap()
      
    # Fill in the numpy arrays
    for ex in example_list:
      self.enc_fact.append(ex.enc_fact)
      self.enc_sent.append(ex.enc_sent)
      self.grap_sim_bert.append(ex.grap_sim_bert)
      self.grap_sim_rouge.append(ex.grap_sim_rouge)
      self.grap_entity.append(ex.grap_entity)
      self.grap_cosent.append(ex.grap_cosent)      
      self.grap_sent.append(ex.grap_sent)      

  def init_decoder(self, example_list, max_len):
    self.dec_fact=[]  
    self.dec_label_sent=[] 
    self.dec_label_bert=[]
    self.dec_label_rouge=[] 
    self.dec_score_bert=[]
    self.dec_score_rouge=[]    
    # Pad the inputs and targets
    for ex in example_list:
        ex.get_dec_fact(max_len)
        ex.get_dec_label_bert()
        ex.get_dec_label_rouge()
        ex.get_dec_label_rouge_sent()
    # Fill in the numpy arrays
    for ex in example_list:
      self.dec_fact.append(ex.dec_fact)
      self.dec_label_sent.append(ex.dec_label_sent)
      self.dec_label_bert.append(ex.dec_label_bert)
      self.dec_label_rouge.append(ex.dec_label_rouge)   
      self.dec_score_bert.append(ex.oral_score_bert)
      self.dec_score_rouge.append(ex.oral_score_rouge) 


class Batcher(object):

  """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, dataset):

    """Initialize the batcher. Start threads that process the data into batches.
    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary object
      hps: hyperparameters
      single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
    """
    self._dataset=dataset
    self._data_path = data_path
    self._max_len=50
    self._batch_size=4

    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self._batch_size)
    # Initialize the tool
    self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    self.rougex=Rouge()
    self.nlp=StanfordCoreNLP('/home/ziqiang/stanfordnlp_resources/stanford-corenlp-full-2018-10-05')
    # Different settings depending on whether we're in single_pass mode or not
    self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
    self._num_batch_q_threads = 1  # just one thread to batch examples
    self._bucketing_cache_size = 50 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
    self._finished_reading = False # this will tell us when we're finished reading the dataset
    #prepear dataloader
    if self._dataset == 'TLDR':
        self.input_gen = threadsafe_generator(example_generator_TLDR(self._data_path))
    if self._dataset == 'MUTIL':
        self.input_gen = threadsafe_generator(example_generator_MUTIL(self._data_path))
    if self._dataset == 'DMCNN':
        self.input_gen = threadsafe_generator(example_generator_DMCNN(self._data_path))

    print('finish prepearing')
    # Start the threads that load the queues


    self._example_q_threads = []
    for _ in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()

    self._batch_q_threads = []
    for _ in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
#      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    print('threads started')

  def next_batch(self):

    """
    Return a Batch from the batch queue.
    batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
    """
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
#      tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      pass
      if self._finished_reading and self._example_queue.qsize() == 0:
        print("Finished reading dataset in single_pass mode.")
        return None
    batch = self._batch_queue.get() # get the next Batch
    return batch

  def fill_example_queue(self):

    """Reads data from file and processes into Examples which are then placed into the example queue."""
    global g_b
    while True:
      g_b+=1
      if g_b%100==0:
          print('--------'+str(g_b)+'--------')
          print(self._example_queue.qsize())
          print(self._batch_queue.qsize())
      try:
        article, abstract = self.input_gen.__next__() # read the next example from file. article and abstract are both strings.
      except StopIteration: # if there are no more examples:
        print("The example generator for this example queue filling thread has exhausted data.")
        self._finished_reading = True
        break
      example = Example(article, abstract, self.tokenizer, self.rougex, self.nlp) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.
      
      
  def fill_batch_queue(self):

    """
    Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.
    In decode mode, makes batches that each contain a single example repeated.
    """
    while True:
    # Get bucketing_cache_size-many batches of Examples into a list, then sort
      inputs = []
      for _ in range(self._batch_size * self._bucketing_cache_size):
          if  self._finished_reading and self._example_queue.qsize() == 0:
              break
          inputs.append(self._example_queue.get())
      # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
      inputs.sort(key=self.get_sort)
      '''
      splits = []
      len_pre=-1
      for indexi,i in enumerate(inputs):
          len_now = i.article_len
          if len_pre != len_now:
              splits.append(indexi)
          len_pre=len_now
      batches=[]
      for indexi,i in enumerate(splits):
          if indexi+1 == len(splits):
              batches.append(inputs[i:])
          else:
              batches.append(inputs[i:splits[indexi+1]])
      batches_max=[]
      for i in batches:
          if len(i) <= self._batch_size:
              batches_max.append(i)
          else:
              batches_max+=[i[j:j+self._batch_size] for j in range(0, len(i), self._batch_size)]
      '''
      batches_max=[]
      for indexi,i in enumerate(inputs):
          if indexi % self._batch_size ==0:
              batches_max.append(inputs[indexi:indexi+self._batch_size])
      for b in batches_max:  # each b is a list of Example objects
        self._batch_queue.put(Batch(b, self._max_len))
        
  def get_sort(self, x):
      return x.article_len

'''
train_data_loader=Batcher('data/DMCNN/train_*', 'DMCNN')

count=0
countx=0
while True:
    batch = train_data_loader.next_batch()
    each_batch_size=len(batch.enc_fact)
    if train_data_loader._finished_reading == True:
        break
    f=open('data_file/DMCNN/train_file/'+str(count)+'_train_batch_of '+str(each_batch_size)+' examples.pkl','wb')  
    pickle.dump(batch,f)  
    f.close() 
    count+=1
    countx+=each_batch_size
print('Total train data:')
print(countx)


train_data_loader=Batcher('data/DMCNN/val_*', 'DMCNN')

count=0
countx=0
while True:
    batch = train_data_loader.next_batch()
    each_batch_size=len(batch.enc_fact)
    if train_data_loader._finished_reading == True:
        break
    f=open('data_file/DMCNN/val_file/'+str(count)+'_val_batch_of '+str(each_batch_size)+' examples.pkl','wb')  
    pickle.dump(batch,f)  
    f.close() 
    count+=1
    countx+=each_batch_size
print('Total val data:')
print(countx)
'''


train_data_loader=Batcher('data/DMCNN/test*', 'DMCNN')

count=0
countx=0
while True:
    batch = train_data_loader.next_batch()
    each_batch_size=len(batch.enc_fact)
    if train_data_loader._finished_reading == True and train_data_loader._batch_queue.qsize() == 0 and train_data_loader._example_queue.qsize() == 0:
        break
    f=open('data_file/DMCNN/test_file_y/'+str(count)+'_test_batch_of '+str(each_batch_size)+' examples.pkl','wb')  
    pickle.dump(batch,f)  
    f.close() 
    count+=1
    countx+=each_batch_size
print("test*")
print('Total test data:')
print(countx)









      