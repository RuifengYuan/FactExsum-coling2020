import glob
import random
import struct
from tensorflow.core.example import example_pb2
import json
import re

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",

         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"', "\n": ''}

def clean(x):
    return re.sub(r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", lambda m: REMAP.get(m.group()), x)

def clean_str(s):
    forbidden=['b"','-lrb-','-rrb-','-','“','"',"'","`",'``',"''","b'",'/','\\','\\n','-','<s>','</s>']
    for i in forbidden:
        s=s.replace(i,'')
    return s

def end_replace(s):
    forbidden=['!', '?',';',':']
    for i in forbidden:
        s=s.replace(i,'.')
   
    return s


def example_generator_TLDR(data_path):
  reader = open(data_path, 'rb')
  while True:
      token=reader.readline()
      if token:
          x=json.loads(token.decode())
          src=x['content']
          ref=x['summary']
          src=clean_str(src)
          ref=clean_str(ref)
          src=end_replace(src)
          ref=end_replace(ref)
          article=src.split('.')
          summary=ref.split('.')
          if '' in summary:
            summary.remove('')
          if '' in article:
            article.remove('')
          yield [article,summary]

      else:
          break
      
        
        
        
def example_generator_MUTIL(data_path):
  reader_src = open((data_path+'.txt.src.tokenized.fixed.cleaned.final.truncated.txt'), 'rb')
  reader_ref = open((data_path+'.txt.tgt.tokenized.fixed.cleaned.final.truncated.txt'), 'rb')
  while True:
      src=reader_src.readline()
      ref=reader_ref.readline()
      if src and ref:
          src=src.decode()
          ref=ref.decode()
          src=clean(src)
          ref=clean(ref)
          src=end_replace(src)
          ref=end_replace(ref)
          multi_src=src.split('story_separator_special_tag')
          multi_article=[]
          for one_src in multi_src:
              article=one_src.split('.')
              if '' in article:
                article.remove('')
              multi_article.append(article)
          summary=ref.split('.')
          if '' in summary:
            summary.remove('')
          yield [multi_article,summary]
      else:
          break

      
def example_generator_DMCNN(data_path):

  """Generates Examples from data files.
  Args:
    data_path:
      Path to data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
    single_pass:
      Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.
  Yields:
    [src(list),ref(list)]
  """

  while True:
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    filelist = sorted(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        e=example_pb2.Example.FromString(example_str)
        article_text = e.features.feature['article'].bytes_list.value[0]
        abstract_text = e.features.feature['abstract'].bytes_list.value[0]
        article_text=article_text.decode()
        abstract_text=abstract_text.decode()
        src= end_replace(article_text)
        ref= end_replace(abstract_text)
        src=clean(src)
        ref=clean(ref)
        src=src.replace('[','(')
        src=src.replace(']',')')
        ref=ref.replace('[','(')
        ref=ref.replace(']',')')
        article=src.split('.')
        summary=ref.split('.')
        summary_f=[]
        for i in summary:
            if len(i) > 25:
                summary_f.append(i.strip())
        article_f=[]
        for i in article:
            if len(i) > 25:
                article_f.append(i.strip())
        if article_f==[] or summary_f==[]:
            pass
        else:
            yield [article_f,summary_f]  
    print ("example_generator completed reading all datafiles. No more data.")
    break

