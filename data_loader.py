import torch
import numpy as np
import pickle


class Batch(object):

  """Class representing a minibatch of train/val/test examples for text summarization."""
  """Contain redundancy code"""
  def __init__(self, example_list, max_len):

    self.init_encoder(example_list, max_len) # initialize the input to the encoder
    self.init_decoder(example_list, max_len) # initialize the input and targets for the decoder
    self.init_result(example_list)
       
        
  def init_result(self, example_list):
    self.original_article=[]
    self.original_abstract=[]
    for ex in example_list:
        self.original_article.append(ex.original_article)
        self.original_abstract.append(ex.original_abstract)
        
        
  def init_encoder(self, example_list, max_len):
    self.enc_fact=[]
    self.grap_sim_bert=[]
    self.grap_sim_rouge=[]
    self.grap_entity=[]
    self.grap_cosent=[]
    
    for ex in example_list:
      ex.get_enc_fact(max_len)
      ex.get_grap()
      
    # Fill in the numpy arrays
    for ex in example_list:
      self.enc_fact.append(ex.enc_fact)
      self.grap_sim_bert.append(ex.grap_sim_bert)
      self.grap_sim_rouge.append(ex.grap_sim_rouge)
      self.grap_entity.append(ex.grap_entity)
      self.grap_cosent.append(ex.grap_cosent)      


  def init_decoder(self, example_list, max_len):
    self.dec_fact=[]     
    self.dec_label_bert=[]
    self.dec_label_rouge=[] 
    self.dec_score_bert=[]
    self.dec_score_rouge=[]    
    # Pad the inputs and targets
    for ex in example_list:
        ex.get_dec_fact(max_len)
        ex.get_dec_label_bert()
        ex.get_dec_label_rouge()
    # Fill in the numpy arrays
    for ex in example_list:
      self.dec_fact.append(ex.dec_fact)
      self.dec_label_bert.append(ex.dec_label_bert)
      self.dec_label_rouge.append(ex.dec_label_rouge)   
      self.dec_score_bert.append(ex.oral_score_bert)
      self.dec_score_rouge.append(ex.oral_score_rouge) 

def pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data



def filter_grap(grap, threshold):
    result=np.zeros((len(grap), len(grap)), dtype=np.float16)
    allx=[]
    for i in grap:
        for j in i:
            allx.append(j)
    allx.sort(reverse = True)
    threhold_number=allx[int(len(allx)*threshold)-1]
    for i,inum in enumerate(grap):
        for j,jnum in enumerate(inum):
            if jnum >= threhold_number:
                result[i][j]=1
    return result


def h_mask(src, s2f, clss, clsf, gs, gf, cls_id, seq_id, pad_id):
    
   
    mask=np.zeros((len(src), len(src)), dtype=np.float16)
    normal=np.zeros((len(src)), dtype=np.float16)
    for i in range (1,len(src)):
        if i not in clss and src[i]!=pad_id and i not in clsf:
            normal[i]=1        
    for i,tokeni in enumerate(src):
        if tokeni == pad_id:
            break
        elif i == 0:
            for j in clss:
                mask[i][j]=1
#                mask[j][i]=1
        elif tokeni == cls_id and i!=0:
            if i in clss:
                for inj,j in enumerate(clss):  
                    
                    mask[i][j]=gs[clss.index(i)][inj]
#                    mask[j][i]=gs[clss.index(i)][inj]                    
                    if i == j:
        
                        facts=s2f[inj]
                        for k in facts:
                            mask[i][clsf[k]] = 1
#                            mask[clsf[k]][i] = 1
                
                
            if i in clsf:
                for inj,j in enumerate(clsf):
                    mask[i][j]=gf[clsf.index(i)][inj]
#                    mask[j][i]=gf[clsf.index(i)][inj]
                    if i == j:
                        for k in range(i,i+100):
                            mask[i][k]=1
#                            mask[k][i]=1
                            if src[k]==seq_id:
                                break
        else:
            mask[i]=normal

    return mask
                    
            
def data_loader_ext(data_path,arg):
    '''
    input: data_path for the file
    output:src,seg,pos,clss,clsf,mask_clsf,mask_src,lab
    '''
    cls_id=101
    seq_id=102
    pad_id=0
    
    f=open(data_path,'rb')
    one_batch= pickle.load(f)
    
    #prepare alignment from sent to fact
    senttofactall=[]
    for i in one_batch.sent_to_fact:
        senttofact={}
        countx=0
        for index,x in enumerate(i):
            senttofact[index]=list(range(countx,countx+x))
            countx+=x
        senttofactall.append(senttofact) 
        
        
    #prepear src (token level)
    pre_src=[]
    sent_lenx=[]
    fact_lenx=[]
    cls_label=[]
    for num,one in enumerate(one_batch.enc_sent):
        count_len=0
        count_time_sent=0
        count_time_fact=0
        article=[]
        one_cls=[]
        one_split=one[:arg.max_src_nsents]
        article.append(cls_id)
        one_cls.append("clsd")
        s2f=senttofactall[num]     
        for i,sent in enumerate(one_split):
            fact_num=len(s2f[i])
            fact_len=0
            for fact_id in s2f[i]:
                fact_len+=len(one_batch.enc_fact[num][fact_id])            
            count_len+=(fact_len+1+fact_num*2)
            if count_len>511:
                break
            article.append(cls_id)  
            one_cls.append("clss")
            count_time_sent+=1
            for fact_id in s2f[i]:
                article.append(cls_id)
                one_cls.append("clsf")
                article+=one_batch.enc_fact[num][fact_id]
                one_cls+=len(one_batch.enc_fact[num][fact_id])*["no"]
                article.append(seq_id)
                one_cls.append("no")                
                count_time_fact+=1
        sent_lenx.append(count_time_sent)
        fact_lenx.append(count_time_fact)
        pre_src.append(article)
        cls_label.append(one_cls)
    
    src = torch.tensor(pad(pre_src, 0))    
        
    #prepear seg (token level)
    pre_seg=[]
    
    for one in cls_label:
        one_seg=[]
        for token in one:
            if token == 'clsd' or token == 'clsf':
#            if token == 'clsf' or token == 'clsf':
                one_seg.append(0)
            else:
                one_seg.append(1)
        pre_seg.append(one_seg)
        
    seg = torch.tensor(pad(pre_seg, 0))     
    
    #prepear pos (token level)
    pre_pos=[]
    if arg.no_pos == 0:
        for one in pre_src:
            if len(one) <=512:
                x=list(range(len(one)))
                pre_pos.append(x)
            else:
                x=list(range(512))
                x+=[511]*(len(one)-512)
                pre_pos.append(x)
    if arg.no_pos == 1:
        for one in pre_src:
            if len(one) <=512:
                x=list([0]*len(one))
                pre_pos.append(x)
            else:
                x=list([0]*len(one))
                pre_pos.append(x)

                
    pos = torch.tensor(pad(pre_pos, 0))    
        
    #prepear clss, clsf and mask_clsf
    #clss refer to sentence level 
    #clsf refer to fact level    
    pre_clss=[]
    for one in cls_label:
        x=[i for i, t in enumerate(one) if t == "clss"]
        pre_clss.append(x)
        
    clss=[]
    for num,one in enumerate(pre_clss):
        one_clss=[]
        s2f=senttofactall[num]
        for idx,i in enumerate(one):
            one_clss+=[i]*len(s2f[idx])
        clss.append(one_clss)
        
    clss = torch.tensor(pad(clss, -1))    
#    mask_clss = 1 - (clss == -1).int()
    clss[clss == -1] = 0        




    pre_clsf=[]
    for one in cls_label:
        x=[i for i, t in enumerate(one) if t == "clsf"]
        pre_clsf.append(x)
    
    clsf = torch.tensor(pad(pre_clsf, -1))    
    mask_clsf = 1 - (clsf == -1).int()
    clsf[clsf == -1] = 0 


    
    #prepear mask_src, graph mask on BERT
    #1 refer to there is an edge, while 0 represent this is no edge
    sent_g=one_batch.grap_sent
    fact_g=one_batch.grap_sim_rouge
    
    mask_src=[]
    for idx,i in enumerate(src):
        one_sent_g=sent_g[idx]
        one_fact_g=fact_g[idx]        
        graph_s=np.array([i[:sent_lenx[idx]] for i in one_sent_g[:sent_lenx[idx]]])
        graph_f=np.array([i[:fact_lenx[idx]] for i in one_fact_g[:fact_lenx[idx]]])        
        graph_s=filter_grap(graph_s, arg.threshold)
        graph_f=filter_grap(graph_f, arg.threshold)        
        
        one_mask = h_mask(i,senttofactall[idx],pre_clss[idx],pre_clsf[idx],graph_s,graph_f,cls_id, seq_id, pad_id)
        mask_src.append(one_mask)
#        mask_src.append(np.zeros((len(i), len(i)), dtype=np.float16))
    mask_src=torch.tensor(mask_src)
    
    
    #prepear lab
    pre_lab=[]
    for num,one in enumerate(one_batch.dec_label_rouge):
        pre_lab.append(one[:fact_lenx[num]])
    
    lab = torch.tensor(pad(pre_lab, 0))

    return src,seg,pos,clss,clsf,mask_clsf,mask_src,lab


def data_loader_ext_test(data_path,arg):
    '''
    Similar to data_loader_ext, but return more data related to evaluation
    '''
    cls_id=101
    seq_id=102
    pad_id=0
    
    f=open(data_path,'rb')
    one_batch= pickle.load(f)
    
    #prepare alignment from sent to fact
    senttofactall=[]
    for i in one_batch.sent_to_fact:
        senttofact={}
        countx=0
        for index,x in enumerate(i):
            senttofact[index]=list(range(countx,countx+x))
            countx+=x
        senttofactall.append(senttofact) 
        
        
    #prepear src
    pre_src=[]
    sent_lenx=[]
    fact_lenx=[]
    cls_label=[]
    for num,one in enumerate(one_batch.enc_sent):
        count_len=0
        count_time_sent=0
        count_time_fact=0
        article=[]
        one_cls=[]
        one_split=one[:arg.max_src_nsents]
        article.append(cls_id)
        one_cls.append("clsd")
        s2f=senttofactall[num]     
        for i,sent in enumerate(one_split):
            fact_num=len(s2f[i])
            fact_len=0
            for fact_id in s2f[i]:
                fact_len+=len(one_batch.enc_fact[num][fact_id])            
            count_len+=(fact_len+1+fact_num*2)
            if count_len>511:
                break
            article.append(cls_id)  
            one_cls.append("clss")
            count_time_sent+=1
            for fact_id in s2f[i]:
                article.append(cls_id)
                one_cls.append("clsf")
                article+=one_batch.enc_fact[num][fact_id]
                one_cls+=len(one_batch.enc_fact[num][fact_id])*["no"]
                article.append(seq_id)
                one_cls.append("no")                
                count_time_fact+=1
        sent_lenx.append(count_time_sent)
        fact_lenx.append(count_time_fact)
        pre_src.append(article)
        cls_label.append(one_cls)
    
    src = torch.tensor(pad(pre_src, 0))    
        
    #prepear seg
    pre_seg=[]
    
    for one in cls_label:
        one_seg=[]
        for token in one:
            if token == 'clsd' or token == 'clsf':
#            if token == 'clsf' or token == 'clsf':
                one_seg.append(0)
            else:
                one_seg.append(1)
        pre_seg.append(one_seg)
        
    seg = torch.tensor(pad(pre_seg, 0))   
        
    #prepear pos
    pre_pos=[]
    if arg.no_pos == 0:
        for one in pre_src:
            if len(one) <=512:
                x=list(range(len(one)))
                pre_pos.append(x)
            else:
                x=list(range(512))
                x+=[511]*(len(one)-512)
                pre_pos.append(x)
    if arg.no_pos == 1:
        for one in pre_src:
            if len(one) <=512:
                x=list([0]*len(one))
                pre_pos.append(x)
            else:
                x=list([0]*len(one))
                pre_pos.append(x)

                
    pos = torch.tensor(pad(pre_pos, 0))    
        
    #prepear clss and mask_clss clsf and mask_clsf
    pre_clss=[]
    for one in cls_label:
        x=[i for i, t in enumerate(one) if t == "clss"]
        pre_clss.append(x)
        
    clss=[]
    for num,one in enumerate(pre_clss):
        one_clss=[]
        s2f=senttofactall[num]
        for idx,i in enumerate(one):
            one_clss+=[i]*len(s2f[idx])
        clss.append(one_clss)
        
    clss = torch.tensor(pad(clss, -1))    
#    mask_clss = 1 - (clss == -1).int()
    clss[clss == -1] = 0        




    pre_clsf=[]
    for one in cls_label:
        x=[i for i, t in enumerate(one) if t == "clsf"]
        pre_clsf.append(x)
    
    clsf = torch.tensor(pad(pre_clsf, -1))    
    mask_clsf = 1 - (clsf == -1).int()
    clsf[clsf == -1] = 0 


    
    #prepear mask_src
    sent_g=one_batch.grap_sent
    fact_g=one_batch.grap_sim_rouge
    
    mask_src=[]
    for idx,i in enumerate(src):
        one_sent_g=sent_g[idx]
        one_fact_g=fact_g[idx]        
        graph_s=np.array([i[:sent_lenx[idx]] for i in one_sent_g[:sent_lenx[idx]]])
        graph_f=np.array([i[:fact_lenx[idx]] for i in one_fact_g[:fact_lenx[idx]]])        
        graph_s=filter_grap(graph_s, arg.threshold)
        graph_f=filter_grap(graph_f, arg.threshold)        
        
        one_mask = h_mask(i,senttofactall[idx],pre_clss[idx],pre_clsf[idx],graph_s,graph_f,cls_id, seq_id, pad_id)
        mask_src.append(one_mask)
#        mask_src.append(np.zeros((len(i), len(i)), dtype=np.float16))
    mask_src=torch.tensor(mask_src)
    
    
    #prepear lab
    pre_lab=[]
    for num,one in enumerate(one_batch.dec_label_rouge):
        pre_lab.append(one[:fact_lenx[num]])
    
    lab = torch.tensor(pad(pre_lab, 0)) 
    #prepear testing
    article=one_batch.original_article
    abstract=one_batch.original_abstract
    

                    
            
    return src,seg,pos,clss,clsf,mask_clsf,mask_src,lab,article,abstract,one_batch.dec_label_rouge
    
    