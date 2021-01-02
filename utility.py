from copy import deepcopy as cp
from stanfordcorenlp import StanfordCoreNLP
import numpy as np




def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

    
def fact_merge(rlist): #redundency
    merged={}
    for i in rlist:
        forward = i['subject']+' '+i['relation']
        backward = i['object']
        if forward not in merged.keys():
            merged[forward]=backward
        else:
            ex=merged[forward]
            if len(ex) < len(backward):
                merged[forward]=backward
    merged2={}
    for i in merged.keys():
        forward = i
        backward = merged[i]
        if backward not in merged2.keys(): 
            merged2[backward]=forward            
        else:
            ex=merged2[backward]
            if len(ex) < len(forward):
                merged2[backward]=forward  
                
    return_list=[]
    for i in merged2.keys():
        x=merged2[i]+' '+i
        return_list.append(x)
    return return_list

def merge(ilist):   #redundency
    rlist=[]
    root=ilist[0]
    ilist.remove(root)
    while(True):
        if ilist == []:
            rlist.append(root)            
            break
        record=[]
        for i in ilist:
            if list(set(root) & set(i)) != []:
                root=list(set(root+i))
                record=record+i
                break
        if record == []:
            rlist.append(root)
            root=ilist[0]
            ilist.remove(root)
        else:
            ilist.remove(record)
        
    return rlist


def fact_parse(token, parsing):     #redundency
    predicate=['nsubj','nsubjpass', 'csubj', 'csubjpass','dobj']
    modify=['amod','nummod','compound','ccomp']
    predicate_tuple=[]
    modify_tuple=[]
    for i in parsing:
        if i[0] in predicate:
            predicate_tuple.append([i[1],i[2]])
        if i[0] in modify:
            modify_tuple.append([i[1],i[2]])
    tuple_merge=predicate_tuple+modify_tuple
    tuple_merge=sorted(tuple_merge, key=lambda x: x[0]) 
    print(tuple_merge)
    if tuple_merge == []:
        return []
    tuple_merge=merge(tuple_merge)
    print(tuple_merge)
    result=[]
    for i in tuple_merge:
        one=''
        if len(i) <=3:
            continue
        for j in i:
            one+=token[j]
            one+=' '
        result.append(one.strip().replace(',','').replace('.',''))
    return result

def word_len(sent):
    return len((sent.strip()).split(' '))

def list_in(listk, sent):
    for i in listk:
        if i in sent:
            return True
    return False

def sent_split(sent,nlp):  #sent is a sentence text string and nlp is stanfordcorenlp
    keyword1=['punct','cc','mark']      #split
    keyword2=['acl:relcl','advcl','appos','ccomp']     #merge 

    min_length=4
    max_length=8
    conj_min_length=4

    token=nlp.word_tokenize(sent)
    token=['ROOT']+token
    parsing=nlp.dependency_parse(sent)
    
    
    
    
    split_pos=[]
    
    for i in parsing:
        if i[0] in keyword1:            
            if i[0]=='cc' and (i[2]-i[1]) > conj_min_length:
                x=i[2]
                split_pos.append([x,0])
                break
            
            elif i[0] == 'punct':
                x=i[2]
                tag=0
                for j in parsing:
                    if j[0] in keyword2:
                        if j[1]<x and j[2] >x:
                            tag=1
                            break
                if tag==0:
                    split_pos.append([x,tag])
                else:
                    split_pos.append([x,tag])                    
                
            elif i[0] == 'mark':
                x=i[1]
                tag=0
                for j in parsing:
                    if j[0] in keyword2:
                        if j[1]<x and j[2] >x:
                            tag=1
                            break
                if tag==0:
                    split_pos.append([x,tag])
                else:
                    split_pos.append([x,tag])       
                    
            else:
                pass


    if len(split_pos) == 0:
        return [sent]
    else:
        tag_list=[]
        raw_split_sent=[]
        
        pointer=1
        for i in split_pos:
            pos=i[0]
            tag_list.append(i[1])
            
            subsent=' '.join(token[pointer:pos])
            raw_split_sent.append(subsent)
            
            pointer=pos+1
        
        raw_split_sent.append(' '.join(token[pointer:]))

        sent_result=[]
        for i,subsent in enumerate(raw_split_sent):
            itoken=subsent.strip()
            if i == 0:
                sent_result.append(itoken)
            else:
                tag_=tag_list[i-1]
                if word_len(itoken) <=min_length:
                    sent_result[-1]+=' , '+itoken
                elif tag_== 1 and word_len(itoken) > max_length:
                    sent_result.append(itoken)
                elif tag_== 1 and word_len(itoken) <= max_length:
                    sent_result[-1]+=' , '+itoken               
                else:
                    sent_result.append(itoken)
        if (word_len(sent_result[0]) <=min_length) and len(sent_result)>=2:
            sent_result[1]=sent_result[0]+' , '+sent_result[1]
            sent_result.remove(sent_result[0])     
            
        return sent_result
            
        



'''
#test code
from stanfordcorenlp import StanfordCoreNLP

x='Ahmadinejad essentially called Yukiya Amano, the director general of the IAEA, a U.S. puppet and said the U.N.A has no jurisdiction in Iran and Irap'

nlp=StanfordCoreNLP('')

print(sent_tokenize(x,nlp))
'''