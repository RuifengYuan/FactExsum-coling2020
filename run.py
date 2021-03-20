from pytorch_pretrained_bert.optimization import BertAdam

from pytorch_pretrained_bert.tokenization import BertTokenizer

from utility.utility import *

from model.BERT import *

import torch

import torch.nn as nn

import torch.nn.functional as F

import glob,time,gc,heapq, argparse, random

from data_loader import data_loader_ext, Batch, data_loader_ext_test

import numpy as np

from rouge import Rouge

from call_rouge import test_rouge, rouge_results_to_str

import collections


def train(config):

    if config.pre_train == 0:
        net = ExtSummarizer()
    else:
        x=torch.load('save_model/DMCNN/'+config.pre_model,map_location='cpu') #train based on checkpoint
        net=x['model']

    if torch.cuda.is_available():
        net = net.cuda()

    A = []

    Q = []

    best_vloss = 100

    best_acc = -1

    lRate = config.lRate

    num_train_optimization_steps = 100000

    step=0

    if config.warmup == 0 or config.warmup == 2:
        
        param_optimizer = list(net.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=lRate,
                             e=1e-9,                       
                             t_total=num_train_optimization_steps,
                             warmup=0)
    if config.warmup == 1:
        
        param_optimizer = list(net.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=lRate,
                             e=1e-9,
                             schedule='warmup_linear',
                             t_total=num_train_optimization_steps,
                             warmup=0.1)


    accumelate=0
    for epoch_idx in range(config.max_epoch):

        train_filelist = glob.glob(config.train_path)
        random.shuffle(train_filelist)
        vaild_filelist = glob.glob(config.vaild_path)
        

        for batch_idx, batch_path in enumerate(train_filelist):
            
            try:

                if (batch_idx + 1) % 5000 == 0:
                    gc.collect()
            
    
                start_time = time.time()
                net.train()
                src,seg,pos,clss,clsf,mask_clsf,mask_src,lab = data_loader_ext(batch_path,config)
                batch_len=lab.size()[0]
                sent_len=lab.size()[1]
                
                if sent_len == 0:
                    continue
                                
                src = src.cuda()
                pos = pos.cuda()
                seg = seg.cuda()
                clss = clss.cuda()
                clsf = clsf.cuda()
                mask_clsf = mask_clsf.cuda()
                mask_src = mask_src.cuda()
                lab = lab.cuda()
                
    
                    
                #lossFunc = nn.BCEWithLogitsLoss(weight=mask_clss.float())
                lossFunc = nn.BCEWithLogitsLoss(weight=mask_clsf.float(),pos_weight=torch.tensor([config.pos_weight]*sent_len))     
                lossFunc = lossFunc.cuda()
                
                predicts, mask_clssx= net(src, pos, seg, mask_src,clss,clsf, mask_clsf)
                sigmoid = nn.Sigmoid()
                predicts_score=sigmoid(predicts)*mask_clssx.float()
                lab=lab.type_as(predicts)
                tloss = lossFunc(predicts, lab).sum()
                
                if bool(10 > float(tloss) > 0):
                    pass
                else:
                    print('error loss')
                    continue
                
                Q.append(float(tloss))
                if len(Q) > 400:
                    Q.pop(0)
                loss_avg = sum(Q) / len(Q)
                
                acc=[]
                for example_id, example in enumerate(predicts_score): #caculate accuracy
                    gold = [i for i,ii in enumerate(lab[example_id]) if ii == 1]
                    answer_num = heapq.nlargest(config.ext_num, list(example))
                    answer=[i for i,ii in enumerate(example) if ii in answer_num]
                    accuracy=0
                    for i in answer[:config.ext_num]:
                        if i in gold:
                            accuracy+=1/config.ext_num 
                    acc.append(accuracy)            
                A.append(np.mean(acc))
                if len(A) > 400:
                    A.pop(0)
                acc_avg = sum(A) / len(A)
            
                try:   #update the network and learn rate
                    if accumelate>=config.train_size:
                        (tloss).backward()
                        optimizer.step()                            
                        optimizer.zero_grad()
                        accumelate=0
                        step=step+1
                        if config.warmup == 2:
                            cur_lr = config.lRate * 100 * min(step ** (-0.5), step * 10000**(-1.5))
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = cur_lr
                    else:
                        (tloss).backward()
                        accumelate+=batch_len
                except:
                    print('optimization fail')
                
                print('Epoch %2d, Batch %6d -0, Loss %9.6f, Average Loss %9.6f,acc %9.6f, Average acc %9.6f, Time %9.6f' % (epoch_idx + 1, batch_idx + 1, tloss, loss_avg,np.mean(acc),acc_avg, time.time() - start_time))
                
            except:
                print('train fail')

            # Checkpoints

            idx = epoch_idx * config.train_set_len + batch_idx + 1

            if (idx >= config.checkmin) and (idx % config.checkfreq == 0):
                
                print("start validation")

                vloss = []
                
                acc=[]

                for val_id, val_path in enumerate(vaild_filelist[0:1500]):
                    
                    src,seg,pos,clss,clsf,mask_clsf,mask_src,lab = data_loader_ext(val_path,config)
                    sent_len=lab.size()[1]
            
                    if sent_len == 0:
                        continue

                    lossFunc = nn.BCEWithLogitsLoss(weight=mask_clsf.float(),pos_weight=torch.tensor([config.pos_weight]*sent_len))
                    
                    lossFunc = lossFunc.cuda()
                    
                    src = src.cuda()
                    pos = pos.cuda()
                    seg = seg.cuda()
                    clss = clss.cuda()
                    clsf = clsf.cuda()
                    mask_clsf = mask_clsf.cuda()
                    mask_src = mask_src.cuda()
                    lab = lab.cuda()
                    
                    with torch.no_grad():
    
                        net.eval()
    
                        predicts, mask_clssx= net(src, pos, seg, mask_src,clss,clsf, mask_clsf)
                        sigmoid = nn.Sigmoid()
                        predicts_score=sigmoid(predicts)*mask_clssx.float()
                        lab=lab.type_as(predicts)    
                        loss=float(lossFunc(predicts, lab[0:4]).sum())
                        
                        if bool(float(loss) > 10):
                            print('error loss')
                            continue
                        
                        vloss.append(loss)
                       
                    for example_id, example in enumerate(predicts_score):
                        gold = [i for i,ii in enumerate(lab[0:4][example_id]) if ii == 1]
                        answer_num = heapq.nlargest(config.ext_num, list(example))
                        answer=[i for i,ii in enumerate(example) if ii in answer_num]
                        accuracy=0
                        for i in answer[:config.ext_num]:
                            if i in gold:
                                accuracy+=1/config.ext_num
                        acc.append(accuracy)


                num_loss=sum(vloss) / len(vloss)  
                acc_loss=sum(acc) / len(acc)
                
                is_best = num_loss < best_vloss or acc_loss > best_acc
                
                best_vloss = min(num_loss, best_vloss)
                best_acc = max(acc_loss, best_acc)
              
                print('CheckPoint: Validation Loss %11.8f, Validation acc %11.8f,Best num Loss %11.8f, Best acc Loss %11.8f' % (num_loss, acc_loss, best_vloss, best_acc))
                if is_best:
                    print('best!!!!')
                if True:
                    torch.save({'epoch': epoch_idx, 'vloss':num_loss, 'acc':acc_loss,'model':net},
                           'save_model/' + 'DMCNN/' +str(config.device)+'---'+str(epoch_idx)+'-'+str(idx)+ '-' + str("%.4f" % num_loss) + '-' + str("%.4f" % acc_loss) + '.pth.tar')

        print('Epoch Finished.')



def test(config):
    
    
    # tri-blocking
    def _get_ngrams(n, text):

        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set


    # tri-blocking
    def _block_tri(c, p):

        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False
    
    rougex=Rouge()
    
    x=torch.load('save_model/'+config.test_model,map_location='cpu')

    net=x['model']

    sigmoid = nn.Sigmoid()

    if torch.cuda.is_available():

        net = net.cuda()
        
    acc = []

    test_filelist = glob.glob(config.test_path)
        

    can_path = 'result/'+config.test_model+'_cand.txt'

    gold_path ='result/'+config.test_model+'_gold.txt'

    rouge1=[]
    rouge2=[]
    
    acc=[]
    
    tloss=[]
    
    an=[]
    an_num=0
    with open(can_path, 'w') as save_pred:
        with open(gold_path, 'w') as save_gold:
            
            for batch_idx, batch_path in enumerate(test_filelist):

                selected_ids=[]
                src,seg,pos,clss,clsf,mask_clsf,mask_src,lab,fact_article,abstract,true_lab = data_loader_ext_test(batch_path,config)                
                gold = []
                pred = []
                if config.test_cal == 'lead':
                    selected_ids=[list(range(8))] * clss.size(0)
                elif config.test_cal == 'oracle':
                    for example_id, example in enumerate(true_lab):
                        if sum(example) == 0:
                            print('no golden one')
                            goldx=[1,2,3,4]
                        else:
                            goldx = [i for i,ii in enumerate(example) if ii == 1] 
                        selected_ids.append(goldx)
                else:

                    src = src.cuda()
                    pos = pos.cuda()
                    seg = seg.cuda()
                    clss = clss.cuda()
                    clsf = clsf.cuda()
                    mask_clsf = mask_clsf.cuda()
                    mask_src = mask_src.cuda()
                    lab = lab.cuda()
                    

                    lossFunc = nn.BCEWithLogitsLoss(weight=mask_clsf.float())   
                    lossFunc = lossFunc.cuda()
                    with torch.no_grad():
                        net.eval()


                        predicts, mask_clssx= net(src, pos, seg, mask_src,clss,clsf, mask_clsf)
                        predicts_score=sigmoid(predicts)*mask_clssx.float()
                        lab=lab.type_as(predicts)
                        loss = float(lossFunc(predicts, lab).sum())
                        tloss.append(loss)                                        
    
                        for example_id, example in enumerate(predicts_score):
                            goldxx = [i for i,ii in enumerate(lab[example_id]) if ii == 1]
                            answer_num = heapq.nlargest(config.ext_num, list(example))
                            answer=[i for i,ii in enumerate(example) if ii in answer_num]
                            accuracy=0
                            for i in answer[:config.ext_num]:
                                if i in goldxx:
                                    accuracy+=1/config.ext_num 
                            acc.append(accuracy)
                            sent_scores = example.cpu().data.numpy()
                            selected_ids.append(np.argsort(-sent_scores))

                if config.test_cal == 'lead':
                    article=fact_article
                else:
                    article=fact_article
                for i, zz in enumerate(selected_ids):
                    _pred = []
                    if config.test_cal == 'oracle':
                        for z in zz:
                            _pred.append(article[i][z])
                    else:
                        if (len(article[i]) == 0):
                            continue
                        for j in selected_ids[i][:len(article[i])]:
                            if (j >= len(article[i])):
                                continue
                            
                            candidate = article[i][j].strip()
                            
                            if (config.block_trigram == 1):
                                if (not _block_tri(candidate, _pred)):
                                    _pred.append(candidate)
                                    an.append(j)
                                    an_num+=1
                            else:
                                _pred.append(candidate)
                                an.append(j)
                                an_num+=1
                                
                            if (len(_pred) == config.ext_num):
                                break
                    _pred = ' <q> '.join(_pred)
                    _gold = ' <q> '.join(abstract[i])
                    _gold = _gold.replace('<s>','').replace('</s>','')
                    if (config.recall_eval == 1):
                        _pred = ' '.join(_pred.split()[:len(_gold.split())]) 
                    pred.append(_pred)
                    gold.append(_gold)                    
                    scores = rougex.get_scores(_pred, _gold)
                    rouge1.append(scores[0]['rouge-1']['f'])
                    rouge2.append(scores[0]['rouge-2']['f'])
                for sent in gold:
                    save_gold.write(sent.strip() + '\n')
                for sent in pred:
                    save_pred.write(sent.strip() + '\n')
                break

    #rouges = test_rouge('result/rouge', can_path, gold_path)
    '''
    print(rouge_results_to_str(rouges))
    print(np.mean(rouge1))
    print(np.mean(rouge2))
    print('accuracy:')
    print(np.mean(acc))
    print('loss:')
    print(np.mean(tloss))
    print('distribution:')
    obj1 = collections.Counter(an)
    print(obj1)
    print("select number:")
    print(an_num)
    '''



        
        
def argLoader():

    parser = argparse.ArgumentParser()
    
    
    #device
    
    parser.add_argument('--device', type=int, default=0)    
    
    # Do What
    
    parser.add_argument('--do_train', action='store_true', help="Whether to run training")

    parser.add_argument('--do_test', action='store_true', help="Whether to run test")

    # Options Setting
    parser.add_argument('--warmup', type=int, default=2)    
    
    parser.add_argument('--pos_weight', type=int, default=1)

    parser.add_argument('--max_src_nsents', type=int, default=80)

    parser.add_argument('--grap', type=str, default='rouge')

    parser.add_argument('--train_set_len', type=int, default=66936)

    parser.add_argument('--max_epoch', type=int, default=10)
    
    parser.add_argument('--lRate', type=float, default=2e-5)    
    
    parser.add_argument('--ext_num', type=int, default=4)      

    parser.add_argument('--threshold', type=float, default=1)     

    parser.add_argument('--train_size', type=int, default=32)     
    
    parser.add_argument('--pre_train', type=int, default=0)    
    # Data Setting

    parser.add_argument('--train_path', type=str, default='data_file/DMCNN/train_file/*')

    parser.add_argument('--vaild_path', type=str, default='data_file/DMCNN/val_file/*')

    parser.add_argument('--test_path', type=str, default='data_file/DMCNN/test_file/*')

    parser.add_argument('--pre_model', type=str, default='')
    # Testing setting

    parser.add_argument('--block_trigram', type=int, default=1)     

    parser.add_argument('--recall_eval', type=int, default=0) 
    
    parser.add_argument('--test_model', type=str, default='1---1-126000-0.2923-0.3453.pth.tar') 
        
    parser.add_argument('--test_cal', type=str, default='') #lead oracle
    
    parser.add_argument('--no_pos', type=int, default=0)     
    
    # Checkpoint Setting

    parser.add_argument('--checkmin', type=int, default=60000)

    parser.add_argument('--checkfreq', type=int, default=6000)

    args = parser.parse_args()
    
    return args





def main():

    args = argLoader()

    torch.cuda.set_device(args.device)

    print('CUDA', torch.cuda.current_device())
    
    if args.do_train:

        train(args)

    elif args.do_test:

        test(args)



if __name__ == '__main__':

    main()