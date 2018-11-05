from __future__ import print_function

from AnalysisModule.celerys import app
from celery.signals import worker_init, worker_process_init
from billiard import current_process

import requests
import json
import torch
import numpy as np
import h5py
import ast
import os
import base64

import datetime

from WebAnalyzer.utils.load_features import load_features
from WebAnalyzer.utils.metric import cosine_similiarity

@worker_init.connect
def model_load_info(**__):
    print("====================")
    print("Worker Analyzer Initialize")
    print("====================")


@worker_process_init.connect
def module_load_init(**__):
    #global analyzer
    global db
    worker_index = current_process().index

    print("====================")
    print(" Worker Id: {0}".format(worker_index))
    print("====================")

    # TODO:
    #    - load DB
    load=datetime.datetime.now()
    db=load_features()
    print("load-time: {}".format(datetime.datetime.now()-load))



@app.task
def extract_and_search(img,options,url):
    print('celery task : ',url,options,img)
    query=send_request_to_extractor(img,options,url)
    result=similarity_search(query,options)
    return result


def send_request_to_extractor(img,options,url):
    json_image=open(img,'rb')
    json_files={'image':json_image}
    json_data=dict()

    response=requests.post(url=url,data=json_data,files=json_files)
    json_image.close()
    response_data=json.loads(response.text)['result'][0]['feature']

    '''
    for l in response_data:
        if l['descriptor']==options['feature']:
            response_data=np.array(ast.literal_eval(l['feature']))
    '''
    print("aaaa",response_data)

    return np.array(ast.literal_eval(response_data))

# images 경로 
def similarity_search(query,options):
    imgPath=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','images')
    threshold=float(options['threshold'])
    topK=options['topK']
    dataset=options['dataset']
    feature=options['feature']


    if(dataset=='all'):      
        tmp=[]        
        name=[]
        for i,dt in enumerate(db.keys()):
            tmp.append(db[dt][feature])
            name.append([os.path.join(dt,n) for n in db[dt]['names']])
        f=torch.cat(tmp,0)
        print(len(name))
    else:
        f=db[dataset][feature]
        name=db[dataset]['names']
        print(len(name))

    print("select features :"+feature,dataset)
    print("dataset size : ",f.shape)
    
    


    calculate=datetime.datetime.now()    
    #GET SIMILARITY
    query=torch.tensor(query.reshape([query.shape[0],-1]),dtype=torch.float32).cuda()

    score,idx=cosine_similiarity(query,f)

    sc=score.cpu().numpy()[0,:]

    if topK == '' or topK== '0':
        print('22222')
        thres=np.where(sc>=threshold)[0]
        if(len(thres)):
            thres=thres[0]
        else:
            thres=len(name)
    else:
        print('1111')
        thres=-int(topK)
        print(thres)

    score=sc[thres:]
    idx=idx[0,thres:].cpu().numpy()

    print(score)
    print(idx)
    print(name[idx])

    #score=score[0,-topN:].cpu().numpy()
    #idx=idx[0,-topN:].cpu().numpy()
    if(len(idx)>100):
        #name=name[:100]
        idx=idx[len(idx)-100:]

    print("search-time: {}".format(datetime.datetime.now()-calculate))


   # ret=[{'name':i.decode(),'similarity':str(score[topN-n-1]),'image':str(base64.b64encode(open(os.path.join(imgPath,dataset,i.decode()),'rb').read()))} for n,i in enumerate(reversed(name[idx]))]
  
    ret=[{'name':i.decode(),'similarity':str(round(score[score.shape[0]-n-1],3)),'image':str(base64.b64encode(open(os.path.join(imgPath,dataset,i.decode()),'rb').read()))} for n,i in enumerate(reversed(name[idx]))]

    return ret
