# Search engine
Image Retrieval 시스템에 붙일 search-engine 모듈입니다. 
페이지에서 검색 요청이 오면 Option에 맞는 Extractor 모듈로 이미지를 전송, feature를 받아옵니다. feature를 DB 와 비교해 선택한 갯수만큼의  유사이미지를 보내줍니다.

### 1. docker 

```bash
nvidia-docker run -it -p [host]:[local] -v [host]:[local] --name [name] floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.31 /bin/bash
```
마찬가지로 도커컨테이너를 생성해줍니다. 이때 포트번호만 적절하게주시면 됩니다.  유사이미지를 검색을 위해 pytorch를 사용할 것입니다.

### 2. 
도커 컨테이너가 설치가 끝난 후,  clone한 경로로 들어갑니다.
 
```bash
pip install -r requirements.txt && apt-get install rabbitmq-server -y  && sudo service rabbitmq-server restart
```
 필요한 모듈들을 설치해줍니다.

### 3. Extractor 모듈 등록
```bash
python manage.py createsuperuser
```
모듈 등록을 위해 관리자 계정을 만듭니다.
localhost의 8000번 포트로 도커를 만들었다는 가정하에
```
http://localhost.com:8000/admin
```
에 가셔서 생성한 관리자 계정으로 로그인합니다.

![enter image description here](https://i.imgur.com/dnpjSAj.png)

Extractor models 에 들어가서 Extractor 모델을 추가합니다.
이름과 url을 입력하고 간단한 설명을 입력 이름 입력시 대소문자에 주의할것,  name은 site 에서 날라온 option의 feature 와 동일해야합니다.
==>
#### options : {'dataset':'db1','feature':**'MAC'**,.....} ####
#### options['feature'] 와 등록하는 모델의 name 이 일치해야합니다.  ####



### 4. load feature
이 모듈이 실행 되면 가장 먼저 데이터셋에 대한 feature를 읽어옵니다. feature는 파일로 저장이 되어 있고, 추후 다른 방법으로 변경될 수도 있습니다. 
```
.
├── features
│   ├── dataset1
│   │   ├── MAC
│   │   ├── RMAC
│   │   └── SPoC
│   ├── dataset2
│   │   ├── MAC
│   │   ├── RMAC
│   │   └── SPoC
│   └── dataset3
│       ├── MAC
│       ├── RMAC
│       └── SPoC
└── images
    ├── dataset1
    ├── dataset2
    └── dataset3
```
```python
WebAnalyzer/tasks.py
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
```
global 변수에 모든 feature 들을 로드하였습니다.  모든 feature 를 로드하시면 됩니다.  
```python
db={'dataset1':{'MAC': ... ,'RMAC':...,'names':...}
,'dataset2':{'MAC': ... ,'RMAC':...,'names':...}
...
}
```
와 같이 전체 db 를 로드 해놓고 
```python
features = db['dataset1']['MAC'] 
```
이런식으로 사용하였습니다.  각 feauture 는 torch.tensor 이고 size는 [이미지 갯수, dim ] 입니다. gpu에 미리 로드 해놓습니다.  유사한 이미지를 다시 돌려줘야 되기 때문에 파일 이름도 같은 index로 저장해 놓습니다.  코드는 
```bash
WebAnalyzer/utils/load_features.py 
```
이고, feature를 저장한 방법이 다르기 때문에 이부분은 각자 알아서 수정해주면 됩니다.


### 5. Extract and Search
관련 코드는 WebAnalyzer/task.py 

site로 부터 받은 이미지와 옵션을 이용, 이미지를 해당 extractor로 보내주고 feature를 받습니다. 그 후, 유사이미지를 찾습니다. 

site로 받은 option-feature를 이용해 등록된 extractor model의 url을 가져옵니다. 
```python
WebAnalyzer/models.py
 def set_url(self,opts):
        extractors=extractorModel.objects.filter(name=opts['feature'])[0]
        self.url=extractors.url
```
이 url 로 이미지를 쏘고 feature를 받아옵니다. 
```python 
@app.task
def extract_and_search(img,options,url):
    print('celery task : ',url,options,img)
    query=send_request_to_extractor(img,options,url)
    result=similarity_search(query,options)
    return result
```
***send_request_to_extractor*** 에서 이미지를 쏘고 ***similarity_search***에서 유사이미지를 찾습니다. app.task는 celery관련 데코레이터로 냅두면됩니다.

```python 
WebAnalyzer/task.py
def similarity_search(query,options):
    imgPath=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','images')
    topN=int(options['topN'])
    dataset=options['dataset']
    feature=options['feature']
    
    f=db[dataset][feature]
    name=db[dataset]['names']

    #GET SIMILARITY
    query=torch.tensor(query.reshape([query.shape[0],-1]),dtype=torch.float32).cuda()
    score,idx=cosine_similiarity(query,f)

    score=score[0,-topN:].cpu().numpy()
    idx=idx[0,-topN:].cpu().numpy()
    
    ret=[{'name':i.decode(),'similarity':str(score[topN-n-1]),'image':str(base64.b64encode(open(os.path.join(imgPath,dataset,i.decode()),'rb').read()))} for n,i in enumerate(reversed(name[idx])))
    return ret 
```
extractor 에서 받은 query에 대한 feature와 사용자가 준 option 으로 유사이미지를 찾습니다. 전체 db 에서 해당되는 feature 만 가져와서 cosine 유사도를 계산합니다. return으로 
***[{'name':...,'similiarity':...,'img':...},{'name':...,'similiarity':...,'img':...} ...{'name':...,'similiarity':...,'img':...}]*** topN개의 {'name':...,'similiarity':...,'img':...}를 유사도 순서대로 반환하여 site에 줍니다. name 은 파일이름 , similiarity는 유사도,이고 추가로 site에 표현해야하는 정보가 있으면 알려주세요.

cosine 유사도를 구하는 코드는 **Webanalyzer/utils/metric.py** 에 있습니다. 
추가로 query expansion , rerank 등 다른 metric들도 올라갈 것입니다.

### 6.

여기까지 되셧으면 
```
http://[localhost]:[port]
```
로 들어가 보면
![enter image description here](https://i.imgur.com/oV51EBx.png)

이렇게 보일 것 입니다. 아래에 직접 옵션과 파일을 입력할수 있습니다.
```
{'dataset': 'photo', 'feature': 'MAC', 'threshold': '0.8'}
http://mleagles.sogang.ac.kr:8000
```
![enter image description here](https://i.imgur.com/vO6fY9u.png)
입력 후 post 를 누르면

![enter image description here](https://i.imgur.com/NRkcTFn.png)

이런식으로 결과가 보이면 완성입니다. ***먼저 해당 feature에 대한 extractor 모듈이 등록되어있는지 확인해 주세요.*** 또 http://localhost:[extractor모듈port] 로 들어가시면이 extractor가 정상적으로 동작하는 것을 확인할 수 있습니다. 


### 7. 
여기까지 완성되셨으면 알려주세요. 사용하는 데이터셋, feature, 다른 옵션들을 알려주시면 site에 반영하도록 하겠습니다.


**초기화**
```bash
rm db.sqlite3
sh server_initialize.sh
sh server_start.sh
```

**Log**
```bash
vim celery.log
vim django.log
```

celery 로그를 바로바로 볼수 있습니다. sh server_shutdown.sh 으로 celery 종료
```bash
sh run_celery.sh &
sh run_django.sh
```

도커 컨테이너를 종료하게 되면 rabbit-mq restart시 fail이 날 확률이 매우 높습니다.
그냥 포기하고 컨테이너를 재설치하는게 속편합니다.

### summary
```
1. 컨테이너 실행 (rabbit - mq 주의)
2. extractor 모듈 등록
3. load_features 작성

```
정도만 하면 될것 같습니다. 


### 수정
option 중 topN -> threshold : topN 개의 유사이미지검색 --> 유사도가 threshold 이상인 이미지 검색
