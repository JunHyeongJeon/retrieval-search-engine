# Trademark

## Introduce
--

## Requirement

- python 3 or later
- pytorch 0.4.0
- cuda
- rabbitmq-server
- django

편의를 위해 docker image floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.31를 사용

## Installation

### From Source

실행에 필요한 service를 설치한다.
```bash
sudo apt-get install rabbitmq-server
sudo service rabbitmq-server restart
```
실행에 필요한 package를 설치한다.
```bash
pip install -r requirements.txt
```
만약 package 설치가 진행되지 않는다면 pip를 업데이트 한 후 다시 시도한다.
```bash
pip install --upgrade pip
pip install setuptools
```

## Setting Module

모든 설치가 끝났다면 Modules을 추가하기 위해 Modules 디렉토리로 이동한다.
여기에는 작성에 도움을 주기 위해 dummy 디렉토리 내 main.py를 참고하여 작성한다.

### Extractor 모듈 등록
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

### Configure Module Class

* Module 내 다른 python import 하기
    ```python
    from Modules.dummy.example import test
    ```
    * Django 실행 시 root 폴더가 프로젝트의 최상위 폴더가 되므로, sub 폴더 내 다른 python 파일을 import 위해서는 위와 같이 최상위 폴더 기준으로 import를 해야한다.

* \__init\__ 
    ```python
    model_path = os.path.join(self.path, "model.txt")
    self.model = open(model_path, "r")
    ```
   * \__init\__에서는 model 불러오기 및 대기 상태 유지를 위한 코드를 작성한다. 
   * model 등의 파일을 불러오기 위해선 model_path를 사용하여 절대경로로 불러오도록 한다. 

* inference_by_path
    ```python
    result = [[(0, 0, 0, 0), {'TEST': 0.95, 'DEBUG': 0.05}], [(100, 100, 100, 100), {'TEST': 0.95, 'DEBUG': 0.05}]]
    ```
    * 이미지 경로를 받고 \__init\__ 에서 불러온 모델을 통해 분석 결과를 반환하여 저장한다. 이때 결과값은 다음과 같은 형태를 가지도록 구성한다.
        ```text
          [ [ ( x, y, w, h ), { Label_1 : Percent_1, Label_2 : Percent_2 } ], [ ( x, y, w, h ), { Label : Percent } ] ]
        ```
    * 이는 결과로서 두 개의 객채를 검출했으며, 첫 객체는 (x, y)를 시작점으로 하고 너비 w, 높이 h를 가지는 사각 영역에서 나타났으며, 그때 그 객체는 순서대로 Label_1 및 Label_2로 예상됨을 나타낸다.

### Modify Tasks

위와 같이 Moduele 설정이 끝났다면 작성한 Module을 추가하기 위해 WebAnalyzer 디렉토리로 이동한다. 그 후 tasks.py를 수정한다.

* Module 불러오기
    ```python
    @worker_process_init.connect
    def module_load_init(**__):
        global analyzer
        worker_index = current_process().index
    
        print("====================")
        print(" Worker Id: {0}".format(worker_index))
        print("====================")
    
        # TODO:
        #   - Add your model
        #   - You can use worker_index if you need to get and set gpu_id
        #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
        from Modules.dummy.main import Dummy
        analyzer = Dummy()
    ```
    * 위에서 작성한 class을 불러온 후, anlyzer에 추가한다.
    * 만약 이 때, Multi-gpu를 사용하여 gpu 별로 나누어 추가하고 싶다면, worker_index를 사용하여 이를 수정할 수 있다.

* Module 실행하기
    ```python
    @app.task
    def analyzer_by_path(image_path):
        result = analyzer.inference_by_path(image_path)
        return result
    ```
    * 위에서 불러온 Module이 실제로 실행되는 부분으로, 분석 결과를 받아 반환한다.


### Additional Settings

실행 시에 필요한 다양한 Setting을 변경하고 싶다면 AnalysisModule 디렉토리의 config.py를 수정한다.

* 불러오는 Module 수 조절하기
```python
TOTAL_NUMBER_OF_MODULES = 2
```

## Setting Database

### Migration
Django 내 필요한 model 구조를 반영하기 위해 다음을 실행한다.
```bash
sh run_migration.sh
```
만약 필요에 의해 model 구조를 변경하였다면, run_migration.sh을 통해 생성된 파일을 지우고 다시 설정해주어야 한다.
```bash
sudo rm db.sqlite3
sh server_initialize.sh
sh run_migration.sh
```

## Run Web Server

* Web Server를 실행하고자 한다면 server_start.sh를 실행한다.
    ```bash
    sh server_start.sh
    ```
    이후 http://localhost:8000/ 또는 구성한 서버의 IP 및 Domain으로 접근하여 접속한다.

* 만약 접속 시 문제가 있어 실행 Log를 보고자 할 때는 다음과 같이 실행하여 확인한다.
    * Web Server에 문제가 있어 Django 부분만 실행하고자 한다면 run_django.sh를 실행한다.
        ```bash
        sh run_django.sh
        ```
    
    * Web Server는 실행되나 분석 결과가 나오지 않아 Module 부분만 실행하고자 한다면 run_celery.sh를 실행한다.
        ```bash
        sh run_celery.sh
        ```
    
* Web Server를 종료하고자 한다면 server_shutdown.sh를 실행한다.
    ```bash
    sh server_shutdown.sh
    ``` 
