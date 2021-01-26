# 딥페이크 변조 영상 탐지 AI 경진대회 - WeAreApe
-  데이콘에서 주관하는 딥페이크 변조 영상 탐지 AI 경진대회에서 입상한 WeAreApe 팀의 코드를 아래와 같이 공유드립니다.
(대회 링크: https://dacon.io/competitions/official/235655/overview/)

## 환경세팅
### version & device
- Ubuntu 16.04(OS)
- Python 3.7.8
- Cuda 10.1
- CuDNN 7.6.5
- TITAN RTX D6 24GB(GPU)
- install Package

      sudo pip3 install -r requirements.txt
      
     
## Inference Flow
![다운로드](https://user-images.githubusercontent.com/78020215/105839279-9926c780-6014-11eb-9e38-375e426a8291.png)


### Face detection with MTCNN
<img src="https://user-images.githubusercontent.com/78020215/105839585-13574c00-6015-11eb-9779-e6746e597107.png" width="80%" height="80%">

### Data Augmentation
> Kaggle – Deepfake Detection Challenge Private 1위 팀 내용 참고 (https://github.com/selimsef/dfdc_deepfake_challenge)
<img src="https://user-images.githubusercontent.com/78020215/105839801-59acab00-6015-11eb-923b-b792518302dc.png" width="80%" height="80%">
### Vector Embedding with CNN (Train)
      1. Network Model
            - Backbone : EfficientNetB4(380x380)
            - train - Load Imagenet pretrained model

            x = Dropout (0.4) (base_network.output)
            x = Dense (512, kernel_regularizer=l2_regularizer (regu_weight), activation=None) (x)
            outputs = tf.nn.l2_normalize (x, 1, 1e-10)
            
      2. Loss
             - Facenet* : semi-hard triplet loss 
             - triplet selection은 진행하지 않음
             - alpha : 1.0
             > FaceNet: A Unified Embedding for Face Recognition and Clustering/2015/Florian Schroff 외 2명/Google Inc
             
      3. Etc
            - Optimizer : SGD (lr=0.01) 사용
            - Batch_size : 4x12 
            - 1.8k train      

### Classifier(SVM)
<img src="https://user-images.githubusercontent.com/78020215/105840829-fde32180-6016-11eb-85fe-45c2f6d97e29.png" width="50%" height="50%">

## 사용법
### inference - 추론
  - inference_dacon_data.py 실행

        sudo python3 inference_dacon_data.py test/leaderboard resource/sample_submission.csv resource/triplet_effB4_ep06_BS28.hdf5 resource/triplet_effB4_ep06_BS28.pkl

  - 입력 parameter

        1. learderboard 경로 **(원본이미지) 
        2. sample_submission.csv 파일 경로
        3. network model (.hdf5) 파일 경로
        4. svm model (.pkl) 파일 경로 

  - 출력

         ( 2번의 입력 .csv 경로에 ) *_out.csv 제출 파일 작성


### train - 학습

#### crop face - 학습데이터 생성

  - crop_face_dacon.py 실행 ( 안면 이미지 생성)

        sudo python3 crop_face_dacon.py data/deepfake_1st data/cropface

  - 입력 parameter
  
        1. deepfake_1st 경로 **(원본이미지)
        2. 생성될 안면데이터 경로

  - 출력
  
        ( 2번의 입력 경로에 ) 안면데이터 생성
          Train 데이터 : fake, real
          Test 데이터 : validation


#### train - 학습 실행

  - train.py 실행

        sudo python3 train.py data/cropface resource resource

  - 입력 parameter
  
        1. 생성한 안면데이터 경로
        2. 저장될 network model (.hdf5) 파일 경로
        3. 저장될 svm model (.pkl) 파일 경로 

  - 출력
  
        ( 2번의 입력 경로에 ) network model (.hdf5) 파일 저장
        ( 3번의 입력 경로에 ) svm model (.pkl) 파일 저장


## 성능 개선의 기록 및 그 밖의 시도들 
### 성능 개선
       - Load Imagenet pretrained model : 학습 속도 및 학습이 진행되지 않는 경우의 개선 효과
       - Data augmentation (Gaussian noise & Image compression & Gaussian Blur) : 평균 70% >> 80% 의 개선 효과
       - InceptionResNetV2 & Xception(240x240) -> EfficientNetB4(380x380) : 평균 80~85% >> 85~90% 의 개선 효과
       - Triplet loss 사용 : 평균 85~90% >> 90~92% 의 개선 효과
       - Kernel_regularizer (L2)사용 : 평균 90~92% >> 94~97% 의 개선 효과
       
### 성능 개선 효과를 보지 못한 경우
     - pretained model freeze : FaceForensics++ 논문 참고하여 Model freeze 진행 (2~5step, lr(0.001))
     - Data augmentation (pattern alpha blending & blind face):
            - 1. 노이즈를 표현하기 위해 임의의 패턴을 blending하여 학습에 활용하였지만 오히려 성능 저하 발생
            - 2. best accuracy는 blind face를 이용하여 학습한 결과지만 평균적으로 큰 효과가 있지 않았음
![다운로드](https://user-images.githubusercontent.com/78020215/105841396-dc366a00-6017-11eb-82e5-a447e348eb34.png)

      - EfficientNetB5(380x380), EfficientNetB5(456x456) : batch size 16 or 12
      - triplet_loss alpha 값 변경 (0.2~1.5)
      - Lgbm

감사합니다.
