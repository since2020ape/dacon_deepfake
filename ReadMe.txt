


###########################################
version & device
###########################################
OS       : Ubuntu 16.04

Python  : Python 3.7.8

Cuda    : 10.1

CuDNN : 7.6.5

GPU     : TITAN RTX D6 24GB

Package: sudo pip3 install -r requirements.txt


###########################################
usage
###########################################

#######################
1. inference 
#######################

inference_dacon_data.py 실행

  sudo python3 inference_dacon_data.py test/leaderboard resource/sample_submission.csv resource/triplet_effB4_ep06_BS28.hdf5 resource/triplet_effB4_ep06_BS28.pkl

입력 parameter
--
1. learderboard 경로 **(원본이미지) 
2. sample_submission.csv 파일 경로
3. network model (.hdf5) 파일 경로
4. svm model (.pkl) 파일 경로 
--

출력
--
( 2번의 입력 .csv 경로에 ) *_out.csv 제출 파일 작성
--


#######################
2. train
#######################

crop_face_dacon.py 실행 ( 안면 이미지 생성)

  sudo python3 crop_face_dacon.py data/deepfake_1st data/cropface

입력 parameter
--
1. deepfake_1st 경로 **(원본이미지)
2. 생성될 안면데이터 경로
--

출력
--
( 2번의 입력 경로에 ) 안면데이터 생성
	Train 데이터 : fake, real
	Test 데이터 : validation
--


train.py 실행

  sudo python3 train.py data/cropface resource resource

입력 parameter
--
1. 생성한 안면데이터 경로
2. 저장될 network model (.hdf5) 파일 경로
3. 저장될 svm model (.pkl) 파일 경로 
--

출력
--
( 2번의 입력 경로에 ) network model (.hdf5) 파일 저장
( 3번의 입력 경로에 ) svm model (.pkl) 파일 저장
--




