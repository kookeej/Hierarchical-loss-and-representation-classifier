# Hierarchical loss, representation, and label embedding with plm classifier
#### *계층적 loss 및 representation과 레이블 임베딩을 이용한 논문 문장 수사학적 분류 모델(국내 논문 문장 의미 태깅 모델 개발)*
### 🏆 KISTI 2022 과학기술 공공 AI 데이터 분석활용 경진대회 국회도서관 장관상(장려상, 상금 100만원) 수상
`국희진`: 모델링, 전체 코드 작성    
`김영화`: 모델링    
`윤세휘`: 데이터 전처리, 웹 페이지    
`강병하`: 웹 페이지
***
![image](https://user-images.githubusercontent.com/74829786/205135412-19c68cd9-c875-44d2-9342-f15309c99122.png)

***
# 1. Model
## 1.1. 전체 모델 구조
![image](https://user-images.githubusercontent.com/74829786/205135649-a260ec96-5af1-4693-b035-2e89e0c35985.png)

## 1.2. 사전 학습 모델을 활용한 레이블 임베딩 테이블
![image](https://user-images.githubusercontent.com/74829786/205135787-ae5efe44-b467-4504-a194-bc1702f4e38a.png)
* 사전 학습 언어모델(KorSciBERT)을 사용하여 레이블 임베딩 테이블 초기화
* 학습과 함께 레이블 임베딩 테이블과 BiLSTM 레이어 파라미터 업데이트

## 1.3. 계층적 손실함수
![image](https://user-images.githubusercontent.com/74829786/205136401-83b8abfb-ebf1-4ef2-8391-12a4e82f2f74.png)
* *Deep Hierarchical Classification for Category Prediction in E-commerce System(ACL 2020 ECNLP3)* 참고
* 계층적 손실 함수를 통해 대분류 및 소분류 카테고리 예측 성능을 향상
* 상위 카테고리와 하위 카테고리 사이의 구조 학습 가능
  * ***Layer loss(Lloss)***
    * 상위/하위 카테고리 예측값에 대한 손실값
  * ***Dependence loss(Dloss)***
    * 예측한 상위, 하위 카테고리가 서로 포함 관계가 아닌 경우 패널티 부여
    
***
# 2. Performance
## 2.1. 섹션명(위치 정보) 사용하지 않았을 경우
![image](https://user-images.githubusercontent.com/74829786/205137099-03c05972-3a6f-4738-b46d-be96a743f700.png)

## 2.2. 섹션명(위치 정보) 사용하였을 경우
![image](https://user-images.githubusercontent.com/74829786/205137268-d09a25a1-5d14-4a12-b8de-e0d5fe699030.png)

***
### 3. References
* *https://github.com/Ugenteraan/Deep_Hierarchical_Classification*
* *https://aida.kisti.re.kr/data/8d0fd6f4-4bf9-47ae-bd71-7d41f01ad9a6/gallery/17*
