# Image_Colorization
🎨 HueRevive: 흑백 사진 컬러 복원 AI

## **[주제]**  
이미지의 흑백사진을 컬러사진으로 복원해주는 AI 구현  

## **[프로젝트를 시작한 이유]**  
-  AI 기술의 발달로 과거의 기록물들을 디지털로 복원하려는 수요가 증가하고 있음.  

- 흑백 사진은 정보 전달력과 감성 전달 측면에서 한계를 가지며, 이를 컬러화 함으로써 시각적 이해도와 몰입도와 생동감을 높일 수 있음.  

- 기존의 수작업으로 진행되던 컬러 복원 방식은 시간과 비용이 많이 들며 이에 딥러닝 기반의 컬러 복원 방식이 효율적임.  

- 자연스러운 컬러 복원을 위해 시각적 유사성을 평가하는 성능 평가 지표(SSIM, LPIPS), 사용자 주관적 평가도 함께 고려함.  

## **[목표 및 성과]**  
딥러닝 모델을 활용한 흑백 사진을 자연스럽고 실사에 가까운 컬러 이미지로 복원하는 AI 시스템 구현

- 기술적 성과    
  ▪️ 흑백 이미지를 자연스럽고 현실감 있게 컬러로 변환  
  ▪️ 성능 지표에서 높은 결과 달성

- 사용자 만족도 성과    
  ▪️ 역사적 사진, 가족 사진 등에서 모두 자연스러운 컬러화 가능  
  ▪️ 사용자 주관적 평가에서 ‘자연스럽다’는 평가 획득 가능

- 사회적 성과    
  ▪️ 컬러화하여 어린이. 청소년, 일반인이 역사적 이미지를 직관적으로 이해  
  ▪️ 교육 자료, 디지털 박물관 등에서 활용 가능  
  ▪️ 가정 내 가족 사진, 졸업 사진 등 추억을 컬러로 복원하는 서비스로 감성적 만족 제공

## 구성  
- 01_preoprocessing_and_loader : 데이터 로드 및 전처리  
- 02_model_and_training : 모델 학습 및 시각화, 평가  
- Flask : flask를 활용한 웹페이지  
# 연구 방법  
## 데이터셋  
본 프로젝트에서는 Kaggle에서 제공하는 **Landscape color and grayscale images** 데이터셋을 사용함.  
전체 데이터셋은 약 14,000장 이상의 이미지로 구성되어 있으나,  
초기 실험 단계에서는 학습 파이프라인 검증 및 성능 점검을 목적으로 일부 샘플만 추출하여 사용하여 실행함.

- Train: 1,500장  
- Validation: 100장  
- Test: 150장  

## **Flowchart**  
<img width="931" alt="image" src="https://github.com/user-attachments/assets/e9747624-d94c-4ca9-afb4-0e1f2b7a4ffd" />

## **모델 구조**
<img width="868" alt="image" src="https://github.com/user-attachments/assets/a7b17315-3bff-4774-ad96-c13ecf121c2e" />

## **성능 지표**   
- PSNR - Peak Signal-to-noise ratio   
▪️ 생성 혹은 압축된 영상의 화질에 대한 손실 정보를 평가하는 지표  
▪️ 30 이상이면 품질이 좋다고 판단  

- SSIM - Structural Similarity Index Measure  
▪️ 수치적인 에러가 아닌 인간의 시각적 화질 차이를 평가하는 지표  
▪️ 1에 가까울수록 품질이 좋음  

- LPIPS – Learned Perceptual Image Patch Similarity  
▪️ 2개의 이미지의 유사도를 평가하기 위해 사용되는 지표  
▪️ 0에 가까울수록 품질이 좋음  

-FID - Fréchet inception distance  
▪️ 실제 이미지와 생성된 이미지가 얼마나 유사한지 계산하는 지표  
▪️ 30 이하이면 품질이 좋다고 판단하고, 10 이하이면 실제 이미지와 거의 유사하다고 판단함  

## **모델 분석**
### **초기 복원 결과**
<img width="857" alt="image" src="https://github.com/user-attachments/assets/a3bca198-04cb-4c95-87a9-8358c6193426" />

### **초기 성능 지표**

| Metric | Score   |  
|--------|---------|  
| PSNR   | 23.04   |  
| SSIM   | 0.8210  |  
| LPIPS  | 0.1865  |  
| FID    | 106.43  |  

### **최종 복원 결과**  
<img width="886" alt="image" src="https://github.com/user-attachments/assets/3c64461c-4589-42de-9c55-72045d4e794a" />

### **초기 성능 지표**

| Metric | Score   |  
|--------|---------|  
| PSNR   | 29.25   |  
| SSIM   | 0.9334  |  
| LPIPS  | 0.0609  |  
| FID    | 33.57   |  

