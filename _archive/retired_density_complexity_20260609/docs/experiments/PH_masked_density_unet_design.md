# PH Masked Density U-Net Design

PH mask를 FCN에 적용할 때는 "이미지를 irregular하게 만드는" 게 아니라, rectangular crop 위에 `valid pixel mask`를 얹는 방식이 맞다. 아래 graph-style 비교는 historical analogy이며, active implementation은 U-Net only이다.

## Historical Graph Analogy

Retired graph-style 방식:

```text
PH contour 내부 pixel -> graph node
node별 target -> y / y_edge_decay
node별 prediction
scene에 node 위치로 다시 scatter
```

FCN/U-Net 방식:

```text
PH contour bbox -> rectangular crop
PH contour 내부 pixel -> roi_mask = 1
PH contour 외부 pixel -> roi_mask = 0
모델은 crop 전체를 보되, loss/output 반영은 roi_mask 내부만
```

## 구체적 흐름

```text
1. PH로 contour mask 추출
2. contour의 bounding box + padding으로 rectangular crop 생성
3. 같은 크기의 roi_mask 생성
4. GeoJSON GT를 crop 좌표로 변환
5. 선박 point마다 sum-preserving kernel 생성
6. kernel을 roi_mask 내부로 제한
7. kernel 합이 1이 되도록 재정규화
8. U-Net이 crop 전체에 density map 예측
9. loss는 roi_mask 내부 pixel만 계산
10. inference 결과도 roi_mask 내부만 scene heatmap에 합성
```

## 핵심 구현

```python
pred = model(image_crop)

loss_map = loss_fn(pred, target_density)
loss = (loss_map * roi_mask).sum() / roi_mask.sum()
```

Scene assembly도 같은 원리로 처리한다.

```python
scene_pred[roi_pixels] += pred[roi_pixels] * lifetime_weight
scene_weight[roi_pixels] += lifetime_weight
```

즉 graph-style 방식에서 `node가 아닌 픽셀은 아예 학습/예측 대상이 아니었던 것`처럼, FCN에서는 `mask 밖 픽셀은 loss와 최종 heatmap 반영에서 제외`한다.

## GT kernel도 mask 안에서만

Retired graph-style `edge-decay`는 graph 내부 node로만 퍼졌다. FCN에서도 동일하게 하려면:

```text
ship kernel 생성
-> roi_mask 밖 값 제거
-> 남은 kernel 합이 1이 되도록 재정규화
```

그래서 한 척은 여전히 총합 1이다.

```text
kernel_sum_before_mask = 1.0
kernel_masked = kernel * roi_mask
kernel_masked = kernel_masked / kernel_masked.sum()
```

## 입력 channel

추천은 input channel을 2개로 하는 것이다.

```text
channel 1: brightness crop
channel 2: PH roi_mask
```

그러면 모델이 "이 crop에서 어디가 유효한 PH island인지" 안다.

단, brightness 자체는 mask 밖을 무조건 0으로 지우지 않는 게 낫다. 처음에는:

```text
input  = [원본 brightness crop, roi_mask]
loss   = roi_mask 내부만
output = roi_mask 내부만 사용
```

이게 가장 안정적이다. mask 밖 밝기는 context로 볼 수 있지만, 정답/오차/최종 출력에는 안 들어간다.

## 모서리에 다른 군집이 끼는 문제

다른 군집이 `roi_mask 밖`이면:

```text
보이긴 하지만 loss에는 안 들어감
최종 heatmap에도 안 들어감
```

다른 군집이 `roi_mask 안`이면:

```text
PH가 같은 위상학적 island로 본 것이므로 같은 sample에서 같이 처리
```

그래서 retired graph-style 방식과 같은 철학을 rectangular U-Net crop에서 구현하는 형태다.

## 결론

`PH contour mask`는 FCN/U-Net에서 `ROI mask + masked loss + masked scene assembly`로 적용하면 된다.
