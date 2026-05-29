좋은 방향입니다. 내가 보기엔 **2 계층적 PH를 region proposal의 rigid structure로 쓰고, 3의 “GT를 놓치지 않는 crop-level target”을 결합**하는 게 제일 타당합니다.

핵심은 이겁니다.

```text
PH hierarchy = region/crop 구조를 rigid하게 결정
GT target = PH mask 내부로 제한하지 않고 crop 내부 전체에서 생성
loss/attention = PH hierarchy를 soft 또는 multi-channel prior로 사용
```

즉 PH는 “정답 영역”이 아니라 **좌표계/구조/관심도 prior**로 격상시키는 겁니다.

**제안 구조**
각 큰 PH parent region을 하나의 training sample로 봅니다.

```text
parent PH region crop
├─ child PH masks
├─ peak/seed map
├─ persistence score map
├─ distance-to-mask map
└─ crop 내부 전체 GT density target
```

모델 입력:

```text
channel 0: DNB brightness crop
channel 1: parent PH mask
channel 2: child PH union mask
channel 3: PH seed/peak map
channel 4: persistence score map
channel 5: distance/soft attention map
```

GT target:

```text
crop 내부 모든 GT point 사용
Gaussian kernel로 density target 생성
sum-preserving
PH mask 내부 여부로 GT를 버리지 않음
```

loss:

```text
base loss: crop 전체 또는 valid crop 전체
weighted loss: PH hierarchy soft mask로 내부를 더 강하게
```

예:

```text
loss_weight = 0.25 + 0.75 * soft_ph_attention
```

이러면 PH 밖 GT도 학습에 남고, PH 안쪽은 더 강하게 학습됩니다.

**왜 이게 좋은가**
이 결합은 네 요구 두 개를 동시에 만족합니다.

- 2의 장점: PH hierarchy가 rigid하게 crop과 구조를 잡아줌
- 3의 장점: PH가 놓친 GT도 crop 안에 있으면 target에서 사라지지 않음
- 밀집 군집: parent region이 넓게 context를 제공
- 고립 선박: child/seed map이 local cue 제공
- 모델 입장: “여기가 PH상 중요한 곳”은 알지만, 거기에만 답을 강제당하지 않음

**중요한 구현 규칙**
나는 다음을 default로 잡는 게 맞다고 봅니다.

```text
require_source_in_roi = False
renormalize_after_roi_mask = False
```

대신 crop boundary에서는 kernel을 잘라야 하므로 crop 내부에서는 renormalize합니다.

정확히는:

```text
GT point가 crop 안에 있음
-> Gaussian kernel 생성
-> crop 밖으로 나간 부분만 제거
-> 남은 kernel을 다시 sum=1로 정규화
-> target에 더함
```

PH mask로 kernel을 자르지는 않습니다. PH는 target censoring에 쓰면 안 됩니다.

**loss mask 정책**
hard mask 하나로 끝내면 다시 PH 누락 문제가 생깁니다. 그래서 loss는 soft weight가 낫습니다.

예:

```text
loss_weight = valid_crop_mask * (base_weight + ph_weight * ph_soft_attention)
```

추천 초기값:

```text
base_weight = 0.25
ph_weight = 0.75
```

의미:

- crop 전체는 최소 0.25로 학습
- PH hierarchy 내부/근처는 최대 1.0으로 학습
- PH 밖 GT도 loss에 반영됨
- PH가 모델의 attention prior 역할을 함

**계층적 PH crop 방식**
parent/child 관계는 이렇게 잡으면 됩니다.

```text
large parent contour:
  충분히 큰 persistent component

child contours:
  parent 내부에 포함되는 작은 persistent components

seed map:
  child birth 좌표 또는 local maxima

persistence score map:
  각 child mask pixel에 normalized lifetime 할당
```

parent crop 하나가 training sample이 됩니다.

고립 선박의 경우:

```text
parent == single blob
child 거의 없음
seed 하나
crop 작음
```

밀집 군집의 경우:

```text
parent 큼
child 여러 개
seed 여러 개
crop 큼
```

이게 계층적으로 자연스럽습니다.

**모델 쪽 의미**
이 구조에서는 U-Net이 main인 이유가 더 명확해집니다.

- brightness만 보면 모호함
- PH hierarchy channel을 보면 구조 prior가 있음
- U-Net은 crop 내부 spatial density map 복원
- fast dilated CNN은 같은 입력을 더 싸게 처리하는 baseline

GAT도 살릴 수 있습니다.

```text
PH child/seed = graph node
parent crop = image context
CNN feature at node location = node feature
GAT = node 간 관계 보정
```

하지만 이건 2단계입니다. 지금은 FCN/U-Net scaffold에 PH hierarchy channels를 넣는 게 우선입니다.

**내 추천 next implementation**
다음 작업은 `DensityPatch`를 확장하는 겁니다.

현재:

```text
image
roi_mask
target_density
raw_count
```

확장:

```text
image
parent_mask
child_union_mask
seed_map
persistence_map
soft_attention
target_density
loss_weight
raw_count
```

그리고 coverage diagnostic에서 비교:

```text
strict_target: PH 내부 GT만 target
hybrid_target: crop 내부 GT 전체 target
```

우리가 원하는 건 hybrid입니다.

**결론**
2와 3은 합치는 게 맞습니다. 다만 합치는 방식은:

```text
PH hierarchy는 rigid하게 crop/structure/input channels로 반영
GT target은 PH mask로 censoring하지 않음
loss는 PH hierarchy로 soft weighting
```

이게 가장 안전합니다.  
PH의 장점은 살리고, PH recall 실패가 학습 label 누락으로 이어지는 문제를 막습니다.
