좋은 방향입니다. 내가 보기엔 **계층적 PH를 region proposal의 rigid structure로 쓰고, hard pixel target을 PH mask로 검열하지 않는 방식**이 제일 타당합니다.

2026-06-09 update: active U-Net input은 3채널로 고정합니다.

```text
brightness
ph_persistence_map
ph_seed_map
```

`parent_ph_mask`, `child_ph_union_mask`, `ph_soft_attention`, `anchor_lifetime_map`은 모델 입력에서 제거했습니다. PH parent/child mask는 proposal/target 생성 중에만 일시적으로 계산하고 patch/batch 메모리에 저장하지 않습니다. 넓은 binary/attention mask를 U-Net에 직접 주지 않습니다.

핵심은 이겁니다.

```text
PH hierarchy = region/crop 구조를 rigid하게 결정
GT target = PH mask 내부로 제한하지 않고 crop 내부 전체에서 생성
model input = brightness + persistence + seed만 사용
```

즉 PH는 “정답 영역”이 아니라 **좌표계/구조/후보 패치 prior**로 격상시키는 겁니다.

**제안 구조**
각 큰 PH parent region을 하나의 training sample로 봅니다.

```text
parent PH region crop
├─ peak/seed map
├─ persistence score map
└─ crop 내부 hard pixel target
```

모델 입력:

```text
channel 0: DNB brightness crop
channel 1: persistence score map
channel 2: PH seed/peak map
```

GT target:

```text
crop 내부 모든 GT count pixel 사용
Y_pixel = 1[raw_count > 0]
PH mask 내부 여부로 GT를 버리지 않음
```

loss는 valid owner mask만 사용합니다. PH attention으로 loss를 재가중하지 않습니다.

**왜 이게 좋은가**
이 결합은 네 요구 두 개를 동시에 만족합니다.

- PH hierarchy가 rigid하게 crop과 구조를 잡아줌
- PH가 놓친 GT도 crop 안에 있으면 target에서 사라지지 않음
- 밀집 군집: parent region이 넓게 context를 제공
- 고립 선박: child/seed map이 local cue 제공
- 모델 입장: “여기가 PH상 중요한 곳”은 알지만, 거기에만 답을 강제당하지 않음

**중요한 구현 규칙**
나는 다음을 default로 잡는 게 맞다고 봅니다.

```text
require_source_in_roi = False
renormalize_after_roi_mask = False
```

Gaussian-smoothed density target은 legacy visualization에만 남깁니다. PH는 target censoring에 쓰면 안 됩니다.

**loss mask 정책**
hard pixel target은 valid owner mask 안에서만 학습합니다. PH는 patch를 만들지만 loss를 재가중하지 않습니다.

```text
supervision_mask = valid_owner_mask
```

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

GAT-style graph refinement was considered, but it is retired from the active path.

```text
PH child/seed = graph node
parent crop = image context
CNN feature at node location = node feature
graph refinement = node 간 관계 보정
```

The current implementation direction is U-Net only: PH hierarchy channels provide the structure prior directly to the image-to-density model.

**내 추천 next implementation**
다음 작업은 `DensityPatch`를 확장하는 겁니다.

현재:

```text
image
target_density
raw_count
```

확장:

```text
image
persistence_map
seed_map
target_density
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
