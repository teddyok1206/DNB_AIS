# codex_stamp

최신 entry를 맨 위에 추가하는 누적 로그 형식으로 유지한다. 기존 entry는 지우지 않는다.

## Entry 35

1. 업데이트 날짜, 시각
- 2026-04-09 17:49 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- 파이프라인 본체는 그대로 두고 모델 구조 시각화 실행 경로를 확장했다. `torchview`를 설치해 체크포인트 기반 모듈 단위 그래프 PNG/DOT를 생성했고, `netron` CLI도 같은 체크포인트로 로컬 서버 기동(127.0.0.1) 및 HTTP 응답을 확인했다. 이로써 외부 웹서비스 없이 로컬에서 연산 그래프(torchviz), 모듈 그래프(torchview), 인터랙티브 뷰어(netron)를 모두 사용할 수 있다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `sub_module/torchview_export.py`를 추가했고 `torchview` 출력 파일을 생성했다. `netron`은 포트 8092에서 실제 리슨/응답 확인 후 종료했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- `torchview`는 PyG `Data` 직접 입력에서 오류가 나서 `(x, edge_index)` 래퍼 모델 방식으로 우회했다. 현재는 정상 동작한다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- 필요 시 representative 실제 graph 입력 기반 torchview 옵션 추가

## Entry 34

1. 업데이트 날짜, 시각
- 2026-04-09 17:45 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- 파이프라인 본체 변경 없이 모델 구조 시각화 툴링 점검을 수행했다. `DNB_AIS` 환경에 `netron`과 `graphviz(dot)`를 설치하고, 기존 `.pt` 체크포인트를 대상으로 CLI 서버 실행 가능 여부를 검증했다. 로컬 호스트 포트 리슨 상태로 정상 기동되는 것을 확인했으며, 외부 웹 서비스 없이 로컬에서 모델 구조 확인이 가능한 상태다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `DNB_AIS`에 `netron==9.0.0` 설치, `graphviz`(dot) 설치 완료. `netron --version`, `--help`, 실제 checkpoint 대상 서버 기동(127.0.0.1:8091) 검증을 마쳤다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 초기에는 `torchviz`의 `distutils` 문제와 dot 미설치가 있었으나 모두 해결했다. 현재 미해결 이슈는 없다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- 필요 시 module-level 도식용 패키지(torchview) 추가 검토

## Entry 33

1. 업데이트 날짜, 시각
- 2026-04-09 17:42 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- 파이프라인 본체는 유지했다. 모델 시각화 유틸리티를 임시 스크립트에서 정식 sub_module 도구로 승격했다. 새 스크립트는 checkpoint 자동 탐색/명시 입력, 출력 경로 지정, 더미 그래프 노드 수 조절 옵션을 제공하며 torchviz PNG/DOT를 생성한다. 반복 분석 시 `python -m sub_module.torchviz_export`로 재사용 가능하다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `sub_module/tmp_torchviz_export.py`를 제거하고 `sub_module/torchviz_export.py`를 추가했다. 옵션 기반 CLI로 일반화했고 실행 검증도 완료했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- torchviz 그래프는 실제 데이터 forward가 아니라 더미 graph 입력 기반 autograd 연산도식이다. 구조 해석 용도에는 충분하지만 실제 batch shape를 그대로 반영하진 않는다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- 필요 시 실제 representative graph 입력 기반 시각화 옵션 추가

## Entry 32

1. 업데이트 날짜, 시각
- 2026-04-09 17:39 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- 파이프라인 본체는 유지했다. 이번 작업은 학습된 `.pt` 체크포인트의 구조 시각화 지원으로, `torchviz` 기반 임시 exporter를 추가해 기존 체크포인트를 로드한 뒤 더미 graph 입력으로 autograd 연산 그래프를 렌더링했다. 결과는 PNG와 DOT 파일로 저장되어 Netron/Graphviz 도구에서 직접 확인 가능하다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `sub_module/tmp_torchviz_export.py`를 추가했고, `outputs/DNB_GAT_v1/model_viz/`에 checkpoint별 `*_torchviz.png`, `*.dot` 2세트를 생성했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 초기에는 `torchviz`의 `distutils` import 오류와 `dot` 바이너리 부재가 있었지만 환경 패키지 업데이트 후 해결했다. 현재 미해결 이슈는 없다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- 필요 시 실제 graph 입력 기반 torchviz/ONNX 시각화 추가

## Entry 31

1. 업데이트 날짜, 시각
- 2026-04-09 17:30 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- 기본 파이프라인은 유지한다. 이번 변경은 모델 깊이 기본값 조정으로, GAT 계층 수를 물리 해석 기준(현재 GT edge-decay가 2-hop)과 맞추기 위해 기본 `num_layers`를 2로 낮췄다. 적용 범위는 `TrainingConfig` 기본값, `GATDensityRegressor` 기본 인자, notebook 생성기의 scene 기본 training 설정(batch_demo/kr_full_scene)이다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `num_layers` 기본값을 3→2로 변경했고 노트북을 재생성했다. 문법 검증(`py_compile`)도 통과했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 과거 실행 결과와의 직접 비교 시 layer 변경 영향이 섞일 수 있다. 동일 조건 재실행이 필요하다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- num_layers=2 기준 재학습/재평가 후 3층과 성능 비교

## Entry 30

1. 업데이트 날짜, 시각
- 2026-04-09 17:22 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- 기본 파이프라인(GeoTIFF→GT→DRUID patch→graph→GAT→scene merge)은 유지한다. 이번 변경으로 학습 상태 저장/재개 경로를 강화했다. `model state + optimizer state + completed_epochs + history_records + RNG state`를 담는 training checkpoint 저장/로드 함수를 추가했고, 모델 파라미터 초기화(reset) 유틸리티도 넣었다. notebook 생성기에도 resume/fresh-reset 제어 변수를 추가해 현장에서 이어학습과 초기학습을 모두 선택 가능하게 했다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `save_training_checkpoint`, `load_training_checkpoint`, `reset_model_parameters`를 추가했다. `train_gat`은 optimizer 주입과 epoch_offset을 지원한다. notebook은 `resume_checkpoint_path`, `reset_training`으로 재개/초기화를 제어한다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- PyTorch 2.6+ 기본 `torch.load(weights_only=True)` 때문에 training checkpoint 로드가 실패할 수 있어 load 함수에서 `weights_only=False`로 고정했다. 신뢰 가능한 로컬 체크포인트만 로드해야 한다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- 실제 scene에서 training checkpoint 저장→resume→추가 epoch 검증

## Entry 29

1. 업데이트 날짜, 시각
- 2026-04-09 17:05 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- 파이프라인 구조 자체는 동일하다(GeoTIFF 입력, GT 매핑, DRUID patch, graph 생성, GAT 추론, scene 병합). 이번 작업은 산출물/캐시 정리 단계로, batch_demo 내 중복 보관된 run zip 산출물과 코드 캐시(`__pycache__`)를 제거했다. 실행 결과 재현에 필요한 핵심 폴더 run 디렉터리와 주요 CSV/GeoTIFF/PNG는 유지했다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `run_0409_010223.zip`, `run_0409_012011.zip`를 삭제했고, `[3]/DRUID` 및 `sub_module`의 `__pycache__` 디렉터리를 전부 제거했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 정리 작업에서 오류는 없었다. 다만 zip 삭제로 인해 해당 시점 결과의 압축 백업본은 사라졌고, 필요 시 run 폴더를 다시 압축해야 한다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- 기본 파이프라인 실험 지속, 필요 시 run 결과 재압축 정책 수립

## Entry 28

1. 업데이트 날짜, 시각
- 2026-04-09 17:01 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 입력, GT 매핑, DRUID patch 분할, graph 생성, GATv2Conv 회귀, scene 병합이라는 기본 파이프라인은 유지된다. 이번 작업은 코드 변경이 아니라 산출물 정리 작업으로, 3개 대체 손실 실험에서 생성된 임시 비교 출력 디렉터리(`tmp_loss_compare_*`)를 batch_demo 경로에서 제거해 작업 트리를 기본 실험 결과 중심으로 정리했다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `outputs/.../batch_demo/tmp_loss_compare_0409_162712`, `tmp_loss_compare_0409_162527` 디렉터리를 삭제했다. 파이프라인 코드와 노트북은 변경하지 않았다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 삭제 작업 자체 이슈는 없다. 다만 임시 실험 결과를 다시 비교하려면 재실행이 필요하다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- 기본 PoissonNLL 경로 기반 실험/튜닝 계속 진행

## Entry 27

1. 업데이트 날짜, 시각
- 2026-04-09 16:58 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. graph supervision은 `sum-preserving y_edge_decay`를 유지하고 기본 학습은 `PoissonNLL + Softplus`다. 이번 변경은 실험 정리 목적의 코드 클린업으로, 3개 대체 손실(ZIP/NB/Poisson+sum) 비교용 임시 스크립트를 제거해 메인 실행 경로를 기본 파이프라인만 남겼다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `sub_module/tmp_loss_compare_physical.py`를 삭제했다. 메인 파이프라인 코드(`dnb_gat_pipeline.py`, notebook 생성기, 노트북)에는 기능 변경이 없다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 코드 삭제만 수행했으므로 미해결 이슈는 새로 추가되지 않았다. 이전에 생성된 결과 산출물(CSV/GeoTIFF/PNG)은 outputs 경로에 남아 있다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- 기본 PoissonNLL 경로 기준 후속 튜닝/검증 진행

## Entry 26

1. 업데이트 날짜, 시각
- 2026-04-09 16:31 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. graph supervision은 `sum-preserving y_edge_decay`를 유지한다. 이번에는 메인 코드 수정 없이 임시 비교 스크립트에서 `ZIP-Poisson`, `Negative Binomial`, `PoissonNLL+sum constraint` 3개 손실을 동일 graph set(batch_demo)으로 MPS 학습·추론해 scene heatmap 지표를 비교했다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `sub_module/tmp_loss_compare_physical.py`를 추가해 3개 손실을 독립 실행했다. 결과는 `tmp_loss_compare_0409_162712` 하위 CSV/JSON/heatmap으로 저장했고, 메인 파이프라인 파일은 수정하지 않았다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 현재 3개 손실 모두 scene top-k hit가 낮고 heatmap peak가 0.02 미만이다. 이번 비교는 고정 하이퍼파라미터 1세트 기준이라 성능 결론 확정 전 추가 튜닝이 필요하다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- ZIP/NB 하이퍼파라미터와 graph radius·layers를 같이 튜닝해 재비교

## Entry 25

1. 업데이트 날짜, 시각
- 2026-04-09 16:01 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. 메인 supervision은 `sum-preserving y_edge_decay`를 유지한다. 이번엔 모델 단순화를 위해 `count_weight_alpha`를 코드에서 완전히 제거했고, loss 쪽 가중치는 `positive_weight`만 남겼다. 기본값은 `positive_weight=0.0`이다. MPS 기준 validation도 재실행해 `pred_graph_to_raw_ratio≈0.9345`를 확인했다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `count_weight_alpha`와 관련 sweep/표기(`count_weight_alphas`, `cwa`)를 코드/노트북에서 전부 삭제했다. weighting 비교는 `positive_weight` 단일 축으로 단순화했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- iCloud 경로에서 고정 파일명 GeoTIFF overwrite 시 timeout이 가끔 발생한다. validation 스크립트는 `run_시간` 하위 출력으로 우회했지만, batch 루트 overwrite는 여전히 주의가 필요하다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- positive_weight 단일 축으로 성능/캘리브레이션 재튜닝

## Entry 24

1. 업데이트 날짜, 시각
- 2026-04-09 14:18 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. 메인 supervision은 `sum-preserving y_edge_decay`를 유지하되, loss weighting(`positive_weight`, `count_weight_alpha`)은 기본값에서 제거됐다. graph-level count calibration은 해석 보조 지표로 유지하고, full-scene merge는 같은 구조의 중복 cluster를 lifetime으로 통합하는 가중평균으로 계속 본다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- 사용자 우선순위를 반영해 총합 보존을 최우선 기준으로 두지 않기로 정리했다. `edge-decay GT`는 유지하고, full-scene merge의 lifetime 가중평균은 물리적으로 타당한 기본 해석으로 유지한다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- graph-level count 해석과 full-scene merge 해석은 분리해서 봐야 한다. merge 결과의 적분값을 척수와 직접 동일시할지 여부는 아직 최종 확정하지 않았다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- 예측 성능 우선 기준으로 sum-preserving GT와 merge 설정 추가 검증

## Entry 23

1. 업데이트 날짜, 시각
- 2026-04-09 12:02 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. graph supervision의 `y_edge_decay`는 sum-preserving target이며, 메인 기본 학습은 이제 `PoissonNLL + Softplus + 무가중치`다. batch_demo를 MPS에서 다시 검증한 결과 `raw_graph_sum=80`, `pred_graph_sum≈74.76`으로 graph-level count가 원본 척수 합과 직접 연결되는 수준까지 개선됐다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- sum-preserving target 아래에서 weighting이 count calibration만 망친다는 것을 확인하고, 기본 `positive_weight`와 `count_weight_alpha`를 0으로 바꿨다. MPS 기준 재검증도 완료했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- graph-level sum은 거의 척수 해석이 가능해졌지만, full-scene merge는 lifetime 가중평균이라 raster 합이 총 척수를 보존하지 않는다. scene assembly 단계의 count 보존이 아직 남아 있다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- count-preserving scene assembly 설계 및 full-scene 척수 보존 검증

## Entry 22

1. 업데이트 날짜, 시각
- 2026-04-09 11:32 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. graph supervision의 `y_edge_decay`는 sum-preserving target으로 유지된다. 이번에는 MPS 상태를 sandbox 밖에서 다시 확인했고, `DNB_AIS` 환경 자체는 정상이며 `torch 2.9.1`에서 `mps_available=True`, 실제 `mps:0` tensor 연산도 성공했다. 즉 현재 GPU 문제는 로컬 환경이 아니라 Codex 기본 sandbox의 제한이었다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- MPS가 실제로는 정상임을 확인했다. sandbox 안 테스트만 거짓 음성이었고, unsandboxed 실행에서는 `mps_available=True`와 `mps:0` 연산 성공을 확인했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 이후 GPU/MPS가 필요한 학습·추론은 Codex 기본 sandbox 밖에서 실행해야 한다. sandbox 상태만 보면 계속 MPS가 꺼진 것처럼 보일 수 있다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- unsandboxed MPS로 batch_demo 재실행 및 sum-preserving GT 추론 검증

## Entry 21

1. 업데이트 날짜, 시각
- 2026-04-09 11:29 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. graph supervision의 `y_edge_decay`는 sum-preserving target으로 유지된다. 다만 현재 `DNB_AIS` 환경에서 MPS는 여전히 비활성이라, 이번 턴은 GPU 복구 여부만 재점검했다. `torch 2.9.1`에서 `mps_built=True`, `mps_available=False`이며 최소 tensor 생성도 실패했다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- 코드 변경 없이 MPS를 다시 테스트했다. macOS 15.7.3, arm64 환경이지만 `DNB_AIS`에서는 여전히 `mps_available=False`로 확인됐다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- OS 버전은 조건을 만족하지만 현재 Codex 세션에서는 MPS가 비활성이다. 따라서 GPU 기준 재학습/재추론은 아직 진행할 수 없다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- MPS 사용 가능 상태 복구 후 batch_demo 재검증

## Entry 20

1. 업데이트 날짜, 시각
- 2026-04-09 11:22 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. graph supervision의 `y_edge_decay`는 이제 source별 hop kernel을 정규화해 총합이 원본 척수와 같게 보존되는 sum-preserving target이다. 따라서 graph 내부 target 합은 다시 `척` 해석과 연결된다. batch_demo 검증에서 `raw_graph_sum=80.0`, `edge_graph_sum≈80.0`으로 일치했다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `y_edge_decay`를 sum-preserving 방식으로 바꿨다. graph block에서 raw GT 합과 edge-decay 합을 같이 출력하도록 notebook을 갱신했고, batch_demo 검증 스크립트도 추가했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 현재 `DNB_AIS` 환경에서 MPS가 잡히지 않아 메인 학습/추론 재실행은 못 했다. CPU fallback은 사용자 지시에 따라 사용하지 않았다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- MPS 복구 후 sum-preserving GT 기준 batch_demo 재학습 및 척수 해석 검증

## Entry 19

1. 업데이트 날짜, 시각
- 2026-04-09 01:29 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. 메인 graph는 edge-decay GT spreading(`y_edge_decay`)을 기본 supervision으로 쓰고, raw point GT는 비교용으로만 유지한다. 현재 MPS 기반 `batch_demo` 최신 결과는 루트 출력 경로에 동기화돼 있으며, 다음 연구 단계의 핵심 과제는 퍼진 GT의 총합을 원본 선박 척수와 일치시키는 `sum-preserving edge-decay GT` 설계와 적용이다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- 코드 변경 없이 다음 진행 목표를 `sum-preserving edge-decay GT`로 확정했다. 이후 GT 해석 단위를 다시 척수 합과 맞추는 방향으로 진행한다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 현재 `y_edge_decay`는 총합이 원본 척수와 일치하지 않아 출력을 직접 `'척'`으로 읽기 어렵다. 이 문제를 해결하는 정규화/보존 방식은 아직 미구현이다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- sum-preserving edge-decay GT 설계 및 메인 경로 적용

## Entry 18

1. 업데이트 날짜, 시각
- 2026-04-09 01:11 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. 메인 graph는 edge-decay GT spreading(`y_edge_decay`)을 기본 supervision으로 쓰고, raw point GT는 비교용으로만 유지한다. 이번에는 `batch_demo` 전체 notebook을 MPS에서 끝까지 재실행했고, 최신 run 산출물을 `batch_demo` 루트 고정 경로로 다시 복사해 현재 결과를 기준으로 output들을 갈아끼웠다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `batch_demo` 전체 파이프라인을 최신 설정으로 재실행했다. `run_0409_010223` 결과를 `batch_demo` 루트로 동기화했고, notebook 실행용 import 누락도 함께 수정했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 실행은 완료됐지만 PROJ DB version warning은 계속 출력됐다. 결과 저장에는 성공했으나 GDAL/PROJ 경고는 추후 환경 정리가 필요하다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- batch_demo 최신 결과 검토 후 larger scene 확장

## Entry 17

1. 업데이트 날짜, 시각
- 2026-04-09 00:56 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. 메인 graph는 edge-decay GT spreading(`y_edge_decay`)을 기본 supervision으로 쓰고, raw point GT는 비교용으로만 유지한다. 이번에는 weighting 두 종류의 역할을 명확히 구분했고, 누적 작업 기록 `MMDD_HHMM.md` 파일들을 새 `codex_logs/` 폴더로 이동해 이후 기록도 그 안에 저장하는 기준으로 정리했다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `04*.md` 작업 기록 파일들을 모두 `codex_logs/`로 옮겼다. directed edge는 현재 규모에선 속도상 이득이 크지 않다고 판단해 유지하지 않았다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- `codex_logs/` 이동은 완료됐지만, 과거 일부 메시지에서 언급한 경로는 루트 기준으로 남아 있을 수 있다. directed edge 최적화는 larger scene에서만 다시 검토하면 된다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- codex_logs 기준 유지, smoothed GT 기본 경로 재검증

## Entry 16

1. 업데이트 날짜, 시각
- 2026-04-09 00:54 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. 메인 graph는 edge-decay GT spreading(`y_edge_decay`)을 기본 supervision으로 쓰고, raw point GT는 비교용으로만 유지한다. 이번에는 테스트 전용 비교 산출물과 overfit 결과 디렉터리를 정리해 출력 폴더를 기본 `batch_demo` 결과 중심으로 정돈했다. MPS 기반 기본 경로와 undirected graph spreading은 그대로 유지한다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `batch_demo_compare`, `inverse_brightness_compare`, `poisson_compare`, `overfit_troubleshooting` 등 테스트 산출물을 삭제했다. 현재 규모에선 directed edge로 바꾸지 않아도 spreading 비용이 감당 가능하다고 판단했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- directed edge 최적화는 아직 구현하지 않았다. 다만 현재 `batch_demo` 규모에선 필요성이 크지 않다. larger scene이나 더 큰 반경에서 edge 수가 급증하면 다시 검토해야 한다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- MPS 기준 smoothed GT 기본 경로 재실행 및 larger scene 비용 점검

## Entry 15

1. 업데이트 날짜, 시각
- 2026-04-09 00:49 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. 이제 메인 graph config 자체에 edge-decay GT spreading이 들어가고, 기본 training target도 `y_edge_decay`를 사용한다. raw point GT(`y`)는 비교/진단용으로만 유지된다. notebook은 MPS가 아니면 즉시 중단되며, overfit troubleshooting에서는 `positive_weight x count_weight_alpha` full grid로 point GT와 smoothed GT를 함께 비교할 수 있다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- GT spreading을 troubleshooting 전용이 아니라 메인 경로에 반영했다. 기본 graph는 `(1.0, 0.6, 0.2)` hop decay를 만들고, 기본 training target은 `y_edge_decay`를 쓰도록 바꿨다. notebook도 MPS 강제 방식으로 정리했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 이번 턴은 설정 승격과 정리 단계라 full notebook 재실행은 하지 않았다. `y_edge_decay` 메인화 이후 patch set 성능과 ranking 변화는 아직 다시 검증되지 않았다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- MPS에서 smoothed GT 기본 경로 재실행 및 patch set 검증

## Entry 14

1. 업데이트 날짜, 시각
- 2026-04-09 00:35 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. 기본 파이프라인은 raw point GT와 compressed brightness를 유지한다. 현재 notebook 기본 training 설정의 `count_weight_alpha`는 20으로 올라갔고, troubleshooting 경로에서는 single-graph overfit으로 `positive_weight x count_weight_alpha`를 비교할 수 있다. weighting-grid와 loss comparison 블록도 같은 기준값을 보도록 정리했다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- 메인 notebook 기본 `count_weight_alpha`를 6에서 20으로 올렸다. weighting-grid 비교와 loss-weighting helper도 20 기준으로 맞췄다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 이번 턴에서는 설정만 바꾸고 full notebook 재실행은 하지 않았다. `cwa=20` 기본값이 patch set 기준으로도 가장 균형적인지는 후속 검증이 필요하다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- cwa=20 기본값으로 소규모 patch set 재검증

## Entry 13

1. 업데이트 날짜, 시각
- 2026-04-09 00:27 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 irregular contour patch를 만든다. 기본 파이프라인은 raw point GT와 compressed brightness를 유지한다. troubleshooting 경로에서는 undirected graph와 edge-decay smoothed GT(`y_edge_decay`)를 만들어 single-graph overfit 진단을 수행한다. 현재 notebook의 troubleshooting block은 `positive_weight x count_weight_alpha` 4x4 sweep를 포함해 두 가중의 단독 효과와 조합 효과를 같이 비교한다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `count_weight_alpha`를 overfit troubleshooting의 full grid에 포함했다. `cwa=100` 단독이 `pred_max≈1.033`으로 가장 높았고, `pw20` 단독도 `≈0.996`까지 올라갔다. 조합도 일부는 1 부근까지 회복됐다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 이번 결론은 single-graph 기준이다. `count_weight_alpha`의 거동이 비단조적이고, 최고 `pred_max`와 최저 `train_loss`가 같은 설정에서 나오지 않아 patch set 학습에서의 일반화 검증이 필요하다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- 소규모 patch set에서 `count_weight_alpha` 일반화 여부 검증

## Entry 12

1. 업데이트 날짜, 시각
- 2026-04-09 00:11 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. 기본 파이프라인은 raw point GT와 compressed brightness를 유지한다. 추가로 troubleshooting 경로에서는 graph를 undirected로 고정하고, edge-decay smoothed GT(`y_edge_decay`)를 별도 생성해 weighted MSE overfit 진단을 수행할 수 있다. notebook에는 single-ship positive cluster를 골라 peak recovery를 확인하는 블록이 추가됐다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- GT smoothing을 메인 정답으로 덮지 않고 별도 troubleshooting target으로 추가했다. single-graph overfit block을 넣고 `gt_sum=1` cluster에서 smoothed GT + weighted MSE가 `pred_max≈0.997`까지 올라가는 것을 확인했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 이번 결과는 single-graph overfit 진단이라 patch set 일반화는 아직 검증되지 않았다. 현재 개선은 peak recovery 관점의 확정 방향이지, full pipeline 최종 loss 결정이 끝났다는 뜻은 아니다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- smoothed GT 경로를 소규모 patch set 학습에 확장해 ranking 개선 여부 확인

## Entry 11

1. 업데이트 날짜, 시각
- 2026-04-08 20:56 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. patch membership은 DRUID smooth field 기준으로 복원한다. DRUID와 GAT brightness feature는 다시 모두 기존 compressed GeoTIFF 밝기값을 사용하며, 각 patch는 pixel-node graph로 변환된다. 현재 기본 회귀는 Softplus head와 PoissonNLLLoss를 사용하고, notebook에서는 weighting grid sweep과 checkpoint 저장을 재현할 수 있다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- inverse brightness 실험을 되돌려 GAT brightness도 다시 compressed 값으로 복구했다. `/tmp`에 쌓였던 비교용 임시 스크립트들도 함께 삭제했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- brightness 입력은 원래대로 복구됐지만, Poisson 설정의 weighting과 epoch 최적화는 여전히 미완료다. inverse brightness 비교 결과물은 outputs에 남아 있으나 현재 기본 파이프라인에는 사용되지 않는다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- compressed brightness 기준 weighting과 epoch 추가 비교

## Entry 10

1. 업데이트 날짜, 시각
- 2026-04-08 20:52 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. patch membership은 DRUID smooth field 기준으로 복원한다. DRUID는 기존 compressed GeoTIFF 밝기를 그대로 보며, GAT brightness feature만 `[A]`의 arctan 압축식을 inverse로 되돌린 radiance 값을 추가 정규화 없이 사용한다. 현재 기본 회귀는 Softplus head와 PoissonNLLLoss를 사용하고, notebook에서 weighting grid sweep을 바로 돌릴 수 있다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- inverse brightness 기준으로 `positive_weight`와 `count_weight_alpha`의 8개 조합 grid sweep을 추가했다. notebook도 단일 positive-weight sweep 대신 weighting-grid CSV를 저장하도록 바꿨다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 현재 inverse brightness 세팅에서는 weighting이 output amplitude만 키우고 spatial ranking은 거의 못 바꿨다. top-k hit가 모든 조합에서 동일해, 학습 안정성과 feature scale 재설계가 필요하다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- inverse brightness feature scaling 또는 epoch 증가 비교

## Entry 9

1. 업데이트 날짜, 시각
- 2026-04-08 20:47 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. patch membership은 DRUID smooth field 기준으로 복원한다. 현재 DRUID는 기존 compressed GeoTIFF 밝기를 그대로 보며, GAT brightness feature만 `[A]`의 arctan 압축식을 inverse로 되돌린 radiance 값을 추가 정규화 없이 사용한다. 각 patch는 pixel-node graph로 변환되며, 기본 회귀는 Softplus head와 PoissonNLLLoss를 사용한다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- `[A]`의 밝기 압축식 inverse를 찾아 GAT brightness feature에만 적용했다. DRUID 입력은 그대로 유지했고 notebook 기본 graph 설정도 `reverse_arctan_raw` 모드로 바꿨다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- inverse brightness는 추가 normalize 없이 바로 쓰기 때문에 feature scale이 이전과 크게 달라졌다. 이 변경 뒤에는 학습 안정성, weighting, loss tuning을 다시 봐야 한다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- inverse brightness 기준으로 학습 결과와 weighting 재검증

## Entry 8

1. 업데이트 날짜, 시각
- 2026-04-08 20:41 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. patch membership은 DRUID smooth field 기준으로 복원하고, GAT 입력 밝기는 raw radiance를 유지한다. 각 patch는 pixel-node graph로 변환되며 GATv2Conv가 픽셀별 선박 밀도를 예측한다. 현재 기본 회귀는 Softplus head와 PoissonNLLLoss를 사용하고, positive_weight sweep과 checkpoint 저장을 notebook에서 직접 재현할 수 있다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- 현재 Poisson 설정에서 `positive_weight=0,10,20,30` sweep을 추가했고 notebook에도 같은 CSV 저장 블록을 넣었다. 해석 혼선을 막기 위해 완전 무가중치 기준도 별도로 확인했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 이론적으로는 Poisson이 더 자연스럽지만, 현재 batch_demo 성능은 아직 이전 MSE 기준보다 낮다. Poisson 쪽은 `positive_weight`와 `count_weight_alpha`, epoch 재튜닝이 더 필요하다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- Poisson에서 `positive_weight=20` 근방과 `count_weight_alpha` 추가 스윕

## Entry 7

1. 업데이트 날짜, 시각
- 2026-04-08 20:36 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. patch membership은 DRUID smooth field 기준으로 복원하고, GAT 입력 밝기는 raw radiance를 유지한다. 각 patch는 pixel-node graph로 변환되며 GATv2Conv가 픽셀별 선박 밀도를 예측한다. 현재 기본 회귀는 Softplus head와 PoissonNLLLoss를 사용하고, 학습 후 checkpoint를 로컬 `.pt`와 `.json`으로 저장한다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- 기본 loss를 `MSE`에서 `PoissonNLLLoss`로, output head를 `ReLU`에서 `Softplus`로 바꿨다. 같은 batch_demo 조건으로 이전 MSE 실험과 직접 비교하는 CSV와 새 heatmap/checkpoint를 저장했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- 이론적으로는 count target에 더 자연스럽지만, 현재 batch_demo 비교에선 Poisson 설정이 이전 MSE보다 top-k hit와 peak에서 우세하지 않았다. Poisson용 가중치와 epoch 재조정이 필요하다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- Poisson 가중치/epoch 재튜닝 또는 MSE 유지 여부 결정

## Entry 6

1. 업데이트 날짜, 시각
- 2026-04-08 20:23 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. patch membership은 DRUID smooth field 기준으로 복원하고, GAT 입력 밝기는 raw radiance를 유지한다. 각 patch는 pixel-node graph로 변환되며 GATv2Conv가 픽셀별 선박 밀도를 예측한다. 학습 후에는 모델 checkpoint를 로컬 `.pt`와 요약 `.json`으로 저장해 이후 재로딩과 재사용이 가능하다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- GAT 모델의 아키텍처, state_dict, graph/training config, 실행 metadata를 로컬 checkpoint로 저장하고 다시 불러오는 기능을 추가했다. notebook에서도 저장 경로와 파일 크기를 출력한다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- KR full-scene용 checkpoint는 아직 저장 검증하지 않았다. Desktop/iCloud 경로에서는 기존 출력 파일 overwrite가 멈출 수 있어 fresh subdir 저장 방식을 계속 유지해야 한다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- KR full-scene checkpoint 검증, min_nodes 확정, count-aware loss 후속 조정

## Entry 5

1. 업데이트 날짜, 시각
- 2026-04-08 20:16 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. patch membership은 DRUID smooth field 기준으로 복원하고, GAT 입력 밝기는 raw radiance를 유지한다. 각 patch는 pixel-node graph로 변환되며 GATv2Conv가 픽셀별 선박 밀도를 예측한다. 결과는 cluster_id와 lifetime을 유지한 채 병합되며, notebook은 fresh run subdirectory에 시각화/CSV/GeoTIFF를 저장한다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- codex_stamp를 최신값 overwrite 방식이 아니라 누적 stack push 형식으로 재구성했다. pipeline 본체 변경은 없다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- min_nodes는 아직 확정하지 않았고 KR full-scene은 재검증 전이다. 기존 `batch_demo` 폴더의 직접 overwrite는 Desktop/iCloud 영향으로 멈출 수 있어 fresh subdir 방식으로 우회 중이다. PROJ warning도 남아 있다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- min_nodes 확정, count-aware loss 후속 튜닝, KR full-scene 재검증

## Entry 4

1. 업데이트 날짜, 시각
- 2026-04-08 20:06 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. patch membership은 DRUID smooth field 기준으로 복원하고, GAT 입력 밝기는 raw radiance를 유지한다. 각 patch는 pixel-node graph로 변환되며 GATv2Conv가 픽셀별 선박 밀도를 예측한다. 결과는 cluster_id와 lifetime을 유지한 채 병합되며, notebook은 fresh run subdirectory에 시각화/CSV/GeoTIFF를 저장한다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- representative graph visualization과 loss weighting sweep을 추가했고 기본 학습 가중치를 `count_weight_alpha=6`으로 유지했다. notebook 출력은 RUN_TAG별 fresh subdir에 저장되도록 바꿨다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- min_nodes는 아직 확정하지 않았고 KR full-scene은 재검증 전이다. 기존 `batch_demo` 폴더의 직접 overwrite는 Desktop/iCloud 영향으로 멈출 수 있어 fresh subdir 방식으로 우회 중이다. PROJ warning도 남아 있다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- min_nodes 확정, count-aware loss 후속 튜닝, KR full-scene 재검증

## Entry 3

1. 업데이트 날짜, 시각
- 2026-04-08 19:09 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. patch membership은 DRUID의 smooth field 기준으로 복원하고, GAT 입력 밝기는 raw radiance를 유지한다. 각 patch를 pixel-node graph로 변환해 반경 기반 엣지와 GATv2Conv로 픽셀별 선박 밀도를 예측하며, 결과는 cluster_id와 lifetime을 보존한 채 lifetime 가중평균으로 병합해 geocoded heatmap GeoTIFF로 저장한다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- graph receptive-field sweep(radius 2/4/6, layers 2/3)을 추가했고 batch_demo 기본 graph radius를 4로 조정했다. notebook에서 sweep 결과 CSV를 저장하고 같은 설정으로 재실행 검증했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- min_nodes는 아직 확정하지 않았고 graph radius도 추가 검증 여지가 있다. KR full-scene은 아직 재검증 전이며 GeoTIFF 저장 시 로컬 PROJ DB warning은 계속 출력된다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- min_nodes 확정, graph radius 후속 비교, KR full-scene 재검증

## Entry 2

1. 업데이트 날짜, 시각
- 2026-04-08 18:34 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보를 찾은 뒤 irregular contour patch를 만든다. patch membership은 DRUID가 실제로 본 smooth field 기준으로 복원하고, GAT 입력 밝기는 raw radiance를 유지한다. 각 patch를 pixel-node graph로 변환해 PyG GATv2Conv로 픽셀별 선박 밀도를 예측하며, 결과는 cluster_id와 lifetime을 보존한 채 lifetime 가중평균으로 병합해 geocoded heatmap GeoTIFF로 저장한다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- DRUID top-k cap을 기본 비활성화했고 contour membership을 smooth image 기준으로 수정했다. area_limit sweep(4/8/12/16)과 seed 고정 셀을 추가했고 batch_demo 기본 area_limit을 12로 조정했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- min_nodes는 아직 확정하지 않았고 이번 sweep 결과를 바탕으로 후속 결정이 필요하다. KR full-scene은 아직 재검증 전이며 GeoTIFF 저장 시 로컬 PROJ DB warning은 계속 출력된다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- min_nodes 확정, KR full-scene 재검증, 필요 시 DRUID sigma와 area_limit 추가 비교

## Entry 1

1. 업데이트 날짜, 시각
- 2026-04-08 16:40 KST

2. 전체 pipeline에 대한 상세한 description (공백 포함 500자 이내)
- GeoTIFF DNB 영상을 입력으로 받아 GT geojson/DB를 확인하고, DRUID로 선박 후보 contour patch를 추출한 뒤 각 patch를 pixel-node graph로 변환한다. PyG GATv2Conv가 밝기와 로컬 좌표를 이용해 픽셀별 선박 밀도를 예측하며, patch 결과는 cluster_id와 lifetime을 보존한 채 저장된다. 마지막에는 겹치는 픽셀을 lifetime 가중평균으로 병합해 원본 격자와 같은 geocoded heatmap GeoTIFF를 생성한다.

3. 가장 최근 pipeline과 비교했을 때의 변경 사항 요약 (공백 포함 200자 이내)
- sub_module class 구조, DNB_GAT_v1.ipynb, notebook 생성 스크립트를 추가했고 batch_demo 기준 DRUID→graph→GAT→GeoTIFF 저장까지 실행 검증했다.

4. 발생한 이슈들 중 해결하지 못한 이슈들에 대한 설명 (공백 포함 200자 이내)
- KR full-scene DRUID는 시간이 길어 demo 기본값을 batch로 뒀다. GeoTIFF 저장 시 로컬 PROJ DB warning이 1회 출력되지만 결과 저장은 정상이다.

5. 다음 단계로 계획 중인 task에 대한 description (공백 포함 100자 이내)
- KR full-scene 파라미터 튜닝, DRUID cluster 상한 조정, git push 후 후속 지시 대응
