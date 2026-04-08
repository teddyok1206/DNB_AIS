# codex_stamp

최신 entry를 맨 위에 추가하는 누적 로그 형식으로 유지한다. 기존 entry는 지우지 않는다.

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
