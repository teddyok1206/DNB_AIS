# codex_stamp

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
