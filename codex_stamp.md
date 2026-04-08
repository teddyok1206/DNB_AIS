# codex_stamp

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
