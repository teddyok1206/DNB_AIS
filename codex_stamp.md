# codex_stamp

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
