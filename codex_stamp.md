# codex_stamp

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
