# codex_stamp

최신 entry를 맨 위에 추가하는 누적 로그 형식으로 유지한다. 기존 entry는 지우지 않는다.

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
