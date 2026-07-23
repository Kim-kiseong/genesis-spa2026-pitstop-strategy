# preprocessing/ — 팀원 A(Data Engineer) 담당

5일 스프린트 문서의 `preprocessing/` 디렉토리 그대로입니다. Day 1~3 작업이 이미
실데이터(2021·2022·2023 SPA, 2024·2025 SPA/IMOLA, 2026 SPA)로 검증 완료된 상태로 들어있습니다.

## 실행

```bash
pip install -r requirements.txt

# raw_data/ 에 원본 CSV 배치 (파일명 규칙은 아래 참고)
python build_dataset.py
# -> processed/laps_train.parquet, processed/laps_val.parquet
```

## 스프린트 문서 체크리스트 대응

| Day | 항목 | 상태 | 코드 위치 |
|---|---|---|---|
| 1 | `stint_id`/`stint_lap` 파생 | ✅ 완료 | `stint.py::reconstruct_stints` |
| 2 | 날씨 데이터 UTC 보정 후 `merge_asof` 조인 | ✅ 완료 | `weather.py::merge_weather` |
| 2 | `weather_category`(0/1/2) 라벨링 | ✅ 완료 | `weather.py` (DRY/WET/MIXED) |
| 3 | 아웃라이어(레드플래그) 제거 | ✅ 완료 | `outliers.py::drop_track_status_outliers` |
| 3 | 2021~24 train / 2025 val 스플릿 | ✅ 완료 | `build_dataset.py`, `config.py::TRAIN_EVENTS/VAL_EVENTS` |
| 3 | 최종 `laps.parquet` | ✅ 완료 | `laps_train.parquet` + `laps_val.parquet` (스프릿 문서 취지상 2개로 분리) |

**Day 1~3는 이미 다 되어 있습니다.** 오늘 3시간은 raw_data에 실제 파일 넣고 `python build_dataset.py`
한 번 돌려서 본인 컴퓨터에서도 똑같이 나오는지 확인 + B/C에게 산출물 넘기는 데 쓰면 됩니다.

## 데이터 범위 (확정)

원본 아카이브에서 받을 수 있는 데이터가 다음 7개 이벤트로 확정됐습니다 — 21~23년 이몰라는
아카이브에 아예 제공되지 않아 더 이상 늘어나지 않습니다.

| Train (5개) | Val (2개) |
|---|---|
| 2021 SPA, 2022 SPA, 2023 SPA, 2024 SPA, 2024 IMOLA | 2025 SPA, 2025 IMOLA |

에피소드 수가 적어 걱정되면(기획서 3-2절 피드백 요청사항 2), PPO 학습 시 한 이벤트 안에서
여러 차량(HYPERCAR 클래스 전체)을 각각 독립 에피소드로 쓸 수 있습니다. 실제로 세어보니:

| 이벤트 | 차량 수 |
|---|---|
| 2021 SPA | 3 (Hypercar 규정 첫 해라 원래 적음) |
| 2022 SPA | 4 |
| 2023 SPA | 13 |
| 2024 SPA / IMOLA | 19 / 19 |
| 2025 SPA / IMOLA (val) | 18 / 18 |

**train 58개, val 36개 (event, 차량) 조합** — 이벤트 수(7)보다 훨씬 많습니다. 각 조합이
하나의 완주 레이스(수백 랩) 에피소드라 학습 신호 자체는 충분할 걸로 보입니다. 다만 2021·2022는
차량 수가 적어서 그 두 해 비중을 너무 높게 학습에 반영하지 않도록(과적합 방지) 주의하면 좋습니다.

## raw_data 파일명 규칙

파일명이 고정 패턴을 안 따라도 됩니다. `loaders.py::_find_raw_file()`이 "2자리 연도 + 서킷명
+ analysis/weather/classification 키워드"가 포함된 파일을 대소문자 무시하고 자동으로 찾습니다.

```
raw_data/21_SPA_Analysis_Race_Hour_6.csv
raw_data/21spa_Weather_Race_Hour_6.csv
```

**주의 (실제로 겪은 함정):** Al Kamel 아카이브 파일명의 숫자 접두어가 항상 연도는 아닙니다
(문서 ID인 경우가 있음). 자동 탐색이 안 되면 `raw_data/{year}_{circuit}_lap.csv` /
`_weather.csv` / `_classification.csv`로 직접 리네임하세요(고정 규칙도 폴백으로 지원됨).

## B(evaluator/), C(rl_env/)에게 넘기는 인터페이스

`processed/laps_train.parquet`, `processed/laps_val.parquet` 두 파일이 전부입니다.
B와 C는 이 저장소의 다른 코드를 볼 필요 없이 아래 컬럼만 알면 됩니다.

**B가 XGBoost 입력으로 쓸 컬럼 (기획서 상태 변수):**
| 컬럼 | 설명 |
|---|---|
| `LAP_PROGRESS_RATIO` | 레이스 진행률 (0~1) |
| `STINT_LAP` | 스틴트 내 경과 랩 수 (타이어/연료 마모 대리 변수) |
| `CLASS_POSITION` | 실시간 클래스 순위 |
| `GAP_TO_LEADER_SEC` | 선두와의 격차(초) |
| `GAP_TO_AHEAD_SEC` | 바로 앞 차량과의 격차(초) |
| `WEATHER_CATEGORY` | 0=DRY, 1=WET, 2=MIXED |
| `TRACK_TEMP` | 트랙 온도 |
| `PODIUM` | 라벨(bool) — 최종 클래스 순위 3위 이내 여부 |

학습 시 `EVENT_ID` + `NUMBER`로 groupby해서 같은 레이스가 train/val에 안 섞이게 주의
(이미 `laps_train.parquet`/`laps_val.parquet`로 분리해뒀지만, 한 이벤트 안에서 추가로 쪼갤 땐 주의).

**C가 Gymnasium 환경에서 쓸 추가 컬럼:**
| 컬럼 | 설명 |
|---|---|
| `IS_PIT_LAP` | 실제 피트인 여부 (B마커 기준, 정확) |
| `STINT_ID` | 스틴트 번호 (에피소드 경계로 쓰기 좋음) |
| `PIT_LOSS_S` | 실측 피트로스(초). `OUTLIER_PIT_LOSS=True`인 행은 사고/기계고장 등 비정상값이니 보상 계산에서 제외 권장 |
| `TRACK_STATUS` | GREEN/SAFETY_CAR/FULL_COURSE_YELLOW/FINISH (RED_FLAG는 이미 드롭됨) |

액션 마스킹 임계값(`MAX_STINT_LAPS=34`, `MIN_STINT_LAPS=3`)은 `config.py`에 있는 값을
C의 환경 코드에도 **반드시 동일하게** 맞춰야 합니다 — 안 그러면 전처리 단계 가정과
RL 환경 가정이 어긋납니다.

## Day 4용 헬퍼

```bash
python extract_actual_timeline.py --event 2025_SPA --number 7
# -> processed/actual_timeline_2025_SPA_7.csv (C가 "실제 vs AI" 비교 그래프에 바로 쓸 수 있음)
```

## 이미 겪은 실데이터 함정들 (다시 안 겪어도 되게 정리)

1. **레드플래그는 `FF`가 아니라 `FLAG_AT_FL`의 `RF`.** 처음에 반대로 알았다가 실데이터로
   검증하면서 정정함. `RF`는 2022_SPA, 2024_SPA에서만 등장.
2. **차량 번호 앞자리 0 소실.** `NUMBER`를 정수로 읽으면 "007"(Aston Martin)이 "7"(Toyota)과
   섞임. 반드시 `dtype={"NUMBER": str}`로 읽어야 함(이미 반영됨).
3. **UTC 오프셋 부호.** CEST는 UTC+2 (마이너스 아님).
4. **날씨 `TIME_UTC_SECONDS`는 유닉스 epoch.** `% 86400`으로 스케일 맞춰야 함(이미 반영됨).
5. **`ELAPSED`/`LAP_TIME`/`PIT_TIME`은 전부 "M:SS.mmm" 문자열.** `pd.to_numeric` 쓰면 안 됨
   → `time_utils.py`의 전용 파서 사용(이미 반영됨).
6. **`PIT_TIME`은 B마커 랩이 아니라 다음 랩(아웃랩)에 찍힘.** `IS_PIT_LAP` 판정은 B마커로만
   해야 함 — PIT_TIME까지 OR 하면 피트 횟수가 2배로 잡힘(이미 반영됨).
7. **23_SPA는 `RAIN=0`인데 실제로는 젖은 노면.** `config.MANUAL_WEATHER_OVERRIDE_EVENTS`로
   이벤트 전체를 WET 강제 라벨링(이미 반영됨).
