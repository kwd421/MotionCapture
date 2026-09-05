# MotionCapture 성능 최적화 후보

기준 브랜치: `feat/capture-runtime-boundary`
기준 커밋: `932e8e8d90a0b86629d4e8d35ea165ba87edc094`

이번 묶음은 **새 파일만 추가**합니다. 기존 `motioncapture-demo`, 모델,
해상도, 추적 설정, `pyproject.toml`, `uv.lock`은 바꾸지 않습니다.
최적화 후보는 `feat/capture-runtime-boundary` 브랜치에 별도 커밋으로 반영할 수 있도록 구성했습니다.

## 적용

기존 MotionCapture 레포 폴더에서, 함께 제공한 패치의 실제 경로로 실행합니다.
기존 변경을 강제로 지우거나 덮어쓰는 명령은 사용하지 마세요.

```bash
git switch feat/capture-runtime-boundary
git apply --check /path/to/MotionCapture_performance.patch
git apply /path/to/MotionCapture_performance.patch
uv run pytest
uv run ruff check .
```

ZIP에는 패치와 동일한 추가 소스, 테스트, 명세, 벤치마크가 있습니다.
원본 레포 전체를 대체하는 ZIP이 아닙니다. 기존 파일 충돌이나 기준 API 변경이
있으면 적용을 중지하고 차이를 검토하세요.

## 실행

```bash
# 원래 코드: 그대로 남아 있습니다.
uv run motioncapture-demo --camera-index 0 --session-dir sessions/baseline

# 최적화 후보: 같은 카메라/CPU 모델. 추론-표시 overlap + display preview.
uv run python -m motioncapture.live_app --camera-index 0 --session-dir sessions/optimized

# 화면 없이 정확히 120프레임. 프리뷰 합성도 실행합니다.
uv run python -m motioncapture.live_app --headless-frames 120 --session-dir sessions/optimized-headless

# 추가 계측은 유지하며 overlap과 프리뷰 변경을 모두 끄는 비교 경로.
uv run python -m motioncapture.live_app --pipeline-scheduling sequential --preview-mode native --session-dir sessions/control

# 평균, 꼬리 지연, 실패/정리 상태, 설정 차이 비교. 자동 품질 판정은 하지 않습니다.
uv run python tools/compare_live_sessions.py sessions/baseline/BASE.json sessions/optimized/NEW.json
```

`--max-frames 3600`은 화면을 표시하며 3600프레임을 처리한 뒤 종료합니다.
`--no-mirror`는 미리보기만 반전 해제합니다. `Q`/`Esc`는 종료입니다.
`--self-check`는 기존 모델 검사 경로를 사용하고 카메라는 열지 않습니다.

## 로그에서 볼 것

기존 schema-v2 필드가 유지되고 `performance` 확장 필드가 추가됩니다.
`performance.stage_distributions`는 단계별 p50/p95/p99 상한을 제공합니다.
`performance.additional_stages`는 `event_pump`, `presentation_submit`,
`result_residence` 등의 누락되었던 구간을 제공합니다.

캡처 큐에서 오래된 입력을 교체한 횟수(`capture.replaced`)와,
종료 시 사용하지 않은 선행 추론 결과(`performance.pipeline.discarded`)는
다른 값입니다. 정상 headless N 실행에서 선행 N+1 추론은 없습니다.
대화형 종료에서는 최대 한 개의 선행 결과가 사용되지 않을 수 있습니다.

이 로그의 시간은 **카메라 센서부터 실제 화면 빛까지의 지연이 아닙니다**.
미리보기 완료, GUI 제출, GUI 이벤트 처리의 호스트 시각을 구분합니다.

## 검증 범위

47개 대상 테스트 통과. 실제 OpenCV 픽셀 처리를 포함하고, 카메라/추론은
명시적 테스트 더블을 사용했습니다. 로컬 환경은 Linux / Python 3.13.5입니다.
Python 3.12 문법 검사는 통과했지만, 3.12 실행·Ruff·원래 레포 전체 테스트·
실제 MediaPipe·Mac/Windows 카메라 및 발열 테스트는 아직 하지 못했습니다.

합성 카메라+오버레이 커널 벤치마크는 4.0582 → 2.6743ms였지만,
**맥북 전체 앱이 34.1% 빨라졌다는 뜻은 아닙니다**.
자세한 근거와 한계는 `benchmarks/2026-09-06-live-performance.md`에 있습니다.

몸, 손, 얼굴 중 하나를 생략하거나 이전 추적값을 현재 값처럼 재사용하지
않습니다. native crash 격리나 스레드 강제 종료 기능도 주장하지 않습니다.
