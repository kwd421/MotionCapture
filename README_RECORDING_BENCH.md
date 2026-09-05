# 녹화 영상 고정 입력 벤치

기존 live_app / motioncapture-demo는 그대로 사용합니다. 이 명령은 **별도로
선택하는 파일 벤치**이며, 라이브 카메라 실패 시 대신 실행되지 않습니다.

필요 조건: 기존 uv 환경과 고정 모델 파일, PATH에 `ffprobe` 실행파일.
`ffprobe -version`으로 확인하세요. 새 Python 의존성은 추가하지 않았습니다.

```bash
uv run python -m motioncapture.recording_bench /path/to/video.mp4 --mode track --repeats 3 --preview display --output sessions/recording60.json
```

영상 경로만 실제 경로로 바꾸면 됩니다. 이미 존재하는 report를 덮어쓰지 않습니다.
평균·p50·p95·p99, 16.67ms 예산 초과 횟수, 디코딩/추론/표시 구성 시간을 각각
기록합니다. CPU Pose/Hands/Face 세 작업과 숫자 결과 검증을 모두 실행합니다.
변경 전후에는 같은 원본 SHA와 환경, 작업 순서, preview 설정을 맞추세요.

영상 프레임을 생략하지 않고 원본 PTS 순서로 최대한 빨리 처리합니다.
따라서 `unpaced_loop_fps`는 이 파일 처리 경로의 용량이며, 카메라 실측 FPS나
모션-광자 지연이 아닙니다. JPEG/PNG, 음성, 개별 얼굴값/관절값은 저장하지 않습니다.
오디오는 입력 영상 안에 있어도 사용하지 않습니다.

동일 원본 전체를 매번 새 트래커로 시작합니다. 초기 60프레임을 포함한 전체 통계와
그 이후의 steady 통계를 동시에 내며, 초기 프레임을 숨기거나 버리지 않습니다.
10초 원본 시간 구간별 지표와 `resolved_hands=0/1/2,face=0/1`별 지표가 있습니다.
이 구분은 검출된 결과 기준이며 실제 보임 여부의 정답이나 palm detector 실행
횟수가 아닙니다. 손이 안 보였다는 것만으로 실패/품질 회귀로 판정하지 마세요.

비교 옵션:

```bash
# 표시는 계산하되 원래 renderer를 선택
uv run python -m motioncapture.recording_bench /path/to/video.mp4 --preview native --output sessions/native-preview.json

# 모델 3개는 모두 실행, preview만 명시적으로 제외
uv run python -m motioncapture.recording_bench /path/to/video.mp4 --preview none --output sessions/tracker-only.json

# 추론 없이 디코더/PTS 검증만. 60FPS 추적 성공으로 해석하면 안 됨.
uv run python -m motioncapture.recording_bench /path/to/video.mp4 --mode decode --decode-threads 4 --output sessions/decode.json
```

`--decode-threads 0`(기본)은 디코더 기본값입니다. 실제 보고값도 기록합니다.
`--verify-pixels`는 모든 출력 BGR 픽셀의 rolling SHA-256을 계산하며, 그 비용은
별도로 기록하지만 loop FPS에는 포함됩니다. 이 옵션의 유무를 섞어 FPS를 비교하지
마세요. 프로브/소스 체크섬/모델 초기화/최종 검증 시간은 프레임 루프와 구분됩니다.

## 시간 정확성

FFprobe의 정수 PTS + rational time base를 보존합니다. OpenCV FFmpeg의
각 프레임 시각이 그 PTS와 일치하는지도 확인합니다. 프레임번호/60으로 시간을
만들지 않으며, 반복 프레임 삽입, 보간, 리사이즈, 자동 회전, 자동 provider/model
변경을 하지 않습니다. 프로브와 디코더 개수/시각이 어긋나면 중단합니다.

`RecordedIdentity`에는 실제 `pts`/`time_base`가 있으며 host receive timestamp가
아닙니다. live와 file은 같은 모델 계산 코드를 쓰지만 같은 트래커 수명 안에서
시계/입력 소스를 섞을 수 없습니다.

## 확인한 범위

26개 신규 대상 테스트를 Linux/Python 3.13.5에서 통과했습니다. FFmpeg/OpenCV
VFR 합성 영상의 실제 디코딩과 원본 두 영상의 전체 디코딩은 실행했습니다.
트래커 시간 계약 테스트는 명시적인 테스트 대역을 사용했습니다.

이 구현 환경에는 MediaPipe 0.10.31이 없고 네트워크 설치도 실패했으므로,
**실제 AI 추론 A/B, 모델 정확도, MacBook 60FPS 달성은 아직 검증하지 못했습니다.**
고정 Python 3.12 실행과 Ruff 전체 검증도 별도 실기기 게이트입니다.
`--mode track`은 해당 패키지가 없으면 실패 보고서를 남기며 decode-only로
자동 전환하지 않습니다. `benchmarks/2026-09-06-recording-inputs.md` 참조.
