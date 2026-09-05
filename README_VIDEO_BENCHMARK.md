# 같은 영상으로 벤치마크하기

기준: `feat/capture-runtime-boundary`, `dd92ae0` 다음의 영상 벤치 확장.
실시간 기본 모델·해상도·검출 임계값·RGB 할당 방식은 바꾸지 않습니다.

`ffprobe`가 PATH에 있어야 합니다(FFmpeg 도구). 모델이 없다면 기존
`uv run motioncapture-models fetch`로 명시적으로 받아야 합니다.

```bash
uv sync
ffprobe -version

# 같은 영상 전체, 원본 PTS 사용. allocated/reuse/reuse/allocated 4회.
uv run python -m motioncapture.video_benchmark \
  /path/to/input-20260905T175716Z.mp4 \
  --stage track --rounds 2 --target-fps 60
```

결과 JSON은 `sessions/video-bench/`에 고유 이름으로 저장됩니다.
같은 출력 파일을 덮어쓰지 않습니다. 기본값은 전처리 버퍼 할당/재사용
후보의 비교입니다. `--rgb-modes allocated`로 기준 방식만 실행할 수 있습니다.
`--task-scheduling serial`은 별도 비교 실험이며 오류 시 자동 선택되지 않습니다.

영상은 720p 약 30fps입니다. `--target-fps 60`은 16.67ms 계산 예산과
비교하라는 뜻이지 영상을 60fps로 복제하거나 PTS를 바꾸는 옵션이 아닙니다.
보고서의 처리속도는 파일 처리 능력이지 실시간 60fps 전체 시스템의 검증이 아닙니다.

```bash
# AI 실행 없이 파일 자체를 검사하는 별도 모드
uv run python -m motioncapture.video_benchmark VIDEO.mp4 --stage inspect

# 실제 픽셀 전체로 전처리만 비교하는 별도 모드
uv run python -m motioncapture.video_benchmark VIDEO.mp4 --stage preprocess --rounds 3
```

추론 실패 시 위 모드들로 자동 대체하지 않습니다. 원본 영상·개별 얼굴 및
관절값은 저장소에 올리지 않습니다. 보고서는 집계 시간·검출 횟수·해시만
남깁니다. 손 개수별 시간은 관측 상관관계이며 내부 재검출 호출 기록은 아닙니다.

이 환경에서는 3,599프레임 실제 디코딩과 세 차례의 픽셀 동등성·전처리
비교를 실행했지만 MediaPipe 설치가 막혀 실제 모델 추론은 실행하지 못했습니다.
버퍼 재사용의 실시간 기본 적용은 맥에서 같은 모델 출력과 성능을 확인한 뒤
결정해야 합니다. 실제 측정과 한계는 `benchmarks/2026-09-06-video-replay.md` 참고.

```bash
uv run pytest
uv run ruff check .
```
