# 아카펠라 자동 채보 (ssonsal-ai)

유튜브 URL(또는 음원 파일)과 성부 구성을 넣으면, **자기 PC에서** 성부별 NoteWorthy Composer 악보 초안
(`.nwctxt`)을 만들어 주는 로컬 앱입니다. 사람은 이 초안을 NWC에서 고쳐 완성합니다.

- 서버·회원가입·구독료 없음. 모든 처리는 동아리원 PC의 CPU에서 끝납니다.
- 성부 하나에 스태프 하나, 화음 없는 단선율 악보를 MIDI를 거치지 않고 nwctxt로 바로 씁니다.
- 음표 전사는 언어와 무관하므로 한국어·영어·일본어·중국어 곡 모두 처리합니다.
- 설계 명세: [docs/design.md](docs/design.md)

![시스템 구성](docs/images/architecture.png)

## 목차

1. [동아리원용: 설치와 사용](#1-동아리원용-설치와-사용)
2. [개발 환경](#2-개발-환경)
3. [명령줄(CLI)](#3-명령줄cli)
4. [파이프라인 구조](#4-파이프라인-구조)
5. [저장소 구조](#5-저장소-구조)
6. [평가](#6-평가)
7. [데이터 구축과 모델 학습 (v2)](#7-데이터-구축과-모델-학습-v2)
8. [설정](#8-설정)
9. [배포](#9-배포)
10. [진행 상황과 확인이 필요한 것](#10-진행-상황과-확인이-필요한-것)
11. [가정](#11-가정)

---

## 1. 동아리원용: 설치와 사용

### 설치 (Windows)

1. 동아리 채널에서 받은 `AcappellaSetup-<버전>.exe`를 실행합니다.
   - "Windows의 PC 보호" 창이 뜨면 **추가 정보 → 실행**을 누릅니다(서명하지 않은 설치 파일이라 나오는 경고입니다).
   - 관리자 권한은 필요 없습니다. `%LOCALAPPDATA%\Programs\Acappella`에 설치됩니다.
2. 바탕화면의 **아카펠라 채보**를 실행하면 잠시 뒤 브라우저에 앱 화면이 열립니다.
   검은 창(콘솔)을 닫으면 앱이 종료됩니다.

### 사용

1. **유튜브 URL**을 붙여 넣거나, **음원 파일**(mp3, wav, m4a)을 올립니다.
2. **성부 프리셋**을 고릅니다: `SATB`, `SSATB`, `Lead+SATB`, `Lead+SSATTB`.
   표에서 성부를 추가·삭제하고 음자리표(`treble` / `treble8vb` / `bass`)와 음역(예: `C4`)을 고칠 수 있습니다.
3. (선택) 곡 제목, 박자표, 템포, 조표를 넣습니다. 비워 두면 자동으로 추정합니다.
   박자가 틀어지면 박자표와 템포를 직접 넣고 다시 돌려 보세요.
4. (선택, v2) **가사**: 리드 성부 가사를 넣으면 리드 스태프에 Lyric1로 들어갑니다.
   한국어는 글자마다, 영어는 `hel-lo`처럼 하이픈으로 음절을 나눌 수 있고, `_`는 음표 하나를 건너뜁니다.
5. **채보 시작** → 진행률이 표시되고, 끝나면 `.nwctxt`(과 검토용 성부별 MIDI)를 내려받습니다.
6. NWC 2.75 이상에서 `.nwctxt`를 열어 고칩니다.

결과 악보는 `%LOCALAPPDATA%\Acappella\scores`에도 남습니다. 내려받은 음원은 처리 후 자동으로 지웁니다.
문제가 생기면 `%LOCALAPPDATA%\Acappella\logs\app.log`를 개발자에게 보내 주세요.

> 동아리 내부용입니다. 유튜브 다운로드는 서비스 약관과 충돌할 수 있으므로 앱을 공개 배포하지 않습니다.

---

## 2. 개발 환경

필요한 것: [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/)(PATH에 있거나 `ACAPPELLA_FFMPEG`로 지정), git.

```bash
git clone https://github.com/latuca/ssonsal-ai.git
```

```bash
cd ssonsal-ai
```

```bash
uv sync
```

```bash
uv run pytest
```

- Python은 **3.10으로 고정**되어 있습니다(uv가 알아서 설치합니다). Basic Pitch 0.4.0은 Windows + Python 3.10에서
  ONNX Runtime 백엔드를 써서 설치본에 TensorFlow가 필요 없기 때문입니다. Linux에서는 tflite-runtime을 씁니다.
- `uv run pytest`는 빠른 테스트만 돌립니다. 합성 샘플 3곡 전체 처리(M2 완료 기준)까지 보려면:

```bash
uv run pytest -m ""
```

- 린트·포맷(CI와 같음):

```bash
uv run ruff check .
```

```bash
uv run ruff format --check .
```

- 실제 동아리 NWC에서 내보낸 nwctxt를 `tests/fixtures/local/`(gitignore)에 넣으면 왕복 테스트가 자동으로 함께 돕니다.

---

## 3. 명령줄(CLI)

```bash
uv run acappella transcribe "https://www.youtube.com/watch?v=..." --voices Lead,S,A,T,B --out scores
```

```bash
uv run acappella transcribe song.mp3 --voices SSATB --tempo 92 --time-sig 4/4 --midi
```

| 옵션 | 설명 |
|---|---|
| `--voices` | 성부 목록(`Lead,S,A,T,B`) 또는 프리셋(`SATB`, `SSATB`, `Lead+SATB`, `Lead+SSATTB`). `이름:음자리표:하한:상한`으로 바꿀 수 있음(예: `T:bass:C3:G4`) |
| `--out` | nwctxt를 저장할 폴더 |
| `--title`, `--tempo`, `--time-sig`, `--key` | 비우면 자동 추정. `--key`는 조표 개수(샵 양수, 플랫 음수) |
| `--lyrics`, `--lyrics-file`, `--lyric-voice` | (v2) 가사와 가사를 넣을 성부 |
| `--voice-method` | `auto`(모델 파일이 있으면 ONNX), `rule`, `onnx` |
| `--midi` | 검토용 성부별 MIDI도 저장 |
| `--keep-audio` | 다운로드·변환한 음원을 지우지 않음 |
| `--no-debug` | 단계별 중간 결과를 저장하지 않음 |

기타 명령:

```bash
uv run acappella ui
```

```bash
uv run acappella update
```

```bash
uv run acappella models --tag v0.2.0
```

`ui`는 로컬 앱 실행, `update`는 yt-dlp 업데이트와 새 버전 확인, `models`는 GitHub Releases에서 성부 배정 모델을 내려받습니다.

---

## 4. 파이프라인 구조

모든 모듈은 [`src/acappella/types.py`](src/acappella/types.py)의 데이터 계약만 주고받고, 입력 수집을 빼면 순수 함수입니다.
단계별 출력은 `<작업 폴더>/outputs/<작업 ID>/`에 저장됩니다(`3_pitch.npz` … `8_score.nwctxt`).

| 단계 | 모듈 | 구현 |
|---|---|---|
| 1. 입력 수집 | `pipeline/ingest.py` | yt-dlp로 오디오만 받거나 파일 입력. ffmpeg로 모노 22.05kHz WAV |
| 2. 전처리 | `pipeline/preprocess.py` | DC 제거, RMS 음량 정규화(-20 dBFS), 피크 제한 |
| 3. 다중 피치 추정 | `pipeline/pitch.py` | Basic Pitch contour(반음당 3빈, 약 11.6ms 프레임)와 노트 후보 |
| 4. 성부 배정 | `pipeline/voices.py` | **v1 규칙 기반**: 프레임별 봉우리 → 배음 감쇠 → 음높이 순위 DP(유니즌 채우기, 음역 벌점) → 성부별 Viterbi(연속성). **v2**: ONNX 모델(전환 가능) |
| 5. 노트 분할 | `pipeline/notes.py` | 메디안 필터, 짧은 구간 흡수, Basic Pitch onset으로 재발음 분할 |
| 6. 박자 추적·양자화 | `pipeline/rhythm.py` | librosa 비트 추적, 템포 옥타브 보정, 3/4·4/4와 강박 추정, 16분음표 격자 |
| 7. 후처리 | `pipeline/postprocess.py` | 짧은 가짜 노트 제거, 유니즌 채우기, 조표 추정(Krumhansl), 음역 경고 |
| 8. 출력 | `nwc/writer.py` | 성부별 스태프, 쉼표 채우기, 마디를 넘는 음은 타이, 조성에 맞는 임시표, Lyric1 |

불변 조건(테스트로 검증): 성부별 단선율, 모든 마디 길이 = 박자표, 음역 밖 음은 로그 경고, 같은 입력·설정이면 같은 출력.

---

## 5. 저장소 구조

```
ssonsal-ai/
├─ README.md
├─ pyproject.toml, uv.lock
├─ docs/design.md            # 설계 명세 사본
├─ src/acappella/            # 앱 코드(배포 대상). PyTorch에 의존하지 않음
│  ├─ types.py               # 데이터 계약
│  ├─ config.py              # 경로, 성부 프리셋, 임계값, 설정 파일
│  ├─ meta.py                # 곡 메타데이터 스키마
│  ├─ score_utils.py         # 단선율·타이·불변 조건
│  ├─ midi_export.py         # 검토용 MIDI
│  ├─ app/ui.py, updates.py  # Gradio 앱, 업데이트·모델 내려받기
│  ├─ pipeline/              # ingest → … → postprocess, lyrics, run(CLI)
│  └─ nwc/                   # writer, reader, pitch_pos
├─ tools/                    # 개발자 전용 데이터 구축
├─ training/                 # 실습실 GPU 전용 학습
├─ eval/                     # 평가 지표, 리포트, 처리 시간 측정
├─ tests/                    # pytest (fixtures/에는 저작권 문제가 없는 예시만)
└─ packaging/                # Windows 설치본 스크립트
```

`data/`, `models/`, `outputs/`, `checkpoints/`, 음원, `.nwc`, 가중치는 `.gitignore`에 있어 레포에 올라가지 않습니다.

---

## 6. 평가

```bash
uv run python -m eval.run_eval --meta-dir data/meta/songs --split test
```

gold 곡(tier C)을 돌려 `outputs/eval/<시각>/report.md`와 `report.json`을 만듭니다.

| 지표 | 보는 것 |
|---|---|
| 노트 F1 (onset 50ms) | 음표 단위 정확도 (mir_eval transcription) |
| 성부별 피치 정확도 | 성부 배정 품질, 성부 간 혼동 (mir_eval melody) |
| 프레임 다중 피치 정확도 | 피치 추정 품질 (mir_eval multipitch) |
| 리듬 일치율 | 양자화 후 박 위치·길이 일치 (자체 스크립트, 전체 이동 보정) |
| 곡당 처리 시간 | CPU 추론 실용성 |

성부 수(4 / 5~6 / 7+)와 장르(메타데이터 `genre`)별로 따로 보고합니다.
수정 시간·수정 음표 수(최종 판단 기준)는 동아리원 비교 실험으로 따로 기록합니다.

곡당 처리 시간은 동아리원 최저 사양 PC에서 다음으로 잽니다. 이 개발용 클라우드 PC(4코어)에서는 3분 합성 5성부 곡이 약 8초 걸렸습니다.

```bash
uv run python -m eval.benchmark --voices Lead+SATB
```

---

## 7. 데이터 구축과 모델 학습 (v2)

데이터는 개발자 PC와 실습실 PC에만 둡니다. 순서:

| 단계 | 명령 | 결과 |
|---|---|---|
| 아카이브 스캔 | `uv run python -m tools.scan_archive <아카이브 폴더>` | `data/meta/archive_index.csv`, `archive_summary.csv` |
| nwctxt 추출 | `uv run python -m tools.extract_nwctxt <아카이브 폴더>` | `data/nwctxt/sXXXX.nwctxt`, `id_map.csv`, `extract_failures.csv` |
| 메타데이터·분할 | `uv run python -m tools.build_dataset --gold data/meta/gold.csv` | `data/meta/songs/*.json` (곡·그룹 단위 train/valid/test) |
| gold 정렬 | `uv run python -m tools.align --nwctxt … --audio … --out data/align/sXXXX.csv` | 박→초 대응표, 조옮김 |
| 임계값 보정 | `uv run python -m tools.match_youtube --calibrate` | gold 곡 정렬 비용 분포 |
| 층 B 자동 매칭 | `uv run python -m tools.match_youtube --auto-threshold … --review-threshold …` | tier B 메타데이터, `review_queue.csv` |
| 층 A 합성 | `uv run python -m training.synth --variants 4` | `data/synth/{train,valid}/*.npz` |

`gold.csv`는 사람이 링크·악보 일치·시간 정렬을 확인한 약 30곡(`song_id,youtube_url,group,genre`)입니다.
4성부와 5~6성부, 장르를 고르게 섞고 메들리는 넣지 않습니다.

학습(실습실 RTX 4090). PyTorch는 앱 의존성이 아니라 `training` 그룹에만 있습니다.
CUDA판 PyTorch는 [PyTorch 안내](https://pytorch.org/get-started/locally/)에 맞춰 설치합니다.

```bash
uv sync --group training
```

```bash
uv run python -m training.train --train data/synth/train --valid data/synth/valid --out checkpoints/pretrain --resume --backup-dir <외부 백업 폴더>
```

```bash
uv run python -m training.train --train data/real/train --valid data/real/valid --out checkpoints/finetune --init checkpoints/pretrain/best.pt --lr 1e-4 --freeze-front --resume
```

```bash
uv run python -m training.export_onnx --ckpt checkpoints/finetune/best.pt --out models/voice_assign.onnx
```

- 실습실 PC가 꺼져도 같은 명령에 `--resume`을 붙여 다시 실행하면 마지막 체크포인트부터 이어집니다.
- train만 가중치를 갱신하고, valid는 조기 종료·체크포인트 선택에만 씁니다. test는 마지막에 한 번만 잽니다.
- 층 B(실제 음원) 학습 데이터는 synth.py와 같은 형식(`salience`, `target_midi`, `ranges`)의 npz로,
  정렬 CSV로 악보 박을 초로 옮겨 만듭니다:
  `uv run python -m training.real_data --split train` (valid도 같은 방법, test는 만들지 않음).
- 배포 기준: `eval.run_eval --voice-method onnx`의 test 성능과 CPU 처리 시간이 규칙 기반보다 나을 때만 Releases에 올립니다.

---

## 8. 설정

우선순위: 환경 변수 > `<작업 폴더>/settings.json` > 기본값.

| 환경 변수 | 설명 |
|---|---|
| `ACAPPELLA_HOME` | 작업 폴더(outputs, scores, models, logs). 기본 Windows `%LOCALAPPDATA%\Acappella`, 그 외 `~/.acappella` |
| `ACAPPELLA_FFMPEG` | ffmpeg 실행 파일 경로(설치본은 실행기가 지정) |
| `ACAPPELLA_YTDLP` | yt-dlp 실행 파일 경로. 비우면 `python -m yt_dlp` |
| `ACAPPELLA_GITHUB_TOKEN` | 비공개 레포 Releases 조회용 읽기 전용 토큰(선택) |

`settings.json` 예:

```json
{
  "keep_audio": false,
  "auto_update_ytdlp": true,
  "release_repo": "latuca/ssonsal-ai",
  "thresholds": {"peak_abs_threshold": 0.25, "unison_fill_max_beats": 1.0}
}
```

임계값 전체 목록은 [`config.py`의 `Thresholds`](src/acappella/config.py)에 있습니다.

---

## 9. 배포

- **설치본**: 임베디드 Python + pip 방식(PyInstaller와 비교한 결과는 [packaging/README.md](packaging/README.md)).
  `packaging/build_windows.ps1`이 Python 3.10 임베디드, 앱, ffmpeg, yt-dlp, (있으면) ONNX 모델을 묶고
  Inno Setup으로 `AcappellaSetup-<버전>.exe`를 만듭니다. PyTorch·TensorFlow가 끼어들면 빌드가 실패합니다.
- **릴리스**: `pyproject.toml`과 `src/acappella/__init__.py`의 버전을 올리고 `v<버전>` 태그를 푸시하면
  `.github/workflows/release.yml`이 설치본을 만들어 GitHub Releases에 올립니다. 모델 파일(`voice_assign.onnx`)도 같은 릴리스에 올립니다.
- **업데이트**: 앱은 실행할 때 yt-dlp 업데이트를 시도하고(실패해도 파일 입력은 동작), 새 버전이 있으면 화면 위에 알립니다.

---

## 10. 진행 상황과 확인이 필요한 것

| 마일스톤 | 상태 |
|---|---|
| M0 레포·개발 환경 | 완료: pyproject/uv, .gitignore, CI(Ubuntu·Windows, ruff·pytest) |
| M1 NWC 읽기·쓰기 | 구현·테스트 완료. **실제 NWC 내보내기 파일로 왕복 검증과 NWC 화면 확인 필요** |
| M2 베이스라인 CLI | 완료: 합성 샘플 3곡 처리·불변 조건 통과 |
| M3 로컬 앱·설치본 | 앱 완료(브라우저 동작 확인). **설치 스크립트는 Windows에서 아직 실행해 보지 않음** |
| M4 평가 도구 | 완료. **gold 곡과 최저 사양 PC 측정 필요** |
| M5 데이터 구축 | 도구 완료. **실제 아카이브로 실행 필요(.nwc 내장 nwctxt 추출 가능 여부 확인)** |
| M6 성부 배정 모델 | 학습·재개·내보내기·앱 탑재 완료(작은 규모로 검증). **실습실 학습 필요** |
| M7 가사 배치 | 완료: 한국어·영어 음절 분할, 리드 스태프 Lyric1. **NWC에서 정렬 확인 필요** |

설계 문서의 열린 질문(동아리원 PC OS·최저 사양, NWC 버전, 처리 시간 목표, 성공 기준 단축 비율, 아카이브 구성,
공개 데이터 사용 조건)은 아직 답이 필요합니다.

**nwctxt 문법**: writer/reader는 설계 문서의 예시와 공개된 nwctxt 문법을 따랐지만, 실제 동아리 NWC에서 내보낸
파일로는 아직 검증하지 않았습니다. NWC에서 내보낸 예시 2~3개(조표·임시표·타이·점음표·가사가 있는 곡)를
`tests/fixtures/local/`에 넣고 `uv run pytest tests/test_nwc_roundtrip.py`를 돌려 주세요. 특히 다음을 확인해야 합니다.
조표 문법 `Signature:F#,C#`, 옥타브 아래 높은음자리표 `OctaveShift:Octave Down`, 임시표 접두어(`#`, `b`, `n`, `x`, `v`)의
마디 안 유지 규칙, 가사 `|Lyrics|…`/`|Lyric1|Text:"…"` 위치.

---

## 11. 가정

명세에 없어서 정하고 진행한 사소한 결정입니다. 사용자 경험이나 데이터 포맷에 영향을 주면 바꿀 수 있습니다.

- `QNote`의 "박"은 박자표와 무관하게 4분음표 하나입니다(6/8 한 마디 = 3박).
- 비트 추적기가 찾은 박을 4분음표로 봅니다. 추정 템포가 65 BPM 미만이면 두 배, 170 BPM 초과면 절반으로 고칩니다.
  6/8처럼 점4분음표가 박인 곡은 박자표와 템포를 직접 넣는 것이 안전합니다.
- 자동 박자표는 3/4와 4/4 중에서 고르고, 강세가 뚜렷하지 않으면 첫 음을 강박에 둡니다.
- 템포 표시는 첫 스태프에만, 조표·박자표는 모든 스태프의 첫 마디에 넣습니다. 마지막 마디 뒤에는 `|Bar`를 쓰지 않습니다(NWC가 끝 마디선을 그림).
- 음가는 박 경계를 지키도록 나눕니다. 박 중간에서 시작한 음은 다음 박을 넘지 않고, 넘으면 타이로 나눕니다.
- 같은 시작점의 음이 겹치면 긴 음을, 단조는 나란한조(장조) 조표를 씁니다.
- reader는 화음에서 가장 높은 음만 남기고, 도돌이표를 펼치지 않습니다(평가·데이터 구축용).
- nwctxt는 UTF-8, CRLF 줄바꿈으로 씁니다.
- 결과 악보는 앱에서 `<작업 폴더>/scores`에 저장합니다. 단계별 중간 결과는 `outputs/<작업 ID>/`에 남고 음원은 지웁니다.
- 비공개 레포의 Releases는 인증 없이 조회되지 않습니다. 새 버전 알림과 모델 내려받기를 쓰려면 토큰을 설정하거나
  공개 배포용 레포를 `release_repo`로 지정해야 합니다.
- 메타데이터의 `genre`, `audio_path`는 스키마 밖의 선택 필드로 다룹니다(`extra`).

## 라이선스

[MIT](LICENSE)
