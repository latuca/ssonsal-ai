# Windows 설치본

동아리원에게는 설치형 로컬 앱을 배포한다. 설치본에는 Python 런타임, 앱 코드, ffmpeg, yt-dlp,
(있으면) ONNX 모델 파일이 들어가고 **PyTorch와 TensorFlow는 들어가지 않는다**
(Basic Pitch는 Windows + Python 3.10에서 ONNX Runtime 백엔드를 쓴다).

## 도구 선택: 임베디드 Python (PyInstaller와 비교)

| 기준 | PyInstaller | 임베디드 Python + pip (채택) |
|---|---|---|
| Basic Pitch·librosa·numba·gradio 호환 | 숨은 import와 데이터 파일(모델, gradio 프런트엔드)을 spec에 일일이 적어야 하고 버전마다 깨지기 쉽다 | 일반 `pip install` 그대로라 개발 환경과 같다 |
| yt-dlp 자동 업데이트 | 묶인 실행 파일 안이라 갱신 불가, yt-dlp.exe를 따로 둬야 함 | `python -m pip install -U yt-dlp`로 바로 갱신 |
| 백신 오탐 | 부트로더 exe가 자주 오탐 | 공식 python.exe라 상대적으로 적음 |
| 크기 | 비슷함(약 400~600MB, 대부분 의존성) | 비슷함 |
| 빌드 시간 | 김 | 짧음(휠 설치) |

M3에서는 임베디드 Python 방식으로 정했다. 실제 동아리원 PC 설치 시험(완료 기준) 결과에 따라 다시 볼 수 있다.

## 빌드 (개발자 Windows PC)

준비물: PowerShell 5+, 인터넷, [Inno Setup 6](https://jrsoftware.org/isinfo.php)(`iscc`가 PATH에 있거나 기본 경로에 설치),
[uv](https://docs.astral.sh/uv/).

```powershell
cd packaging
.\build_windows.ps1                      # ffmpeg를 자동으로 내려받는다
.\build_windows.ps1 -FfmpegExe C:\tools\ffmpeg.exe -ModelFile ..\models\voice_assign.onnx
```

결과: `packaging/dist/AcappellaSetup-<버전>.exe`

- 설치 위치: `%LOCALAPPDATA%\Programs\Acappella` (관리자 권한 불필요)
- 작업 폴더(결과 악보, 로그, 모델): `%LOCALAPPDATA%\Acappella`
- 바탕화면·시작 메뉴 바로가기 "아카펠라 채보"

태그(`v*`)를 푸시하면 `.github/workflows/release.yml`이 같은 스크립트로 설치본을 만들어 GitHub Releases에 올린다.

## 서명 없는 설치 파일 경고

코드 서명 인증서가 없어서 처음 실행할 때 Windows SmartScreen이 "Windows의 PC 보호" 경고를 띄운다.
동아리원 안내: **추가 정보 → 실행**. 설치 파일은 동아리 내부 채널로만 배포한다.
