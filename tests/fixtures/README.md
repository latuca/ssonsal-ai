# 테스트 픽스처

- `handmade_satb.nwctxt`: 개발 중 손으로 쓴 저작권 없는 예시. **실제 NWC에서 내보낸 파일이 아니다.**
  문법 확인용이 아니라 reader/writer 동작 확인용이다.
- `local/` (gitignore): 동아리 NWC에서 내보낸 실제 nwctxt를 넣는 곳. 외부 공유 금지 파일은 여기에만 둔다.
  `tests/test_nwc_roundtrip.py`가 이 폴더의 모든 `*.nwctxt`로 왕복 테스트를 돌린다(없으면 건너뜀).
