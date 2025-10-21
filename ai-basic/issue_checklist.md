# AI-Basic 프로젝트 이슈 체크리스트

## 사내망 환경에서 발생하는 이슈들

### 1. 파이썬 가상환경 활성화 스크립트 실행 오류

**문제**: 가상환경 활성화 스크립트 실행 시 권한 오류 발생

**해결방법**:
1. 관리자 권한으로 PowerShell 실행
2. 다음 명령어 수행:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. 가상환경 활성화 재시도

---

### 2. requirements.txt 한글 유니코드 에러 (cp949 인코딩 문제)

**문제**: requirements.txt 파일에 한글이 포함되어 cp949 인코딩 에러 발생

**해결방법**:
1. requirements.txt 파일을 UTF-8 인코딩으로 열기
2. 한글이 포함된 주석이나 내용 제거
3. 영문과 숫자, 특수문자만 남기고 UTF-8로 저장
4. 다시 pip install 실행

**추가 해결방법**:
```bash
# 파일 인코딩을 UTF-8로 변환
pip install -r requirements.txt --encoding utf-8
```

---

### 3. 사내 DRM으로 인한 파일 접근 제한

**문제**: 사내 DRM 시스템으로 인해 requirements.txt 파일 접근이 제한됨

**해결방법**:
1. requirements.txt 파일 내용 복사
2. 확장자 없는 새 파일 생성 (예: requirements_copy)
3. 복사한 내용을 새 파일에 붙여넣기
4. 다음 명령어로 설치:
   ```bash
   pip install -r 'requirements_copy'
   ```

---

### 4. pip install 명령어 실행 불가

**문제**: pip install 실행 시 SSL 인증서 오류 또는 네트워크 접근 불가

**해결방법 A - SSL 인증서 교체**:
1. 크롬/엣지 등 웹브라우저에서 사내 인증서(.crt 파일) 다운로드
2. 다운로드한 인증서 파일을 시스템에 설치/교체
3. pip install 재시도

**해결방법 B - 명령어 옵션 추가**:
```bash
# SSL 인증서 검증 비활성화 (외부망 조건적 허용 환경)
pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

# 또는 개별 패키지 설치 시
pip install 패키지명 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
```

**참고**: 방법 B는 보안상 권장되지 않지만 임시 해결책으로 사용 가능

---

## 추가 참고사항

- 모든 명령어는 프로젝트 루트 디렉토리에서 실행
- 가상환경 활성화 후 패키지 설치 권장