# PokerStars 플레이머니 적용 셋업 가이드 (한국 거주자용)

> **목적**: 한국 거주자는 PokerStars 캐시게임 진행이 불가하므로 **플레이머니(플레이 칩) 테이블**에서 연구·실험 목적으로 봇을 적용한다.
>
> **브랜치**: `play-money-kr` (이 브랜치에서 작업)
>
> **전제**: 본 작업은 학술 연구·알고리즘 검증 목적이며 PokerStars 약관(자동화 도구 금지)을 인지하고 진행한다. 실머니 적용은 금지.
>
> **클라이언트 다운로드**: 반드시 **[pokerstars.net](https://www.pokerstars.net/)** (플레이머니 전용 도메인) 에서 받을 것. `.com`은 한국 IP 차단 + 실머니 모드라 봇 적용 시 도박죄 리스크.

---

## 0. 사전 점검 (외부 PC, 인터넷만)

이 저장소를 집 PC로 가져갈 수 있는지 확인.

```powershell
git clone https://github.com/BudongJW/Poker.git
cd Poker
git checkout play-money-kr
```

---

## 1. 환경 설치 (집 PC)

### 1-1. 필수 구성요소

| 구성요소 | 버전·링크 | 비고 |
|---|---|---|
| Windows 10 x64 이상 | — | PokerStars 자체 요구사항 |
| Python | **3.11.x 권장** | tesserocr cp311 휠 호환 |
| Anaconda or Miniconda | https://www.anaconda.com/products/distribution | 가상환경 격리용 |
| Tesseract OCR | Windows 빌드 5.3.1 | 카드 숫자 인식 |
| tesserocr | https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.6.0-tesseract-5.3.1/tesserocr-2.6.0-cp311-cp311-win_amd64.whl | 휠 직접 설치 |
| VS C++ Redistributable | https://visualstudio.microsoft.com/downloads/ | 일부 휠 빌드 의존 |
| VirtualBox 7.0.12+ | https://www.virtualbox.org/ | (선택) 호스트 마우스 분리용 |
| PokerStars 클라이언트 | https://www.pokerstars.com/ | 최신 버전 |

### 1-2. 가상환경 생성 & 패키지 설치

```powershell
conda create -n pokerbot python=3.11 -y
conda activate pokerbot

cd Poker
pip install -r requirements_win.txt

# tesserocr는 휠 직접 설치
pip install tesserocr-2.6.0-cp311-cp311-win_amd64.whl
```

### 1-3. 환경 검증

```powershell
python scripts/check_env.py
```

모든 항목이 `OK`면 다음 단계 진행.

---

## 2. PokerStars 클라이언트 셋업 (플레이머니용)

### 2-1. 클라이언트 설정 고정

봇은 이미지 인식 기반이므로 **클라이언트 시각 요소가 매번 같아야** 한다.

- **언어**: English (한국어 버튼 라벨은 매핑 시 인식 불가)
- **테마**: Classic 또는 Hyper Simple (기본)
- **카드 색상**: **4-Color Deck** 활성화
- **덮인 카드 스타일**: 기본
- **테이블 스타일**: 기본
- **DPI 스케일**: 100% (Windows 디스플레이 설정 → 배율 100%)
- **테이블 창 크기**: 기본 (`Table Options → Back to Default Size`)

### 2-2. 플레이머니 테이블 입장

- 로비 → **Play Money** 탭
- Hold'em → 6-Max → Zoom 또는 일반 캐시
- 권장: **Zoom 6-Max Play Money**, 평균 100~500 플레이 칩 스택 테이블

### 2-3. 플레이머니가 실머니와 다른 점 (매핑 시 주의)

| 요소 | 실머니 | 플레이머니 |
|---|---|---|
| 통화 표시 | `$0.01/$0.02` | `100/200` (칩) |
| 폰트 크기 | 작음 (소액 표기) | 큼 |
| 베팅 단위 | 센트 단위 | 정수 칩 |
| 테이블 컬러 | 사이트별 표준 | 동일 (보통) |
| 버튼 배치 | 동일 | 동일 |
| OCR 난이도 | 소수점 인식 필요 | **정수만 → 더 쉬움** |

→ 플레이머니가 OCR 인식은 더 쉬운 편이나, **숫자 자릿수가 길어** (예: `1,250,000`) 베팅 칸 너비가 다를 수 있으므로 매핑 단계에서 베팅 영역(call/raise 값 영역)을 충분히 넓게 잡을 것.

---

## 3. 새 테이블 매핑 (플레이머니 전용 템플릿)

기본 `Official Poker Stars` 템플릿은 실머니 기준이므로 그대로 못 쓴다. 본 브랜치는 [`poker/config_default.ini`](poker/config_default.ini)에서 `table_scraper_name = Pokerstars Play Money KR`로 미리 지정해두었다.

### 3-1. 매핑 GUI 실행

```powershell
python poker/main.py
```

상단 메뉴 → **Table setup** → **Blank new**.

템플릿 이름: `Pokerstars Play Money KR` (`config_default.ini`에 적힌 이름과 **정확히 일치**해야 봇이 자동으로 로드).

### 3-2. 매핑 순서 (체크리스트)

PokerStars 플레이머니 Zoom 테이블을 열어두고, 봇 셋업 창을 옆에 두고 진행:

- [ ] **Take screenshot** — 테이블 스크린샷 캡처
- [ ] **Top left corner** — 테이블 창 좌상단 모서리 영역 지정 → 저장
- [ ] **Crop from top left corner** — 잘라내기
- [ ] **Buttons search area** — Fold/Check/Call/Bet 버튼 영역
- [ ] **Card values (2~A)** — 카드 숫자 13장 각각 등록 (4-color deck 기준)
- [ ] **Card suits (♠♥♦♣)** — 무늬 4종 등록
- [ ] **Dealer button** — 딜러 버튼 이미지
- [ ] **Player names search area** — 6개 좌석별
- [ ] **Player funds search area** — 6개 좌석별 스택 표시 영역
- [ ] **Pot search area** — 팟 금액 영역
- [ ] **Call/Bet value areas** — 베팅 값 영역 (**플레이머니 숫자 자릿수 고려해서 넓게**)
- [ ] **Mouse positions** — Fold/Call/Check/Bet/Raise/All-in/Half-pot/Full-pot/Increase 버튼 클릭 좌표
- [ ] **Save** — MongoDB 또는 로컬 저장

### 3-3. 검증

- 봇 셋업 창에서 **Test scraper** 또는 **Run test**로 한 핸드 인식 시도
- 카드·팟·스택이 정확히 읽히는지 로그(`poker/log/`)와 스크린샷(`poker/log/screenshots/`)으로 확인
- 실패한 항목만 재매핑

---

## 4. 전략 선택

- 기본 전략 `Trial 1`은 실머니 0.01/0.02 NL 기준으로 튜닝됨
- 플레이머니에서는 **상대가 훨씬 루즈**하므로 다음 조정 권장:
  - Preflop equity threshold: +3~5%p 상향 (루즈한 콜에 대응)
  - Bluff frequency: 하향 (블러프가 안 먹힘)
  - Bet sizing: pot 대비 비율을 약간 키움
- 전략 편집은 **풀 버전** 구독 필요. 또는 코드에서 `poker/decisionmaker/` 의 XML 파라미터 직접 수정 가능

---

## 5. 첫 실행 체크리스트

봇을 실제로 돌리기 전:

- [ ] `python scripts/check_env.py` 전부 OK
- [ ] PokerStars 플레이머니 6-max Zoom 테이블 정상 입장 가능
- [ ] 새 템플릿 매핑 완료 (`Pokerstars Play Money KR`)
- [ ] Test scraper에서 카드·팟·스택 정확 인식
- [ ] (선택) VirtualBox에 PokerStars 격리 설치
- [ ] 시작 자본: 최소 바이인의 **2배 이상** (예: 100/200 테이블이면 400+ 칩)
- [ ] 로그·스크린샷 경로 쓰기 권한 확인 (`poker/log/`)

---

## 6. 운영 시 주의사항

- **연속 8시간 이상 가동 금지** — 플레이머니라도 PokerStars Game Integrity AI가 행동 패턴 감지
- **마우스 패턴 변동성 부족** — `poker/tools/mouse_mover.py`의 jitter는 균등분포 단순 형태. 가능하면 향후 개선 (베지어 곡선·로그정규 sleep 등)
- **계정 1개 전용** — 여러 계정에서 동시 가동 시 즉시 탐지
- **로그 보존** — `poker/log/` 폴더를 별도 보관해 핸드 히스토리·전략 분석에 활용
- **실머니 모드로 전환 금지** — 클라이언트 우상단 토글 실수 주의. 본 템플릿은 플레이머니 시각 요소 기준이라 실머니에서는 어차피 잘못 동작함

---

## 7. 트러블슈팅 빠른 참조

| 증상 | 원인 후보 | 해결 |
|---|---|---|
| 좌상단 모서리 못 찾음 | DPI 스케일, 클라이언트 비표준 색상 | 디스플레이 100%, 카드 4-color 재설정 |
| 카드 인식 실패 | 4-color deck 미적용, 매핑 누락 | 클라이언트 카드 스타일 재확인 후 13장 재매핑 |
| 팟 숫자 OCR 실패 | 플레이머니 자릿수가 매핑 영역 초과 | Pot search area 폭 확장 |
| 버튼 클릭 안 됨 | DPI 변경, 테이블 크기 변경 | Default Size로 복귀 |
| MongoDB 연결 실패 | deepermind 서버 다운 또는 방화벽 | 로컬 저장 모드로 전환 (templates를 JSON으로 export) |

---

## 7-A. 2026 PokerStars UI 변화 사전 대응

원본 봇은 2023~2024년 1월 PokerStars 클라이언트 기준으로 만들어졌다. 2025~2026 사이 다음과 같은 변화가 있었으므로, 매핑 단계에서 의식하고 진행할 것.

### 7-A.1 코드 패치로 사전 대응한 부분

| 항목 | 변경 | 위치 |
|---|---|---|
| **OCR 콤마/접미사 처리** | 플레이머니 `1,250,000`, `1.5K`, `10M` 등 파싱 가능 | [`poker/tools/text_normalize.py`](poker/tools/text_normalize.py) (신규), [`poker/tools/screen_operations.py`](poker/tools/screen_operations.py) 의 `get_ocr_number` |
| **통화 기호 확장** | `$£B` → `$£€₩¥BP` | 위와 동일 |
| **템플릿 매칭 threshold 상수화** | 매핑 후 조정 가능 | `DEFAULT_TEMPLATE_MATCH_THRESHOLD = 0.01` |

검증: `python scripts/smoke_test_extended.py` 의 OCR normalize 테스트 5종 통과 확인.

### 7-A.2 클라이언트 설정으로 대응 (매핑 전 필수)

| 2025/2026 변화 | 대응 |
|---|---|
| **Cash Game Lobby Seatfinder** (2025-07) — 자동 시팅으로 봇의 수동 테이블 선택 흐름 우회됨 | 로비에서 Seatfinder 기능을 **꺼둘 것**. 또는 봇 시작 시 이미 테이블에 앉은 상태로 가정하고 매핑 |
| **4K 테이블 그래픽 재설계** — 카드·버튼 이미지가 미세하게 다름 | 모든 카드·버튼 템플릿을 **현재 클라이언트 기준으로 새로 캡처**. 구버전 템플릿(MongoDB Official 시리즈) 그대로 쓰지 말 것 |
| **태블릿 최적화 레이아웃** | 데스크탑 모드 강제, DPI 100%, 테이블 크기 Default |
| **신규 폰트 (가독성 개선)** | OCR 인식 실패 시 `prepareImage`의 binarize threshold(76, 125)를 조정해볼 것 |
| **HUD/서드파티 사전 등록 의무화** | 봇은 HUD가 아니므로 무관. 다만 보안 모니터링은 강화됨을 인지 |

### 7-A.3 매핑 후 인식 실패 시 점검 순서

1. **카드 인식 실패**: 4-color deck 활성화 재확인 → 13장 전부 재매핑
2. **숫자 OCR 실패**: 매핑 영역 폭을 늘려보고 (긴 자릿수 대응), 그래도 안 되면 `screen_operations.py`의 `DEFAULT_TEMPLATE_MATCH_THRESHOLD`를 0.05~0.1로 완화
3. **좌상단 모서리 못 잡음**: 클라이언트 윈도우 데코레이션이 OS 테마 영향 받음. Windows 11 라이트 테마 권장
4. **버튼 위치 오프셋**: 4K 재설계로 버튼이 미세하게 이동. 매핑 GUI에서 `mouse_*` 좌표 재지정

---

## 8. 다음 작업 (집에서 할 일)

1. `git pull origin play-money-kr`
2. `conda env` 구성 & `python scripts/check_env.py` 통과
3. PokerStars 클라이언트 설정 정리 (위 2-1)
4. 플레이머니 6-max Zoom 테이블 입장
5. **본 가이드 3장**의 매핑 체크리스트 수행 → 새 템플릿 저장
6. Test scraper로 인식 검증
7. 한 핸드만 수동 모드로 봇 동작 확인
8. (선택) `Trial 1` 전략 파라미터 플레이머니에 맞게 조정
9. 짧은 세션(30분~1시간)으로 안정성 확인
10. 데이터 수집·분석 단계로 진입

---

## 부록 A. 관련 파일

- [readme.rst](readme.rst) — 원본 영문 README
- [readme.ko.rst](readme.ko.rst) — 한국어 번역 README
- [poker/config_default.ini](poker/config_default.ini) — 템플릿 이름 지정
- [scripts/check_env.py](scripts/check_env.py) — 환경 검증
- [poker/scraper/](poker/scraper/) — 스크래퍼 모듈
- [poker/tools/mouse_mover.py](poker/tools/mouse_mover.py) — 마우스 자동화

## 부록 B. 연구 목적 보조 자료

알고리즘 비교·논문 작성용으로 함께 활용 가능한 외부 자원:

- [dickreuter/neuron_poker](https://github.com/dickreuter/neuron_poker) — OpenAI Gym 환경, RL 베이스라인 (2025-08 정비됨)
- PyPokerEngine — 활발히 유지보수 중인 Python 포커 엔진
- PokerRL / Texas Solver — CFR 기반 솔버
