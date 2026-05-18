PokerStars, PartyPoker, GGPoker용 DeeperMind 포커봇 (BudongJW/Poker 포크)
=========================================================================

이 포커봇은 PokerStars, PartyPoker, GG Poker에서 자동으로 플레이한다. 다른 테이블도 매핑하여 추가할 수 있다.
이미지 인식, 몬테카를로 시뮬레이션, 기본 유전 알고리즘으로 동작한다.
마우스는 자동으로 움직이며, 다수의 파라미터에 기반해 장시간 플레이 가능하다.

바이너리를 다운로드해 실행 파일을 바로 실행할 수도 있다 (업스트림):
http://www.deepermind-pokerbot.com

----

한국어 사용자 빠른 시작 (BudongJW 포크)
----------------------------------------

.. warning::

   본 포크는 **학술 연구 + 플레이머니 검증 전용**. PokerStars 약관상 자동화 도구는
   플레이머니에서도 금지되어 있고, RTA 탐지율은 95%를 넘는다 (영구 정지 + 잔액 몰수).
   **실머니(pokerstars.com) 적용은 한국에서 도박죄 성립 가능**. 절대 금지.

본 포크가 업스트림(dickreuter/Poker)에 추가한 것
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **플레이머니 OCR**: ``1,250,000`` / ``1.5K`` / ``10M`` 등 대형 정수·접미사 파싱 (``poker/tools/text_normalize.py``)
* **베지에 + Fitts 마우스 모델**: HCI 표준 적용 (``poker/tools/mouse_mover.py`` 재작성)
* **numpy 2.x 호환성 수정**: 기존 ``int(np.random.uniform(0, 500, 1))`` TypeError 패치
* **한국어 셋업 가이드**: ``SETUP_PLAY_MONEY.ko.md`` (집 PC 매핑까지 단계별)
* **코어 알고리즘 스모크 테스트**: PokerStars 미설치 환경에서 봇 두뇌 무결성 검증 (``scripts/smoke_test.py`` 5/5, ``scripts/smoke_test_extended.py`` 15/15)
* **환경 검증 스크립트**: Python/패키지/Tesseract/해상도/MongoDB 자동 점검 (``scripts/check_env.py``)
* **2026 PokerStars UI 변화 대응 문서**: Aurora 강제, Seatfinder, 4K 그래픽, Throwables 등 (``SETUP_PLAY_MONEY.ko.md §7-A``)

브랜치 구조
~~~~~~~~~~~

* ``master`` — 업스트림 미러
* ``play-money-kr`` — 한국 거주자용 플레이머니 작업 브랜치 (**여기서 작업**)

1단계: 외부 PC (코드 검증, PokerStars 미설치)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PokerStars 없이도 봇의 **수학·이미지·마우스 로직**을 격리 검증할 수 있다.

.. code-block:: bash

   git clone https://github.com/BudongJW/Poker.git
   cd Poker
   git checkout play-money-kr

   # Python 3.12 으로 충분 (코어 로직 + OCR 정규화 + 베지에 마우스)
   py -3.12 -m venv .venv

   # 최소 의존성 (TF / tesserocr 제외)
   .venv/Scripts/python.exe -m pip install -U pip
   .venv/Scripts/python.exe -m pip install "numpy<2" pandas requests pytest
   .venv/Scripts/python.exe -m pip install opencv-python Pillow PyQt6

   # 코어 알고리즘 검증 (5/5 통과해야)
   .venv/Scripts/python.exe scripts/smoke_test.py

   # 마우스 + 이미지 + OCR + Qt 확장 검증 (15/15 통과해야)
   .venv/Scripts/python.exe scripts/smoke_test_extended.py

기대 결과:

.. code-block:: text

   [OK] hand evaluator
   [OK] outs calculator (gutshot, made hand)
   [OK] montecarlo equity (AA vs random ~0.85)
   [OK] mouse path Bezier curvature
   [OK] mouse velocity profile (smoothstep)
   [OK] mouse sleep distribution (log-normal ~22ms)
   [OK] OCR normalize (play money 1,250,000 / 1.5K / 10M)
   ...

여기서 실패가 나면 ``SETUP_PLAY_MONEY.ko.md §1`` 의존성 설치 부분 재점검.

2단계: 집 PC (PokerStars.net 설치 환경)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**중요**: 반드시 ``pokerstars.net`` (플레이머니 전용) 에서 클라이언트 다운로드. ``.com`` 은
한국 IP 차단 + 실머니 모드라 도박죄 리스크.

.. code-block:: bash

   git clone https://github.com/BudongJW/Poker.git
   cd Poker
   git checkout play-money-kr

   # Python 3.11 권장 (tesserocr cp311 휠 + TF 2.12 호환성)
   conda create -n pokerbot python=3.11 -y
   conda activate pokerbot

   pip install -r requirements_win.txt

   # tesserocr 는 별도 휠 (Windows)
   # https://github.com/simonflueckiger/tesserocr-windows_build/releases
   pip install tesserocr-2.6.0-cp311-cp311-win_amd64.whl

   # 1. 환경 자동 검증
   python scripts/check_env.py

   # 2. 코어 회귀 확인 (외부 PC 와 같은 결과)
   python scripts/smoke_test.py
   python scripts/smoke_test_extended.py

   # 3. 봇 GUI 실행 → Table Setup 메뉴
   python poker/main.py

매핑 워크플로우 상세 (카드 13장 + 무늬 4종 + 버튼 + OCR 영역):
SETUP_PLAY_MONEY.ko.md §3 의 체크리스트 참조.

3단계: PokerStars.net 클라이언트 설정 (매핑 전 필수)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

매핑 정확도와 OCR 인식률을 좌우하는 클라이언트 측 설정:

.. code-block:: text

   Settings → Language: English
   Settings → Cards: 4-Color Deck
   Settings → Table Appearance:
     - Theme: Classic (Saloon/Black 같은 어두운 테마 회피)
     - Aurora Graphics Quality: Low
     - Aurora Ambient Animations: Disabled  ← 매우 중요
     - Throwables: Disabled
     - Card animations: Minimal
   Settings → Table Options:
     - Auto-Center buttons: OFF
     - Big card values: ON
   Display (Windows):
     - DPI scaling: 100%
     - Resolution: 1920x1080 이상
   Lobby:
     - Seatfinder: Disabled (2025-07 도입된 자동 시팅, 봇 흐름과 충돌)

4단계: 매핑 → 첫 한 핸드 통합 테스트
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. PokerStars.net Play Money → Hold'em → 6-Max → Zoom 테이블 입장
2. 봇 GUI 의 Table Setup → ``Blank new`` → 이름: ``Pokerstars Play Money KR``
   (이미 ``poker/config_default.ini`` 에 등록되어 있어 봇이 자동 로드)
3. Take screenshot → Top left corner → Crop → Buttons search area 순차 매핑
4. 카드 13장 (2~A) × 무늬 4종 (CDHS) 매핑
5. Dealer/Pot/Stack/Call/Bet 영역 마킹 (플레이머니 큰 자릿수 고려해 영역을 넓게)
6. Save → Test scraper 로 한 핸드 인식 검증
7. 인식 성공 → 짧은 세션(30분) 으로 안정성 확인

상세 단계는 SETUP_PLAY_MONEY.ko.md §3 의 체크리스트.

알려진 한계 (정직한 한계 명시)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **2024-01 매핑 기반 → 2026 Aurora UI 미스매치**: 카드·버튼 템플릿 모두 재매핑 필요
* **베지에 마우스 모델도 PokerStars 95% 탐지를 깬다고 보장 못 함**: 다층 신호 (IP/세션/핸드 빈도) 가 더 있음. 본 변경은 HCI 표준 적용이지 탐지 우회 주장 아님
* **유전 알고리즘은 글로벌 최적화, 상황별 적응 불가**: 모던 RL (DQN/PPO/NFSP) 와 비교 시 한계 명확
* **상대 핸드 레인지 모델링 부재**: 균등분포 가정 → CFR/Pluribus 류와 격차
* **MongoDB 서버 (dickreuter.com:7778) 의존**: 템플릿·NN 모델 다운로드. 서버 다운 시 매핑 불가

도구·자료 한곳에
~~~~~~~~~~~~~~~~~

* `포크 저장소 <https://github.com/BudongJW/Poker>`_ (이 페이지)
* `play-money-kr 브랜치 <https://github.com/BudongJW/Poker/tree/play-money-kr>`_
* `업스트림 (dickreuter/Poker) <https://github.com/dickreuter/Poker>`_
* `자매 RL 프로젝트 (dickreuter/neuron_poker) <https://github.com/dickreuter/neuron_poker>`_ — Gym 환경, self-play
* SETUP_PLAY_MONEY.ko.md_ — 매핑·트러블슈팅 상세 가이드
* ``scripts/check_env.py`` — 환경 검증 스크립트
* ``scripts/smoke_test.py`` — 코어 알고리즘 5건
* ``scripts/smoke_test_extended.py`` — 마우스/이미지/OCR/Qt 15건

.. _SETUP_PLAY_MONEY.ko.md: SETUP_PLAY_MONEY.ko.md

----

(이하 업스트림 영문 README 한국어 번역 — 원본 참조용)


봇 실행하기
-----------

PartyPoker
~~~~~~~~~~
* Fast Forward 테이블 사용
* Official PartyPoker table scraper 선택

.. figure:: doc/partypoker.gif

PokerStars
~~~~~~~~~~
* Zoom 테이블 사용
* Official Poker Stars table scraper 선택
* PokerStars의 경우 클라이언트를 조정해야 한다. Official PokerStars Table scraper로 실행하려면 아래와 정확히 같은 모습이어야 한다.
* 카드 스타일(4색), 덮인 카드 스타일, 테이블 스타일이 일치하는지 확인할 것.

.. figure:: doc/ps-example.png

GGPoker
~~~~~~~
Official GGPoker Table을 사용할 때는 GGPoker 설정이 아래와 같아야 한다:

.. figure:: doc/ggpk2.png


일반 셋업
~~~~~~~~~

봇을 그냥 실행만 하고 싶다면 다음을 따라할 것:

다음 링크에서 봇 바이너리를 설치한다: http://www.deepermind-pokerbot.com

설치 후 봇을 바로 실행할 수 있다. 포커 클라이언트를 분석해 마우스를 움직이며, 미리 프로그래밍된 전략 중 하나에 따라 플레이한다.
대부분의 전략은 Zoom 또는 Fast Forward 테이블 기준이다.
필요에 맞게 전략을 수정하고 개선해 봇의 플레이를 최적화해도 좋다.

대개 포커 클라이언트를 가상머신 안에 두는 게 유리하다. 봇과 클라이언트가 서로 간섭하는 것을 막을 수 있고,
가상머신 안에서만 마우스가 움직이므로 호스트 PC 전체가 잠기는 것을 피할 수 있다:

* VirtualBox 다운로드: https://www.virtualbox.org/
* Windows 10 ISO 파일 다운로드. 예시: https://www.softlay.com/apps/operating-system
* 새 VirtualBox 환경을 만들고 ISO 파일을 시작 디스크로 사용
* VirtualBox에는 CPU를 1개만 할당할 것. 계산 자원(주로 OCR용)은 봇 쪽에서 필요하다.
* VirtualBox 환경 안에 PartyPoker 또는 PokerStars를 설치
* 포커봇은 VirtualBox 바깥, 즉 호스트 PC에 직접 설치
* Setup 화면에서 직접 마우스 제어 대신 VirtualBox 인스턴스를 선택
* 채팅 버튼이나 링크로 디스코드 채팅 참여: https://discord.gg/xB9sR3Q7r3

* 현재 버전은 Windows에서만 동작한다.
* 봇은 이미지 인식 기반이므로 테이블 시야를 가리지 말 것.
* 테이블 창은 한 개만 보이게 할 것.
* 성능을 위해 VM 안에서 테이블 창 외의 모든 창을 최소화할 것.
* VirtualBox에서 DPI 스케일링을 쓰지 말 것.
* Setup에서 직접 마우스 제어 대신 가상머신을 선택할 것. 그래야 마우스 움직임이 작업을 방해하지 않는다.


전략 분석기 (Strategy Analyzer)
--------------------------------

- 전략 분석기에서 각 전략의 수익성을 확인할 수 있다.
- 막대 차트는 각 스테이지(preflop, flop, turn, river)에서 어떤 액션 유형이 승패로 이어졌는지 보여준다.

누적 막대 차트를 더 자세히 보면 전략을 어떻게 조정해 수익을 극대화할지 단서를 얻을 수 있다:

.. figure:: doc/analyzer_bar2.png

각 스테이지의 라운드별로 분석:

.. figure:: doc/analyzer_bar3.png

개별 핸드 분석:

.. figure:: doc/strategy_analyzer.gif


전략 편집기 (Strategy Editor)
-----------------------------
봇이 결정을 내릴 때 다양한 요소가 고려되며, 전략 편집기에서 조정 가능하다:

- Equity(승리 확률). 몬테카를로 시뮬레이션으로 계산됨.
- 봇이 폴드하지 않으려면 Equity와 최소 콜/벳 값이 해당 곡선의 왼쪽에 있어야 한다.
- 이전 라운드의 행동 등 다른 다양한 요소도 고려된다. 자세한 내용은 전략 편집기에서 확인할 것.
- 각 항목 위에 마우스를 올리면 더 자세한 설명이 나온다.

.. figure:: doc/strategy2.png

전략별로 다양한 옵션을 수정할 수 있다:

.. figure:: doc/strategy_editor.gif

전략 개선 조언
~~~~~~~~~~~~~~

* 일반적으로 한 전략에서 의미 있는 결론을 끌어내려면 최소 2,000 핸드, 가능하면 5,000 핸드 플레이가 필요하다. 1,000 핸드 미만은 사실상 무작위에 가깝다.
* 최소 바이인보다 더 많이 사는 것을 권장한다. 봇에게 운신의 폭이 생겨 성능이 개선된다. 예를 들어 0.01/0.02 테이블에서 최소 $2 대신 $4 이상으로 시작하는 것이 이상적.
* 전략 분석기를 보면서 거꾸로 작업할 것. 먼저 River 플레이를 개선하고, 좋아지면 Turn → Flop 순으로 올라간다. 게임이 경로 의존적이기 때문.
* 레인지를 바꿨다면 Equity 계산이 달라지므로 모든 최소 Equity 값을 함께 조정해야 할 수 있다.
* 행운을 빈다!

풀 버전
~~~~~~~

풀 버전에서는 다음이 가능하다:

* 전략 편집 및 신규 전략 생성
* 모든 사용자의 모든 전략 조회

구매하려면 다음 링크를 따라간다. 24시간 이내에 비밀번호가 발송된다.
http://www.deepermind-pokerbot.com/purchase

또는 이메일이나 디스코드로 직접 연락해 비밀번호를 요청하고 비트코인으로 지불할 수도 있다:
1Py5o4WLYMizXc8pFPqzD4yeCAm53BhJit

코드베이스에 의미 있는 기여를 하면 무료 구독을 받을 수도 있다.

가장 쉬운 기여 방법은:

- 새 테이블 추가
- 새 전략 추가
- 코드에 직접 수정 사항을 만들어 풀 리퀘스트 생성


신규 테이블 매핑
----------------

봇은 새 테이블을 읽도록 학습할 수 있다. 템플릿을 이용하거나, 주어진 템플릿을 기반으로 데이터 증강을 사용하는 신경망을 학습시키는 두 가지 방식이 있다.

`이 링크를 누르면 새 테이블 추가 방법 영상 설명을 볼 수 있다 <https://rb.gy/jut3ws>`_. www.deepermind-pokerbot.com 에서도 볼 수 있다.

신규 테이블 추가 셋업 화면은 다음과 같다:

.. figure:: doc/scraper.png

- 포커봇을 열고 table setup을 누르면 새 테이블 생성을 돕는 창이 나타난다.
- 포커 클라이언트를 그 옆에 띄우되, 포커봇이 스크린샷을 찍을 수 있도록 DPI 스케일을 끄고 둘 것.
- 먼저 새 템플릿을 생성한다. 템플릿 이름(예: Pokerstars 1-2 zoom poker)을 입력하고 'Blank new' 클릭.
- 첫 번째로 할 일은 PokerStars 창과 테이블의 스크린샷을 찍는 것. Take screenshot 버튼을 누르면 하단 창에 화면 전체 스크린샷이 표시된다.
- 다음 단계는 포커 테이블 창의 좌상단 모서리 표시. 이후 모든 좌표의 기준점이 된다. 좌상단 모서리의 좌상단 부분을 먼저 클릭하고, 그 우하단 부분을 클릭한다. 그러면 두 번째 창에 표시된다. "save newly selected top left corner" 버튼으로 저장.
- 그다음 "crop from top left corner" 버튼으로 잘라낸다. 스크린샷 대부분이 버려지고 좌상단 모서리와 그 우측·하단으로 수백 픽셀만 남는다.
- 다음으로 창 안의 나머지 요소들을 표시한다. Buttons search area부터 시작. 버튼들의 좌상단 영역을 클릭한 후 우하단 영역을 클릭. 선택이 만족스러우면 "Buttons search area" 버튼을 누른다.
- 버튼 위에 마우스를 올리면 주의해야 할 점에 대한 더 자세한 설명이 나온다.
- 스크린샷을 여러 번 찍어 자르는 작업을 반복해야 한다(좌상단 모서리는 한 번만 선택, 이후엔 이미지를 로드해 자르기만 한다). 그 후 각 이미지에 대해 선택 영역을 만들고 해당 버튼으로 저장. 카드 한 장 한 장, 버튼 하나하나를 가르쳐야 한다.

유의사항
--------

**봇이 정상 동작하는 것을 검증하기 전에는 실계좌로 테스트하지 말 것 (계좌·자금 손실 방지)**

- 봇 실행 최소 사양:
- Windows 10 x64 이상 (이전 버전에서도 동작할 수 있으나 검증되지 않음)
- 4GB 이상 RAM
- 1.6GB 이상 하드디스크 공간 (많을수록 좋음)
- 4코어 4스레드 이상 CPU
- GPU는 필수 아님 (신경망 학습 시 GPU 옵션 사용 가능)
- 1920×1800 해상도 (그 이하에서도 동작할 수 있으나 검증되지 않음)

- VirtualBox를 쓴다면 그쪽 자원 소모도 위 사양에 더해서 고려할 것
- 포커 앱은 보통 Windows 7 이상에서 실행됨
- VirtualBox 7.0.12 이상 + 확장 팩 필요

Docker로 실행하기
-----------------

- ``$ git clone https://github.com/dickreuter/Poker.git``
- ``$ cd Poker``
- ``$ docker compose up -d``
- ``$ xhost local:root  # 로컬 머신의 root가 X windows 디스플레이에 연결 허용``
- ``$ docker-compose exec app python3 main.py  # 컨테이너 실행 후 봇 시작``

Python 소스 코드로 실행하기
---------------------------
- PyCharm Community Edition 다운로드: https://www.jetbrains.com/pycharm/download/#section=windows
- Anaconda 설치: https://www.anaconda.com/products/distribution
- tesserocr 다운로드: https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.6.0-tesseract-5.3.1/tesserocr-2.6.0-cp311-cp311-win_amd64.whl 받아서 pip로 휠 파일 설치
- ``pip install -r requirements.txt``로 환경 구성 후 tesserocr는 별도 pip install
- C++ 런타임 재배포 패키지가 필요할 수 있음: https://visualstudio.microsoft.com/downloads/
- VirtualBox 설치 후 포커 클라이언트를 VM 안에 둘 것. 호스트 마우스를 쓰지 않고도 제어 가능. https://www.virtualbox.org/wiki/Downloads
- 위에서 만든 가상환경을 인터프리터로 설정한 뒤 PyCharm에서 ``main.py`` 실행 (관련 YouTube 영상 참고)


패키지 및 모듈
~~~~~~~~~~~~~~

main.py: 진입점

poker.scraper
^^^^^^^^^^^^^

사용자 인터페이스와 새 테이블 매핑 보조 루틴이 들어 있다.

- ``recognize_table``: 매핑 정보에 기반해 테이블 위 각 요소를 인식하는 함수들
- ``screen_operations``: 스크린샷 캡처, 자르기 등 각종 루틴
- ``table_setup``: GUI 관련 루틴
- ``ui_table_setup``: QT 사용자 인터페이스. 상위 폴더의 makegui.bat로 .py 파일이 생성된다. GUI를 수정하려면 QT Designer를 다운로드해 .ui 파일을 연다.


poker.decisionmaker
^^^^^^^^^^^^^^^^^^^

- ``decisionmaker.py``: 입력을 바탕으로 어떤 액션을 취할지 최종 결정
- ``montecarlo_numpy2.py``: NumPy 기반의 빠른 몬테카를로 시뮬레이션으로 Equity 계산. 아직 정상 동작하지 않음. 일부 테스트가 실패 중. 수정 환영.
- ``montecarlo_python.py``: 비교적 느린 순수 Python 몬테카를로. 상대방의 preflop 레인지를 지원한다.

poker.tests
^^^^^^^^^^^

- ``test_montecarlo_numpy.py``: NumPy 몬테카를로 테스트
- ``test_pylint.py``: PEP8 준수와 정적 분석을 위한 pylint, pydoc 테스트


그래픽 사용자 인터페이스 (GUI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- 오픈소스용 QT 다운로드: https://www.qt.io/download-open-source
- QT Designer로 gui/ui 폴더의 .ui 파일을 편집


다음 우선순위
-------------


- [ ] 테스트 업데이트. 일부가 낡았다. 더 많은 테스트가 필요.
- [ ] 전략 추가
- [ ] pytesseract에서 tesserocr로 전환해 OCR 속도 개선. 봇이 상당히 빨라질 것으로 예상.
- [ ] 수집된 데이터를 더 잘 분석해 전략 개선


코드 수정·기여하기
~~~~~~~~~~~~~~~~~~

- 코드를 수정하고 origin/master로 풀 리퀘스트 보내기:

혼자서 세계 최고의 포커 플레이어를 이기는 건 어렵다. 그래서 이 저장소는 모델을 추가하고 평가하는 협업 환경을 지향한다.

기여하려면 다음을 따른다:

- PyCharm을 받고 Python 가상환경을 구성한다. ``pip install -r requirements.txt`` 사용 (위 참조)
- 포크를 로컬로 클론. PyCharm에서 VCS → check out from version control → git로 직접 가능
- 포크 원본 저장소를 upstream이라는 이름의 리모트로 추가 (포크에 대한 연결은 origin으로 둘 것). VCS → git → remotes
- 새 브랜치 생성: 우하단 master 클릭 후 'new branch'
- 수정 작업
- 모든 테스트 통과 확인. File → Settings → Python integrated tools에서 pytest로 전환. 그 후 tests 폴더 우클릭 → 전체 실행. 모든 테스트가 통과해야 한다. 본인의 테스트는 함수명을 test\_... 로 시작하면 된다.
- pytest로 모든 테스트가 통과하는지 확인할 것 (PyCharm에서 tests 폴더 우클릭 → 실행). 실패한 테스트는 우클릭으로 디버그하면서 브레이크포인트나 콘솔로 살펴볼 수 있다: https://stackoverflow.com/questions/19329601/interactive-shell-debugging-with-pycharm
- 변경 사항 커밋 (CTRL+K)
- origin(자신의 포크)로 푸시 (CTRL+SHIFT+K)
- upstream master가 진척되어 브랜치를 최신화해야 한다면 upstream master로 리베이스: PyCharm 우하단 브랜치명 → upstream/master → rebase onto. 충돌은 해결할 것. 끝난 뒤에는 항상 force-push (단순 push 아님). push 옆 드롭다운에서 force-push 선택 (중요: 리베이스한 브랜치를 원격과 push & merge 하지 말 것)
- github.com에서 자신의 브랜치를 upstream master로 머지하는 풀 리퀘스트 생성
- 풀 리퀘스트가 승인되면 upstream/master로 머지됨
- pylint 테스트를 포함한 모든 테스트가 통과하는지 확인할 것. 로컬에서 실행하거나 푸시 후 GitHub의 travis 로그로 확인. [현재 다수 실패 중. 수정 도와주면 환영!]



FAQ
---

좌상단 모서리가 잡히지 않음
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 모든 것이 이 문서 상단의 그림과 정확히 같은지 확인할 것.
    * 버튼은 그림과 똑같아야 하고, 영어여야 하며, 스케일되지 않아야 한다. 색상은 기본값.
    * 대부분의 테이블은 실머니 기준으로 매핑되어 있다. 플레이머니에서는 동작하지 않는다.
    * 포커 테이블 창이 전부 보여야 하며 스케일되면 안 된다. 그렇지 않으면 제대로 감지되지 않는다.
    * PartyPoker는 테이블을 연 뒤 table options → **back to default size**로 기본 크기로 맞출 것.

- 테이블은 fast forward 및 zoom 실머니 게임 기준으로 매핑되어 있다. PartyPoker는 Supersonic3, PokerStars는 McNaught 테이블 사용.
- 그래도 안 되면 위에서 설명한 대로 신규 테이블을 가르쳐 볼 것.

카드가 인식되지 않음
~~~~~~~~~~~~~~~~~~~~

- 모든 것이 문서 상단의 그림과 정확히 같은지 확인할 것.
    * 실머니인지 확인. 플레이머니는 테이블이 다르다.
    * 위 그림과 같은 자리에 앉아 있는지 확인.
    * 현재는 6인 테이블에서만 동작.


가상머신을 꼭 써야 하나?
~~~~~~~~~~~~~~~~~~~~~~~~

- PokerStars는 반드시 써야 한다. 그렇지 않으면 몇 분 안에 차단되고 계정이 동결될 것이다. PartyPoker는 확실하지 않으니 약관을 확인할 것.


로그 분석과 문제 보고
~~~~~~~~~~~~~~~~~~~~~

- 포커봇 설치 폴더 안의 /log 하위 폴더에 로그 파일이 있다. /log/screenshots에 디버깅을 도와줄 스크린샷도 있다.
- 본 GitHub 페이지 상단 링크에서 이슈를 만들거나 dickreuter@gmail.com 으로 메일.


관련 프로젝트
-------------
봇끼리 self-play로 학습시키는 것은 별도 프로젝트로 분리되어 있다:
https://github.com/dickreuter/neuron_poker
