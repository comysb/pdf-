# PDF 요약 사이트 (PDF Summarizer)

이 애플리케이션은 업로드된 PDF 파일의 내용을 분석하고 OpenAI의 GPT 모델을 사용하여 3~5문장으로 요약해 주는 Streamlit 기반 웹 앱입니다.

## 주요 기능

-   **PDF 텍스트 추출**: `PyPDF2`를 사용하여 PDF 파일에서 텍스트를 읽어옵니다.
-   **텍스트 분할 및 임베딩**: LangChain의 `CharacterTextSplitter`와 OpenAI의 임베딩 모델을 사용하여 긴 문서를 조각내고 벡터화합니다.
-   **시맨틱 검색**: FAISS 벡터 스토어를 사용하여 질문(요약 요청)과 가장 관련 있는 문서 조각을 검색합니다.
-   **AI 요약**: OpenAI의 `gpt-3.5-turbo-16k` 모델을 통해 정확하고 간결한 요약을 제공합니다.
-   **API 키 관리**: 사용자가 직접 OpenAI API 키를 입력하여 사용할 수 있으며, 실시간으로 유효성을 검사합니다.
-   **비용 추적**: 요약 과정에서 발생하는 OpenAI API 비용을 표시합니다.

## 설치 및 실행 방법

1.  **레포지토리 클론**:
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **가상환경 설정 (권장)**:
    ```bash
    python -m venv venv
    ./venv/Scripts/activate  # Windows
    # source venv/bin/activate  # macOS/Linux
    ```

3.  **필수 패키지 설치**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **애플리케이션 실행**:
    ```bash
    streamlit run "PDF 요약.py"
    ```

## 기술 스택

-   **Frontend**: Streamlit
-   **LLM Framework**: LangChain
-   **LLM**: OpenAI GPT-3.5 Turbo
-   **Vector Store**: FAISS
-   **PDF Parsing**: PyPDF2
