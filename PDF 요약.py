import os #os: 운영체제 관련 작업(예: 파일 경로 처리)을 위해 임포트. 현재는 직접 사용되는 부분은 없음.
from PyPDF2 import PdfReader #PyPDF2.PdfReader: PDF 파일을 읽고 텍스트 추출.
import streamlit as st #streamlit: 웹 앱 UI를 쉽게 만들기 위한 라이브러리.
from langchain_text_splitters import CharacterTextSplitter #LangChain에서 긴 텍스트를 **조각(chunk)**으로 나눌 때 사용. 모델 입력 길이 제한 관리용.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI #OpenAI 임베딩 생성과 LLM(대화형 모델) 호출용.
from langchain_community.vectorstores import FAISS #FAISS: 텍스트 벡터를 저장하고 유사 문서를 검색하는 벡터 DB.
from langchain_classic.chains.question_answering import load_qa_chain #load_qa_chain: LangChain의 질문-응답 체인(QA 체인)을 생성.
from langchain_community.callbacks import get_openai_callback #get_openai_callback: OpenAI API 호출 시 비용 추적용 콜백.
import openai  # 키 유효성 검사를 위해 추가

# API 키 유효성 검사 함수
def check_api_key(api_key):
    try:
        
        client = openai.OpenAI(api_key=api_key) #openai.OpenAI(api_key=...) 객체 생성
        client.models.list() #client.models.list() → 아주 작은 API 호출
        return True
    except Exception: # 예외가 발생하면 여기 코드 실행
        return False

def process_text(text, api_key): 
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, #chunk_size=1000: 최대 1000자 단위로 분할.
        chunk_overlap=200, #chunk_overlap=200: 연속된 chunk 간 200자 중복 → 문맥 유지.
        length_function=len #separator="\n": 줄바꿈 단위로 분리 시도.
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    documents = FAISS.from_texts(chunks, embeddings) #각 chunk를 임베모델은 이용해서 → 벡터 , 벡터 DB에 저장
    return documents

#streamlit 인터페이스 ui
def main(): 
    st.set_page_config(page_title="PDF 요약 사이트", page_icon="📄")
    st.title("📄 PDF을 올려주시면 요약해 드립니다.")
    st.divider() #st.divider: 시각적 구분선 → UI 깔끔하게

    # 사이드바 설정
    with st.sidebar:
        st.title("설정")
        # secrets에서 API Key 가져오기 시도
        default_key = st.secrets.get("OPENAI_API_KEY", "")
        
        user_api_key = st.text_input("OpenAI API Key를 입력하세요", value=default_key, type="password") #type="password" → 입력값 숨김 처리
        
        # 키 입력 여부에 따른 상태 메시지 표시
        if user_api_key:
            if check_api_key(user_api_key):
                st.success("✅ 연결되었습니다!")
            else:
                st.error("❌ 유효하지 않은 키입니다. 다시 확인해 주세요.")
        else:
            st.warning("🔑 API Key를 입력해 주세요.")
            
        st.markdown("[API Key 발급받기](https://platform.openai.com/api-keys)")

    pdf = st.file_uploader('PDF파일을 업로드해주세요', type='pdf')

    if pdf is not None:
        # 키 검증이 실패하면 진행하지 않음
        if not user_api_key or not check_api_key(user_api_key):
            st.info("먼저 유효한 OpenAI API Key를 입력해 주세요.")
            st.stop()

        pdf_reader = PdfReader(pdf) #PDF에서 텍스트 추출
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = process_text(text, user_api_key) #process_text → FAISS 객체 반환
        query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요." #query → 요약 요청

        if query:
            docs = documents.similarity_search(query) #similarity_search(query): FAISS 벡터 DB에서 query와 가장 유사한 chunk 검색
            llm = ChatOpenAI(model="gpt-3.5-turbo-16k", api_key=user_api_key, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with st.spinner('PDF 내용을 분석하여 요약 중입니다...'):
                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)

            st.subheader('-- 요약 결과 --')
            st.write(response)
            st.caption(f"발생 비용: ${cost.total_cost:.4f}")

if __name__ == '__main__':
    main()