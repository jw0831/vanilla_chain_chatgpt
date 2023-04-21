### requirements ###
# pip install streamlit
# pip install st-annotated-text

# https://github.com/syasini/sophisticated_palette
    # 내부 기능 탭
# https://github.com/wjbmattingly/youtube-shakespeare
    # tab 기능
# %%
# sys.path.append('/home/aift-ml/workspace/home/aift-ml/workspace/JinWon/NER/119NER_model_develop_v3_evalutation/ner_api_init')
# https://github.com/tvst/st-annotated-text
# https://www.youtube.com/watch?v=ipVnSCWFgis
import os, sys
from dotenv import load_dotenv
load_dotenv('./../../ChatGPT/.env')

import streamlit as st
import tiktoken
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from pprint import pprint

# .env 파일에서 환경변수 불러오기로 다른경로에서 읽어오도록 만들기
# OpenAI API Key
# print(os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# %%
# youtube 주소 입력란
############################################
# '''
# 만약 내부적으로 youtube 주소가 있을 경우 그리고 요약본이 있을경우 
# - 미리 추출된 transcript 요약을 제공
# - transcript만 있을경우 요약 돌리기
# - 둘다 없을 경우 다운받고 transcript추출하기 그리고 요약 돌리기
# '''

############################################

# Transcript 받아오기 : 함수로 만들어서 사용하기
############################################
def read_transcript(video_id):
    with open(f"./transcript_file/{video_id}_transcript.txt", 'r', encoding='utf-8') as f:
        _transcript = f.read()

    return _transcript
# print('transcript read:', transcript[:50])

def chunk_transcript(transcript):
    transcript = transcript.replace('. ', '.\n\n')
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    texts = text_splitter.split_text(transcript) # 문장 분리 길이 4000 미만 및 chunk_overlap 200
    _docs = [Document(page_content=t) for t in texts]
    return _docs

enc35 = tiktoken.encoding_for_model("gpt-3.5-turbo")
# tokenized_text_chatgpt35 = enc35.encode(transcript)
# print(len(tokenized_text_chatgpt35))
############################################

# Prompt template 생성
############################################
prompt_template = """다음 문장을 요약해주세요:

{text}

한국어 문장요약 결과:
"""
############################################################################################################
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

refine_template = (
    "당신의 임무는 최종 요약을 작성하는 것입니다\n"
    "특정 지점까지 기존 요약을 제공했습니다: {existing_answer}\n"
    "기존 요약을 다듬을 기회가 있습니다."
    "(필요한 경우에만) 아래에 더 많은 컨텍스트를 추가할 수 있습니다.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "새로운 문맥이 주어지면, 원래 요약을 한국어로 수정하세요."
    "문맥이 유용하지 않은 경우 원래 요약을 반환합니다."
)
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
)


############################################

# 요약 함수 # refine 먼저 만들어보기
############################################
# 나중에 option 받도록 하기
    # option refine
    # option map_reduce
llmc = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") # chatgpt 3.5 model

map_reduce_chain = load_summarize_chain(llmc, 
                             chain_type="map_reduce",  # map_reduce 방식 각각의 chunk에 대해 요약을 먼저 진행하고 그 결과를 다시 요약하는 방식 -> 각각의 요약에대해 병렬처리함
                             return_intermediate_steps=True,
                             map_prompt=PROMPT,
                             combine_prompt=PROMPT,
                             verbose=False)

refine_chain = load_summarize_chain(llmc, 
                             chain_type="refine",  # refine 방식
                             return_intermediate_steps=True,
                             question_prompt=PROMPT,
                             refine_prompt=refine_prompt,
                             verbose=False)

############################################

# db 영역
############################################
df = pd.read_csv('./db/sample_youtube_db.csv') # 경로는 main.py를 기준으로 함
############################################
# video search
############################################
def get_video_metadata(video_id):
    try:
        _video_index = df.video_id.tolist().index(video_id)
        # _video = df.iloc[_video_index]
        _metadata = df.iloc[_video_index].to_dict()
    except:
        _video_index = None
        _metadata = None
    
    return _video_index, _metadata

############################################

################ streamlit 작업 영역 ################
st.header("Youtube 영상 요약기")

################################################################################
# print(sentence)

debug = 1
if debug:
    st.caption("test youtube 주소 입력: https://www.youtube.com/watch?v=9vbwK3OMyiM")
    sentence = st.text_input("Youtube 주소를 입력해주세요:").strip()

    if sentence.startswith("https://www.youtube.com"):
        video_id = sentence.split('/')[-1] # video_id = "watch?v=9vbwK3OMyiM"

    elif sentence.startswith("watch?v="):
        # https://www.youtube.com/watch?v=eDQhkDaaEV4
        video_id = sentence

    else:
        st.write("youtube 주소를 입력해주세요")
        video_id = None


    if video_id is not None:
        st.caption("youtube 주소가 입력되었습니다:")
        # provide options to either select an image form the gallery, upload one, or fetch from URL
        summary_tab, transcript_tab = st.tabs(["Gallery", "transcript"])

        with summary_tab:
            # todo : 
                # 1. db에서 video_id가 있는지 확인
                # 2. 있으면 transcript를 읽어옴
                # 3. 없으면 transcript를 다운받고 읽어옴
                # 4. 요약본 있는지 확인
                # 5. 있으면 요약본을 읽어옴
                # 6. 없으면 요약을 진행함
                # 7. 요약본을 db에 저장함

            video_index = df.video_id.tolist().index(video_id)
            video_index, metadata = get_video_metadata(video_id)
            # if video_index is None:
                # st.caption("db에 video_id가 없습니다.")
                # st.caption("transcript를 다운로드합니다.")
                # download_transcript(video_id=video_id)
                # st.caption("transcript를 다운로드했습니다.")
                # metadata = get_video_metadata(video_id)
                # st.caption("metadata를 읽어왔습니다.")
                # st.write(metadata)
            # else:
                # st.caption("db에 video_id가 있습니다.")
            

            st.subheader("video metadata")
            st.write(metadata)

            st.write('video_title:', metadata["title"])

            transcript = read_transcript(video_id=metadata["video_id"])
            st.caption("transcript를 읽어왔습니다.")
            st.write('transcript:', transcript[:50], '...')

            docs = chunk_transcript(transcript)
            st.caption("transcript를 chunk로 나눴습니다.")

            st.caption("요약을 시작합니다.")
            st.caption("요약 중입니다...") # 나중에 요약이 끝나면 해당 내용을 지우고 완료됨을 알리기

            res = refine_chain({"input_documents": docs}, return_only_outputs=True)
            # res = {"output_text": "2030세대는 하락장을 경험하지 않아 하락장에 대한 상상력이 부족하며, 부동산 대출로 집을 산 경우 하락장에서 어떤 일이 벌어지는지에 대해 경고하고 있다. 집값이 떨어지면 대출 이자를 갚는 것만으로도 어려움을 겪게 되며, 이로 인해 가정 내부에서는 불만이 쌓이고 이혼, 자살 등의 문제가 발생할 수 있다. 또한, 부동산 시장에서는 손님이 없어 더 싸게 내놓아야 한다는 생각으로 집값이 더 떨어지는 악순환에 빠질 수 있다는 경고를 하고 있다. 부동산 시장에서는 타이밍 싸임이 중요하며, 수요 공급에 대한 이해를 제대로 해야 한다는 것이다. 부동산 시장은 사이클이 있으며, 이를 정확하게 이해하고 적절한 대응을 하면 부동산으로 잃지 않고 돈을 많이 벌 수 있다는 것이다."}
            st.caption("요약이 완료되었습니다.")

            st.subheader("요약 중간 과정")
            st.write(res['intermediate_steps'])

            st.subheader("요약 결과")
            st.write(res['output_text'])

            st.caption("transcript chunking 결과는 다음과 같습니다.")
            st.write(docs)

        with transcript_tab:
            st.caption("추출된 transcript는 다음과 같습니다.")
            st.text(transcript.replace('. ', '.\n'))
    
    
else:
    sentence = st.text_input("Youtube 주소를 입력해주세요:")
    if sentence.startswith("https://www.youtube.com/"):
        pass
    else:
        st.write("youtube 주소가 아닙니다.")

        
    # st.subheader("개체 리스트")
    # st.text("ORG:기관|식당\nPER:사람|이름\nMET:거리|길이|높이\nSIT:응급상황\nLOC:장소|주소\nQTY:수량\nNUM:기타숫자|번호\nDIR:방향\nTIM:시간")
    



# %%
