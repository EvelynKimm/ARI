import re
from typing import List
from unicodedata import normalize

DAMAGE_TOKEN = "□"
RESTORE_END_TOKEN = "[RES_END]"

KING_KOREAN_TO_HANJA = {
    "태조": "太祖",
    "정종": "定宗",
    "태종": "太宗",
    "세종": "世宗",
    "문종": "文宗",
    "단종": "端宗",
    "세조": "世祖",
    "예종": "睿宗",
    "성종": "成宗",
    "연산": "燕山",
    "중종": "中宗",
    "인종": "仁宗",
    "명종": "明宗",
    "선조": "宣祖",
    "광해": "光海",
    "인조": "仁祖",
    "효종": "孝宗",
    "현종": "顯宗",
    "숙종": "肅宗",
    "경종": "景宗",
    "영조": "英祖",
    "정조": "正祖",
    "순조": "純祖",
    "헌종": "憲宗",
    "철종": "哲宗",
    "고종": "高宗",
    "순종": "純宗",
}


def process_text(text: str) -> str:
    # normalize
    text = normalize("NFKC", text)

    # 어려운 특수기호를 쉬운 기호로 치환 및 통합
    text = re.sub("、|，", ", ", text)
    text = re.sub("。|․", ". ", text)
    text = re.sub("‘|’", "'", text)
    text = re.sub("“|”", '"', text)  # 추가
    text = re.sub("〈|《", "<", text)
    text = re.sub("〉|》", ">", text)
    text = re.sub("【|〔|［", "[", text)
    text = re.sub("】|〕|］", "]", text)
    text = re.sub("｢|『", "「", text)
    text = re.sub("｣|』", "」", text)
    text = re.sub("ㆍ|･|ᆞ", "·", text)

    # 불필요한 특수기호 제거 (주로 문서 시작 부분에 나타남)
    text = text.replace("○", "")
    text = text.replace("〈〉", "")

    # 훼손된 글자 종류를 □로 통합
    text = re.sub("◆|●|■", "□", text)

    # 중복 스페이스 제거 및 strip
    text = " ".join(text.split())

    # 시작시에 일자 간지 기호 제거 (e.g. ○癸丑/輪對. -> 輪對.)
    # text = re.sub(r"^○?\S{2}\/", "", text)
    text = re.sub(r"^○", "", text)

    return text


def process_ner_texts(ner_texts: List[str], min_len: int, max_len: int) -> List[str]:
    ner_list = []
    for ner_text in ner_texts:
        ner_text = process_text(ner_text)
        # <>, () 등 특수 문자 제거
        ner_text = re.sub(r"[<>\(\)\[\]]", "", ner_text)
        for ner_sub_text in re.split(r"\s+|,|\.", ner_text):
            ner_sub_text = ner_sub_text.strip()
            if min_len <= len(ner_sub_text) <= max_len:
                ner_list.append(ner_sub_text)

    return ner_list
