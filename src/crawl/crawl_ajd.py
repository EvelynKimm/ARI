import os
import re
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from functools import partial
from glob import glob
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple
from urllib import parse

import ujson as json
import urllib3
from tqdm import tqdm

from src.crawl.utils import log_exception_info, request_and_build_soup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

argparser = ArgumentParser("Collect Hanja–Korean documents from the Annals of the Joseon Dynasty.")
argparser.add_argument("--save-dir", type=str, default="./data/test")
argparser.add_argument("--resume", action="store_true")
argparser.add_argument("--num-workers", type=int, default=12)

hanja_unicode_ranges = "|".join(["\u3400-\u9fff", "\uf900-\ufaff", "\U00020000-\U0003134f"])
HANJA_PATTERN = re.compile(f"[{hanja_unicode_ranges}]+")


def crawl_king_page(king_id: str) -> List[Tuple[str, str, str, str, str]]:
    data = {"id": king_id}
    year_list_soup = request_and_build_soup("https://sillok.history.go.kr/search/inspectionMonthList.do", data=data)

    book_name = year_list_soup.find("li", {"class": "loc_thispage"}).text.strip()
    king_name = year_list_soup.find("div", {"class": "page_tit clear2 tit_wrap_small"}).find("h3")
    king_name = re.match("자료열람 : (.*)", king_name.text).group(1)

    year_box = year_list_soup.find("ul", {"class": "king_year2"})
    year_list = year_box.find_all("span", {"class": "king_desc02"})
    year_months_list = year_box.find_all("ul")

    king_month_metadata = []
    for year_item, year_months_item in zip(year_list, year_months_list):
        real_year = re.match(r"(.*) \(.*\)", year_item.text).group(1)

        for month_a in year_months_item.find_all("a"):
            real_month = month_a.text.strip()
            month_id = re.match(r"javascript:search\('(.*?)'", month_a["href"]).group(1)

            king_month_metadata.append((king_name, book_name, real_year, real_month, month_id))

    return king_month_metadata


def crawl_document_page(document_id: str, crawl_english_page: bool = False) -> Tuple[str, Dict[str, Any]]:
    real_day, korean_paragraphs, hanja_paragraphs = None, None, None
    for trial in range(10):
        try:
            document_id = parse.quote(document_id)
            soup = request_and_build_soup(f"https://sillok.history.go.kr/id/{document_id}", request_mode="get")
            real_day = soup.find("span", {"class": "tit_loc"}).text.strip()
            if real_day.endswith("미상"):
                real_day = "[미상]"
            else:
                real_day = re.match(".*월 (.*?) .*", real_day).group(1)

            korean_paragraph_box = soup.find("div", {"class": "ins_view_left"}).find("div", {"class": "ins_view_pd"})
            hanja_paragraph_box = soup.find("div", {"class": "ins_view_right"}).find("div", {"class": "ins_view_pd"})

            left_right_view = korean_paragraph_box.find("div", {"class": "ins_view_right"})
            if left_right_view is not None:
                left_right_view.extract()

            korean_paragraphs = []
            hanja_paragraphs = []
            ner_words = []

            for paragraph in korean_paragraph_box.find_all("p", {"class": "paragraph"}):
                for sup in paragraph.find_all("sup"):
                    sup.extract()

                text = " ".join(paragraph.text.split())
                text = re.sub(r"[\[\(].*?[\)\]]", "", text)
                korean_paragraphs.append(text)

            for paragraph in hanja_paragraph_box.find_all("p", {"class": "paragraph"}):
                for annotation_tag in paragraph.find_all("span", {"class": "idx_annotation01"}):
                    annotation_tag.extract()

                ner_words.extend(
                    [(ner.text.strip(), "person") for ner in paragraph.find_all("span", {"class": "idx_person"})]
                )
                ner_words.extend(
                    [(ner.text.strip(), "place") for ner in paragraph.find_all("span", {"class": "idx_place"})]
                )
                ner_words.extend(
                    [(ner.text.strip(), "book") for ner in paragraph.find_all("span", {"class": "idx_book"})]
                )
                ner_words.extend(
                    [(ner.text.strip(), "era") for ner in paragraph.find_all("span", {"class": "idx_era"})]
                )

                text = " ".join(paragraph.text.split())
                hanja_paragraphs.append(text)

            break

        except AttributeError:
            log_exception_info(document_id=document_id)
            time.sleep(1)
            continue

    if korean_paragraphs is None or hanja_paragraphs is None:
        return real_day, {"korean": [], "hanja": [], "ner": []}

    korean_document = " ".join(korean_paragraphs)
    korean_document = " ".join(korean_document.split())
    hanja_document = " ".join(hanja_paragraphs)
    hanja_document = " ".join(hanja_document.split())

    if crawl_english_page:
        page_id = document_id.replace("kda", "eda")
        soup = request_and_build_soup(
            "http://esillok.history.go.kr/record/getDetailViewAjax.do",
            request_mode="post",
            data={"id": page_id, "sillokViewType": "Eng"},
        )

        english_text_box = soup.find("div", {"class": "inner"})
        english_text = ""
        if english_text_box is not None:
            english_texts = []
            for p in english_text_box.find_all("p", {"class": "txt"}):
                for sup in p.find_all("a", {"class": "sup"}):
                    sup.extract()

                english_texts.append(" ".join(p.text.split()))

            english_text = " ".join(english_texts)

            english_text = HANJA_PATTERN.sub("", english_text)
            english_text = re.sub(r"^On .*? day,", "", english_text)
            english_text = re.sub(r"\(.*?\)", "", english_text)
            english_text = re.sub(r"[\[\]]", "", english_text)
            english_text = " ".join(english_text.split())
            english_text = re.sub(r" ([\.\?,])", r"\1", english_text)

    document = {"korean": korean_document, "hanja": hanja_document, "ner": ner_words}
    if crawl_english_page and len(english_text) > 0:
        document["english"] = english_text

    return real_day, document


def main(args: Namespace):
    num_total_documents = 0

    if args.resume:
        crawled_metadata = []
        crawled_file_paths = glob(f"{args.save_dir}/*/*/*.json")
        for file_path in crawled_file_paths:
            book_name, real_year, real_month = re.match(f"{args.save_dir}/(.*)/(.*)/(.*).json", file_path).groups()
            crawled_metadata.append((book_name, real_year, real_month))

    king_list_soup = request_and_build_soup("https://sillok.history.go.kr/search/inspectionList.do")
    king_id_book_name_list = []
    for tbody in king_list_soup.find_all("tbody"):
        for tr in tbody.find_all("tr"):
            king_id = re.match(r"javascript:search\('(.*)'\)", tr.find("td").find("a")["href"]).group(1)
            king_id_book_name_list.append(king_id)

    king_month_metadata_list = []
    with Pool(args.num_workers) as pool:
        for king_month_metadata in pool.imap_unordered(crawl_king_page, king_id_book_name_list, chunksize=20):
            king_month_metadata_list.extend(king_month_metadata)

        if args.resume and len(crawled_metadata) > 0:
            king_month_metadata_list = [data for data in king_month_metadata_list if data[1:4] not in crawled_metadata]
        king_month_metadata_list = sorted(king_month_metadata_list)

        progress = tqdm(total=len(king_month_metadata_list))
        for king_name, book_name, real_year, real_month, month_id in king_month_metadata_list:
            save_folder_dir = os.path.join(args.save_dir, book_name, real_year)
            os.makedirs(save_folder_dir, exist_ok=True)
            documents = defaultdict(list)

            data = {"id": month_id, "level": 3}
            month_soup = request_and_build_soup("https://sillok.history.go.kr/search/inspectionDayList.do", data=data)
            document_ids = [li.find("a")["href"] for li in month_soup.find("dd").find_all("li")]
            document_ids = [re.match(r"javascript:searchView\('(.*?)'", href).group(1) for href in document_ids]

            crawl_document_page_fn = partial(crawl_document_page, crawl_english_page=king_name == "세종")
            for real_day, document in pool.imap_unordered(crawl_document_page_fn, document_ids):
                if len(document["hanja"]) > 0 and len(document["korean"]) > 0:
                    num_total_documents += 1
                    documents[real_day].append(document)

            save_file_path = os.path.join(save_folder_dir, f"{real_month}.json")
            with open(save_file_path, "w") as f:
                json.dump(
                    {
                        "data_type": "ajd",
                        "king": king_name,
                        "real_year": real_year,
                        "month": real_month,
                        "documents": documents,
                    },
                    f,
                    ensure_ascii=False,
                )

            progress.set_postfix({"documents": num_total_documents})
            progress.update(1)


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args)
