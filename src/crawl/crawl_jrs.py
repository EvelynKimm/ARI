import os
import re
import time
from argparse import ArgumentParser, Namespace
from functools import partial
from glob import glob
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import ujson as json
import urllib3
from tqdm import tqdm

from src.crawl.utils import log_exception_info, request_and_build_soup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

argparser = ArgumentParser("Collect Hanja documents from the Diaries of the Royal Secretariat.")
argparser.add_argument("--save-dir", type=str, default="./data/jrs")
argparser.add_argument("--resume", action="store_true")
argparser.add_argument("--num-workers", type=int, default=12)


def crawl_king_page(king_href: str) -> List[Tuple[str, str, str, str]]:
    king_id, king_name = re.match(r"javascript:searchMonthList\('(.*)', '1', false, '(.*)'\)", king_href).groups()

    data = {"treeID": king_id, "treeLevel": 1, "treeType": "왕대별", "treeKingName": king_name}
    year_list_soup = request_and_build_soup("https://sjw.history.go.kr/search/inspectionMonthList.do", data=data)

    year_box = year_list_soup.find("ul", {"class": "king_year2"})
    year_list = year_box.find_all("span", {"class": "king_desc02"})
    year_months_list = year_box.find_all("ul")

    king_month_metadata = []
    for year_item, year_months_item in zip(year_list, year_months_list):
        real_year = re.match(r"(.*)년 \(.*\)", year_item.text).group(1)

        for month_a in year_months_item.find_all("a"):
            real_month = month_a.text.strip()
            month_start_day_id = re.match(r"javascript:searchDayList\('(.*?)'", month_a["href"]).group(1)

            king_month_metadata.append((king_name, real_year, real_month, month_start_day_id))

    return king_month_metadata


def crawl_day_page(
    king_name: str, day_id: str, return_day_ids: bool = False
) -> Tuple[str, List[Dict[str, Any]], Optional[List[str]]]:
    while True:
        try:
            data = {"treeID": day_id, "treeLevel": 3, "treeType": "왕대별", "treeKingName": king_name}
            soup = request_and_build_soup("https://sjw.history.go.kr/search/inspectionDayList.do", data=data)

            day_list = soup.find("span", {"class": "day_list"}).find("ul")
            real_day = day_list.find("a", {"class": "on"}).extract().text.strip()

            day_ids = None
            if return_day_ids:
                day_ids = []
                for day in day_list.find_all("a"):
                    day_ids.append(re.match(r"javascript:searchDayList\('(.*?)'", day["href"]).group(1))

            if real_day == "요목":
                return real_day, [], day_ids

            crawl_korean_page = False
            for button in soup.find_all("a", {"class": "btn_subview2 btn_connect"}):
                if button.text == "국역":
                    crawl_korean_page = True
                    break

            document_list = soup.find("ul", {"class": "sjw_list"})
            for nav in document_list.find_all("div", {"class": "ins_side_box"}):
                nav.extract()

            documents = []
            for document in document_list.find_all("li"):
                document_link = document.find("a")["href"]
                document_link = re.match(r"javascript:searchView\('(.*?)'\)", document_link).group(1)
                document_box = document.find("span")

                for idx_place in document_box.find_all("span", {"class": "idx_place"}):
                    idx_place.extract()

                for annotation_tag in document_box.find_all("span", {"class": "idx_annotation01"}):
                    annotation_tag.extract()
                for annotation_tag in document_box.find_all("span", {"class": "idx_name"}):
                    annotation_tag.extract()

                ner_words = []
                ner_words.extend(
                    [
                        (ner_word.text.strip(), "person")
                        for ner_word in document_box.find_all("span", {"class": "idx_person"})
                    ]
                )
                ner_words.extend(
                    [
                        (ner_word.text.strip(), "place")
                        for ner_word in document_box.find_all("span", {"class": "idx_place2"})
                    ]
                )
                ner_words.extend(
                    [
                        (ner_word.text.strip(), "book")
                        for ner_word in document_box.find_all("span", {"class": "idx_book"})
                    ]
                )
                ner_words.extend(
                    [(ner_word.text.strip(), "era") for ner_word in document_box.find_all("span", {"class": "idx_era"})]
                )

                document_text = " ".join(document_box.text.split()).strip()
                if len(document_text) > 0:
                    document_dict = {"hanja": document_text, "ner": ner_words}

                    if crawl_korean_page:
                        soup = request_and_build_soup(
                            f"https://sjw.history.go.kr/id/{document_link}", request_mode="get"
                        )

                        korean_translation_page_id = None
                        for button in soup.find_all("a", {"class": "btn_subview2 btn_connect"}):
                            if button.text == "국역":
                                korean_translation_page_id = button["href"].split(",")[-2].replace("'", "").strip()

                        if korean_translation_page_id is not None:
                            korean_page_url = f"https://db.itkc.or.kr/m/dir/view?dataId={korean_translation_page_id}"
                            soup = request_and_build_soup(korean_page_url, request_mode="get")
                            text_body = soup.find("div", {"class": "text_body"})

                            for span in text_body.find_all("span"):
                                span.extract()

                            korean_text = ""
                            for paragraph in text_body.find_all("div", {"class": {"xsl_para"}}):
                                text = " ".join(paragraph.text.strip().split())
                                text = re.sub(r"[\[\(].*?[\)\]]", "", text)
                                korean_text = f"{korean_text} {text}"

                            korean_text = " ".join(korean_text.split())
                            if len(korean_text) > 0:
                                document_dict["korean"] = korean_text

                    documents.append(document_dict)

            break

        except AttributeError:
            log_exception_info(day_id=day_id, page_url=korean_page_url)
            time.sleep(1)
            continue

    return real_day, documents, day_ids


def main(args: Namespace):
    num_total_documents = 0

    crawled_metadata = []
    if args.resume:
        crawled_file_paths = glob(f"{args.save_dir}/*/*/*.json")
        for file_path in crawled_file_paths:
            king_name, real_year, real_month = re.match(f"{args.save_dir}/(.*)/(.*)/(.*).json", file_path).groups()
            crawled_metadata.append((king_name, real_year, real_month))

    king_list_soup = request_and_build_soup("https://sjw.history.go.kr/search/inspectionList.do")
    king_href_list = [item.find("td").find("a")["href"] for item in king_list_soup.find("tbody").find_all("tr")]

    king_month_metadata_list = []
    with Pool(args.num_workers) as pool:
        for king_month_metadata in pool.imap_unordered(crawl_king_page, king_href_list, chunksize=20):
            king_month_metadata_list.extend(king_month_metadata)

        if args.resume and len(crawled_metadata) > 0:
            king_month_metadata_list = [data for data in king_month_metadata_list if data[:3] not in crawled_metadata]
        king_month_metadata_list = sorted(king_month_metadata_list)

        progress = tqdm(total=len(king_month_metadata_list))
        for king_name, real_year, real_month, month_start_day_id in king_month_metadata_list:
            save_folder_dir = os.path.join(args.save_dir, king_name, real_year)
            os.makedirs(save_folder_dir, exist_ok=True)
            output_documents = {}

            real_day, documents, day_ids = crawl_day_page(king_name, month_start_day_id, return_day_ids=True)
            if real_day.endswith("미상"):
                real_day = "[미상]"

            if len(documents) > 0:
                num_total_documents += len(documents)
                output_documents[real_day] = documents

            crawl_day_page_fn = partial(crawl_day_page, king_name)
            for real_day, documents, _ in pool.imap_unordered(crawl_day_page_fn, day_ids):
                if len(documents) > 0:
                    num_total_documents += len(documents)
                    output_documents[real_day] = documents

            save_file_path = os.path.join(save_folder_dir, f"{real_month}.json")
            with open(save_file_path, "w") as f:
                json.dump(
                    {
                        "data_type": "jrs",
                        "king": king_name,
                        "real_year": real_year,
                        "month": real_month,
                        "documents": output_documents,
                    },
                    f,
                    ensure_ascii=False,
                )

            progress.set_postfix({"documents": num_total_documents})
            progress.update(1)


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args)
