import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger


def log_exception_info(**logging_items):
    # Get current system exception
    ex_type, ex_value, ex_traceback = sys.exc_info()
    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(ex_traceback)
    # Format stacktrace
    stack_trace = [
        f"File: {trace[0]} , Line: {trace[1]}, Func.Name: {trace[2]}, Message: {trace[3]}" for trace in trace_back
    ]

    fname = os.path.split(ex_traceback.tb_frame.f_code.co_filename)[1]
    logger.error(f"Exception type : {ex_type.__name__}")
    logger.error(f"Exception message : {ex_value}")
    logger.error(f"Stack trace : {stack_trace}")
    # log ex_type, fname, ex_traceback.tb_lineno
    logger.error(f"Exception type: {ex_type}, File name: {fname}, Line number: {ex_traceback.tb_lineno}")
    for key, value in logging_items.items():
        logger.error(f"{key}: {value}")


def request_and_build_soup(url, request_mode: str = "post", data: Optional[Dict[str, Any]] = None) -> BeautifulSoup:
    while True:
        try:
            if request_mode == "post":
                res = requests.post(url, data=data, verify=False)
            elif request_mode == "get":
                res = requests.get(url, verify=False)
            else:
                raise ValueError("Invalid request mode!")
            break
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError):
            logger.exception(f"request error - {url}, {data}")
            time.sleep(10)
        except Exception as e:
            logger.exception(e)
        continue

    soup = BeautifulSoup(res.text, features="lxml")
    return soup
