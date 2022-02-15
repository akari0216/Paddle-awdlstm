import logging
import datetime

def get_logger(path,log_name):
    today = str(datetime.date.today())
    #创建logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    #handler用于创建log文件
    handler = logging.FileHandler(("%s/%s-%s.txt") % (path,log_name,today))
    handler.setLevel(level = logging.INFO)
    #formatter用于设置log日志格式
    formatter = logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s",datefmt = "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger