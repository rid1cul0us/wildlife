import logging
import os.path
import time


class Logger(logging.Logger):
    def __init__(
        self,
        logger_name: str,
        log_file_path: str = None,
        file_log_level: int = logging.INFO,
        console_log_level: int = logging.DEBUG,
        enable=0,
    ):
        """_summary_

        Args:
            logger_name (str, optional): _description_.
            log_file_path (str, optional): _description_. Defaults to None.
            file_log_level (int, optional): _description_. Defaults to logging.INFO.
            console_log_level (int, optional): _description_. Defaults to logging.DEBUG.
            enable (int optional): _description_. Defaults to 0 to enable.
        """

        super().__init__(logger_name)

        if enable != 0:
            self.__logger = None
            return

        # 创建一个logger
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(logging.DEBUG)
        """
        CRITICAL = 50
        FATAL = CRITICAL
        ERROR = 40
        WARNING = 30
        WARN = WARNING
        INFO = 20
        DEBUG = 10
        NOTSET = 0
        """

        # 输出格式
        formatter = logging.Formatter(
            "[%(asctime)s] [file:%(pathname)s]\t\t\t[line: %(lineno)d]\t[%(levelname)s]\t==> %(message)s"
        )
        """
        %(levelno)s:打印日志级别的数值
        %(levelname)s:打印日志级别名称
        %(pathname)s:打印当前执行程序的路径，sys.argv[0]
        %(filename)s:打印当前执行程序名
        %(funcName)s:打印日志的当前函数
        %(lineno)d:打印日志的当前行号
        %(asctime)s:打印日志的时间
        %(thread)d:打印线程ID
        %(threadName)s:打印线程名称
        %(process)d:打印进程ID
        %(message)s:打印日志信息
        """

        # 创建一个handler，用于写入日志文件
        if log_file_path is None:
            rq = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
            log_dir_path = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
            if not os.path.exists(log_dir_path):
                os.mkdir(log_dir_path)
            log_file_path = log_dir_path + self.__logger.name + "_" + rq + ".log"
        self.log_file_path = log_file_path
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)

        # 再创建一个handler，用于输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        # console_handler.setFormatter(formatter)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.__logger.addHandler(console_handler)

        self.__logger.info("\n\nstart logging {}...".format(self.log_file_path))

    def debug(self, msg, *args):
        if self.__logger:
            self.__logger.debug(msg, *args)

    def info(self, msg, *args):
        if self.__logger:
            self.__logger.info(msg, *args)

    def error(self, msg, *args):
        if self.__logger:
            self.__logger.error(msg, *args)


if __name__ == "__main__":
    log = Logger(os.path.basename(__file__).split(".")[0])
    log.debug("debug")
    log.info("info")
    log.error("error")
