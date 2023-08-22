"""
Base type of feigen
"""
from feigen import log


class FeigenBase:
    __slots__ = ()

    def __init_subclass__(cls, *args, **kwargs):
        """
        Add logger shortcut.
        """
        super().__init_subclass__(*args, **kwargs)
        cls._logi = log.prepend_log("<" + cls.__qualname__ + ">", log.info)
        cls._logd = log.prepend_log("<" + cls.__qualname__ + ">", log.debug)
        cls._logw = log.prepend_log("<" + cls.__qualname__ + ">", log.warning)
        return super().__new__(cls)
