import collections
import io
import pickle
import zipfile

from typing import Any

from ...type_alist import PathLike

class DroppedObject:
    """
    哨兵值：静默吸收所有参数和状态，代表被丢弃的不安全对象。
    采用单例模式，所有被丢弃的对象都将指向这同一个实例。
    """
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "DroppedObject":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # 吸收所有 pickle 传入的构造参数
        pass
    
    def __setstate__(self, state: Any) -> None:
        # 吸收所有 pickle 传入的属性状态
        pass

    def __repr__(self) -> str:
        return "<DroppedObject>"

    def __bool__(self) -> bool:
        # 方便在 if 判断中识别：if val is DROPPED: ...
        return False
    
class SafeStandardUnpickler(pickle.Unpickler):
    """遇到非白名单类型时，静默替换为 DroppedObject 哨兵"""
    
    SAFE_CLASSES = {
        'builtins': {
            v.__name__: v for v in [
                type(None), type(...), type(NotImplemented),
                int, float, complex, bool, str, bytes, bytearray, memoryview,
                list, tuple, dict, set, frozenset, range, slice, object
            ]
        },
        'collections': {
            'OrderedDict': collections.OrderedDict,
            'defaultdict': collections.defaultdict,
            'deque': collections.deque,
            'Counter': collections.Counter,
        },
    }

    def find_class(self, module: str, name: str):
        if module in self.SAFE_CLASSES and name in self.SAFE_CLASSES[module]:
            return self.SAFE_CLASSES[module][name]
        
        # 静默丢弃：返回哨兵类，不抛出异常
        return DroppedObject
    
    def persistent_load(self, pid: Any) -> DroppedObject:
        """
        PyTorch 用 persistent_id 机制存储 tensor 的 storage。
        我们不需要 tensor 数据，直接返回哨兵值。
        """
        return DroppedObject()
    
def safe_load_pth(filepath: PathLike) -> Any:
    """
    安全加载 PyTorch .pth/.pt 文件。
    自动处理 ZIP 格式（PyTorch 1.6+）和旧版原始 pickle 格式。
    """
    with open(filepath, 'rb') as f:
        magic = f.read(2)
        f.seek(0)

        if magic == b'PK':
            # ===== ZIP 格式（PyTorch 1.6+）=====
            # .pth 文件本质是一个 ZIP，里面包含:
            #   archive/data.pkl  ← 结构信息（我们要的）
            #   archive/data/0    ← tensor 原始数据（不需要）
            with zipfile.ZipFile(filepath, 'r') as zf:
                # 找到 .pkl 文件
                pkl_names = [n for n in zf.namelist()
                             if n.endswith('.pkl')]
                if not pkl_names:
                    raise ValueError(
                        f"ZIP 格式的 .pth 文件中未找到 .pkl 文件，"
                        f"包含的文件: {zf.namelist()}"
                    )
                pkl_data = zf.read(pkl_names[0])
                return SafeStandardUnpickler(io.BytesIO(pkl_data)).load()
        else:
            # ===== 旧版原始 pickle 格式 =====
            return SafeStandardUnpickler(f).load()

def direct_read_sr(model: PathLike) -> int:
    loaded_data = safe_load_pth(model)
    config = loaded_data.get("config")
    if config is None:
        raise ValueError("找不到 config")
    if len(config) < 18: 
        raise ValueError("config 长度不足 18")
    sr = config[17]
    if not isinstance(sr, int):
        raise ValueError("采样率不是整数")
    return sr