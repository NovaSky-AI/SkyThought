from .aime.aime_handler import AIMETaskHandler
from .amc23.amc23_handler import AMC23TaskHandler
from .apps.apps_handler import APPSTaskHandler
from .arc.arc_handler import ARCChallengeTaskHandler
from .common import TaskHandler, TaskConfig
from .gpqa_diamond.gpqa_diamond_handler import GPQADiamondTaskHandler
from .gsm8k.gsm8k_handler import GSM8KTaskHandler
from .livecodebench.livecodebench_handler import LiveCodeBenchTaskHandler
from .math.math_handler import MathTaskHandler
from .mmlu.mmlu_handler import MMLUProTaskHandler, MMLUTaskHandler
from .numina.numina_handler import NUMINATaskHandler
from .taco.taco_handler import TACOTaskHandler

TASK_HANDLER_MAP = {
    "numina": NUMINATaskHandler,
    "apps": APPSTaskHandler,
    "taco": TACOTaskHandler,
    "math500": MathTaskHandler,
    "aime": AIMETaskHandler,
    "gpqa_diamond": GPQADiamondTaskHandler,
    "mmlu": MMLUTaskHandler,
    "mmlu_pro": MMLUProTaskHandler,
    "livecodebench": LiveCodeBenchTaskHandler,
    "gsm8k": GSM8KTaskHandler,
    "arc_c": ARCChallengeTaskHandler,
    "amc23": AMC23TaskHandler,
}

__all__ = [
    AIMETaskHandler,
    APPSTaskHandler,
    TACOTaskHandler,
    MathTaskHandler,
    AMC23TaskHandler,
    NUMINATaskHandler,
    GPQADiamondTaskHandler,
    MMLUTaskHandler,
    MMLUProTaskHandler,
    LiveCodeBenchTaskHandler,
    GSM8KTaskHandler,
    ARCChallengeTaskHandler,
    TaskHandler,
    MathTaskHandler,
    TASK_HANDLER_MAP,
]
