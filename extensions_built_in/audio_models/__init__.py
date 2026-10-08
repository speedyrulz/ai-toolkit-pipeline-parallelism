from .ace_step import AceStep15Model, AceStep15XLModel
from .yue2 import YuE2AudioModel
from .omnivoice import OmniVoiceModel
from .qwen3_tts import Qwen3TTSTrainModel

AI_TOOLKIT_MODELS = [
    # put a list of models here
    AceStep15Model,
    AceStep15XLModel,
    YuE2AudioModel,
    OmniVoiceModel,
    Qwen3TTSTrainModel,
]
