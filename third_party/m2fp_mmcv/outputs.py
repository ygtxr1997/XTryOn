from typing import Dict, List, Tuple

import numpy as np

class OutputKeys(object):
    LOSS = 'loss'
    LOGITS = 'logits'
    SCORES = 'scores'
    SCORE = 'score'
    LABEL = 'label'
    LABELS = 'labels'
    INPUT_IDS = 'input_ids'
    LABEL_POS = 'label_pos'
    POSES = 'poses'
    CAPTION = 'caption'
    BOXES = 'boxes'
    KEYPOINTS = 'keypoints'
    MASKS = 'masks'
    DEPTHS = 'depths'
    DEPTHS_COLOR = 'depths_color'
    LAYOUT = 'layout'
    TEXT = 'text'
    POLYGONS = 'polygons'
    OUTPUT = 'output'
    OUTPUT_IMG = 'output_img'
    OUTPUT_IMGS = 'output_imgs'
    OUTPUT_VIDEO = 'output_video'
    OUTPUT_PCM = 'output_pcm'
    OUTPUT_PCM_LIST = 'output_pcm_list'
    OUTPUT_WAV = 'output_wav'
    OUTPUT_OBJ = 'output_obj'
    OUTPUT_MESH = 'output_mesh'
    IMG_EMBEDDING = 'img_embedding'
    SPK_EMBEDDING = 'spk_embedding'
    SPO_LIST = 'spo_list'
    TEXT_EMBEDDING = 'text_embedding'
    TRANSLATION = 'translation'
    RESPONSE = 'response'
    PREDICTION = 'prediction'
    PREDICTIONS = 'predictions'
    PROBABILITIES = 'probabilities'
    DIALOG_STATES = 'dialog_states'
    VIDEO_EMBEDDING = 'video_embedding'
    UUID = 'uuid'
    WORD = 'word'
    KWS_LIST = 'kws_list'
    SQL_STRING = 'sql_string'
    SQL_QUERY = 'sql_query'
    HISTORY = 'history'
    QUERY_RESULT = 'query_result'
    TIMESTAMPS = 'timestamps'
    SHOT_NUM = 'shot_num'
    SCENE_NUM = 'scene_num'
    SCENE_META_LIST = 'scene_meta_list'
    SHOT_META_LIST = 'shot_meta_list'
    MATCHES = 'matches'
    PCD12 = 'pcd12'
    PCD12_ALIGN = 'pcd12_align'
    TBOUNDS = 'tbounds'


OutputTypes = {
    OutputKeys.LOSS: float,  # checked
    OutputKeys.LOGITS: np.ndarray,  # checked.
    OutputKeys.SCORES: List[float],  # checked
    OutputKeys.SCORE: float,  # checked
    OutputKeys.LABEL: str,  # checked
    OutputKeys.LABELS: List[str],  # checked
    OutputKeys.INPUT_IDS: np.ndarray,  # checked
    OutputKeys.LABEL_POS: np.ndarray,  # checked
    OutputKeys.POSES:
    List[np.ndarray],  # [Tuple(np.ndarray, np.ndarray)]  # checked doubtful
    OutputKeys.CAPTION: str,
    OutputKeys.BOXES: np.ndarray,  # checked
    OutputKeys.KEYPOINTS: np.ndarray,  # checked
    OutputKeys.MASKS: np.ndarray,  # checked
    OutputKeys.DEPTHS: List[np.ndarray],  # checked
    OutputKeys.DEPTHS_COLOR: List[np.ndarray],  # checked
    OutputKeys.LAYOUT: np.ndarray,  # checked
    OutputKeys.TEXT: str,  # checked
    OutputKeys.POLYGONS: np.array,  # checked
    OutputKeys.OUTPUT: Dict,
    OutputKeys.OUTPUT_IMG: 'image',  # checked
    OutputKeys.OUTPUT_IMGS: List[np.ndarray],  # checked
    OutputKeys.OUTPUT_VIDEO: 'bytes',
    OutputKeys.OUTPUT_PCM: np.ndarray,
    OutputKeys.OUTPUT_PCM_LIST: List[np.ndarray],
    OutputKeys.OUTPUT_WAV: np.ndarray,
    OutputKeys.OUTPUT_OBJ: Dict,
    OutputKeys.OUTPUT_MESH: np.ndarray,
    OutputKeys.IMG_EMBEDDING: np.ndarray,
    OutputKeys.SPK_EMBEDDING: np.ndarray,
    OutputKeys.SPO_LIST: List[float],
    OutputKeys.TEXT_EMBEDDING: np.ndarray,
    OutputKeys.TRANSLATION: str,
    OutputKeys.RESPONSE: Dict,
    OutputKeys.PREDICTION: np.ndarray,  # checked
    OutputKeys.PREDICTIONS: List[np.ndarray],
    OutputKeys.PROBABILITIES: np.ndarray,
    OutputKeys.DIALOG_STATES: object,
    OutputKeys.VIDEO_EMBEDDING: np.ndarray,
    OutputKeys.UUID: str,
    OutputKeys.WORD: str,
    OutputKeys.KWS_LIST: List[str],
    OutputKeys.SQL_STRING: str,  # checked
    OutputKeys.SQL_QUERY: str,  # checked
    OutputKeys.HISTORY: Dict,  # checked
    OutputKeys.QUERY_RESULT: Dict,  # checked
    OutputKeys.TIMESTAMPS: str,
    OutputKeys.SHOT_NUM: int,
    OutputKeys.SCENE_NUM: int,
    OutputKeys.SCENE_META_LIST: List[int],
    OutputKeys.SHOT_META_LIST: List[int],
    OutputKeys.MATCHES: List[np.ndarray],
    OutputKeys.PCD12: np.ndarray,
    OutputKeys.PCD12_ALIGN: np.ndarray,
    OutputKeys.TBOUNDS: Dict,
}

OutputTypeSchema = {
    OutputKeys.LOSS: {
        'type': 'number'
    },  # checked
    OutputKeys.LOGITS: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },  # checked.
    OutputKeys.SCORES: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },  # checked
    OutputKeys.SCORE: {
        'type': 'number'
    },  # checked
    OutputKeys.LABEL: {
        'type': 'string'
    },  # checked
    OutputKeys.LABELS: {
        'type': 'array',
        'items': {
            'type': 'string'
        }
    },  # checked
    OutputKeys.INPUT_IDS: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },  # checked
    OutputKeys.LABEL_POS: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },  # checked
    OutputKeys.POSES: {
        'type': 'array',
        'items': {
            'type': 'array',
            'items': {
                'type': 'number'
            }
        }
    },  # [Tuple(np.ndarray, np.ndarray)]  # checked doubtful
    OutputKeys.CAPTION: {
        'type': 'string'
    },
    OutputKeys.BOXES: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },  # checked
    OutputKeys.KEYPOINTS: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },  # checked
    OutputKeys.MASKS: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },  # checked
    OutputKeys.DEPTHS: {
        'type': 'array',
        'items': {
            'type': 'array',
            'items': {
                'type': 'number'
            }
        }
    },  # checked
    OutputKeys.DEPTHS_COLOR: {
        'type': 'array',
        'items': {
            'type': 'array',
            'items': {
                'type': 'number'
            }
        }
    },  # checked
    OutputKeys.LAYOUT: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },  # checked
    OutputKeys.TEXT: {
        'type': 'string'
    },  # checked
    OutputKeys.POLYGONS: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },  # checked
    OutputKeys.OUTPUT: {
        'type': 'object'
    },
    OutputKeys.OUTPUT_IMG: {
        'type': 'string',
        'description': 'The base64 encoded image.',
    },  # checked
    OutputKeys.OUTPUT_IMGS: {
        'type': 'array',
        'items': {
            'type': 'string',
            'description': 'The base64 encoded image.',
        }
    },  # checked
    OutputKeys.OUTPUT_VIDEO: {
        'type': 'string',
        'description': 'The base64 encoded video.',
    },
    OutputKeys.OUTPUT_PCM: {
        'type': 'string',
        'description': 'The base64 encoded PCM.',
    },
    OutputKeys.OUTPUT_PCM_LIST: {
        'type': 'array',
        'items': {
            'type': 'string',
            'description': 'The base64 encoded PCM.',
        }
    },
    OutputKeys.OUTPUT_WAV: {
        'type': 'string',
        'description': 'The base64 encoded WAV.',
    },
    OutputKeys.OUTPUT_OBJ: {
        'type': 'object'
    },
    OutputKeys.OUTPUT_MESH: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },
    OutputKeys.IMG_EMBEDDING: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },
    OutputKeys.SPK_EMBEDDING: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },
    OutputKeys.SPO_LIST: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },
    OutputKeys.TEXT_EMBEDDING: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },
    OutputKeys.TRANSLATION: {
        'type': 'string'
    },
    OutputKeys.RESPONSE: {
        'type': 'object'
    },
    OutputKeys.PREDICTION: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },  # checked
    OutputKeys.PREDICTIONS: {
        'type': 'array',
        'items': {
            'type': 'array',
            'items': {
                'type': 'number'
            }
        }
    },
    OutputKeys.PROBABILITIES: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },
    OutputKeys.DIALOG_STATES: {
        'type': 'object'
    },
    OutputKeys.VIDEO_EMBEDDING: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },
    OutputKeys.UUID: {
        'type': 'string'
    },
    OutputKeys.WORD: {
        'type': 'string'
    },
    OutputKeys.KWS_LIST: {
        'type': 'array',
        'items': {
            'type': 'string'
        }
    },
    OutputKeys.SQL_STRING: {
        'type': 'string'
    },  # checked
    OutputKeys.SQL_QUERY: {
        'type': 'string'
    },  # checked
    OutputKeys.HISTORY: {
        'type': 'object'
    },  # checked
    OutputKeys.QUERY_RESULT: {
        'type': 'object'
    },  # checked
    OutputKeys.TIMESTAMPS: {
        'type': 'string'
    },
    OutputKeys.SHOT_NUM: {
        'type': 'integer'
    },
    OutputKeys.SCENE_NUM: {
        'type': 'integer'
    },
    OutputKeys.SCENE_META_LIST: {
        'type': 'array',
        'items': {
            'type': 'integer'
        }
    },
    OutputKeys.SHOT_META_LIST: {
        'type': 'array',
        'items': {
            'type': 'integer'
        }
    },
    OutputKeys.MATCHES: {
        'type': 'array',
        'items': {
            'type': 'array',
            'items': {
                'type': 'number'
            }
        }
    },
    OutputKeys.PCD12: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },
    OutputKeys.PCD12_ALIGN: {
        'type': 'array',
        'items': {
            'type': 'number'
        }
    },
    OutputKeys.TBOUNDS: {
        'type': 'object'
    },
}