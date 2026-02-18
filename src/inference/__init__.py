from src.inference.entity_utils import NEREntity, post_process_entities

try:
    from src.inference.predictor import NERPredictor
except ImportError:
    pass
