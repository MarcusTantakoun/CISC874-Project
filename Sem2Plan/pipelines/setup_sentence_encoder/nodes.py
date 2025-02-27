from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer
from better_leveraging_llm_to_construct_world_models.prompt_template.main_prompting_template import get_llm_input_dict
from better_leveraging_llm_to_construct_world_models.utils.pddl_parser import get_action_schema_answer_str, get_domain_model_from_name
from icecream import ic

def create_sentence_encoder_helper(setup_sentence_encoder_cfg):
    model_name = setup_sentence_encoder_cfg['model_name']
    model_type = setup_sentence_encoder_cfg['model_type']

    device = setup_sentence_encoder_cfg['device']

    if model_type == "cross_encoder":
        model = CrossEncoder(model_name)
    elif model_type == "bi_encoder":
        model = SentenceTransformer(model_name)

    return model

