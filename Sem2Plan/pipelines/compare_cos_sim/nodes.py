
from copy import deepcopy
import json
import random
from typing import Sequence
from tqdm.auto import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 
from pathlib import Path
import os 
from glob import glob 
from pddl.parser.domain import DomainParser
from enum import Enum
from pddl.logic import Predicate, Constant, Variable
from pddl.logic.base import And, Not, Or, BinaryOp, UnaryOp
from pddl.core import Domain, Problem, Action, Requirements, Formula
from pddl.logic.effects import AndEffect

from utils.import_py import import_from_filepath
from utils.pddl_manipulation import get_manipulated_action_lst
from utils.pddl_parser import get_action_schema_answer_str



def local_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class PAIR_TYPE(Enum):
    POSITIVE = "Positive (Correct Match)"
    EASY_NEGATIVE = "Easy Negative (Inter-Domain Mismatch)"
    SEMI_HARD_NEGATIVE = "Semi-Hard Negative (Intra-Domain Mismatch)"
    HARD_NEGATIVE = "Hard Negative (Manipulated Action)"

class COMPARING_SETTING(Enum):
    NL_DESC_VS_ACTION_SCHEMA = "NL Desc vs. Action Schema"
    EXPLANATION_VS_ACTION_SCHEMA = "Explanation vs. Action Schema" # the Explanation here is the query explanation, which means if the action is negative, the explanation will not fit the action schema
    NL_DESC_VS_EXPLANATION_ACTION_SCHEMA = "NL Desc vs. Explanation + Action Schema"
    # the explanation here is the answer explanation, which means if the action is negative, the explanation will also be negative that fits the action schema


def analyzing_cos_sim_between_action_and_nl_desc(cross_encoder_model, setup_sentence_encoder_cfg):
    def local_comparing_setting_helper(
            answer_action_lst: Sequence[Action],
            query_content_dict: dict,
            comparing_setting: COMPARING_SETTING,
            pair_type_str: str,
            query_action_schema_str: str,
            query_explanation: str,
            answer_explanation_lst: Sequence[str],
            query_domain_name,
            query_action_name,
            df: pd.DataFrame,
            if_complex_context: bool = False,
            **kwargs,
    ):
        nonlocal cross_encoder_model
        nonlocal setup_sentence_encoder_cfg

        alternative_embedding_model = None
        if "activate_alternative_embedding" in setup_sentence_encoder_cfg:
            if setup_sentence_encoder_cfg["activate_alternative_embedding"]:
                alternative_embedding_model = setup_sentence_encoder_cfg['alternative_embedding_model']

        if "manipulation_details_lst" in kwargs:
            manipulation_details_lst = kwargs["manipulation_details_lst"]
        else:
            manipulation_details_lst = None

        # we will update the dataframe and return the updated dataframe
        if comparing_setting == COMPARING_SETTING.NL_DESC_VS_ACTION_SCHEMA:
            query_content = query_content_dict['context'] if if_complex_context else SIMPLER_PROMPT_CONTEXT + query_content_dict['query']
        elif comparing_setting == COMPARING_SETTING.EXPLANATION_VS_ACTION_SCHEMA:
            query_content =  query_content_dict['context'] if if_complex_context else SIMPLER_PROMPT_CONTEXT + query_content_dict['query'] + "\n" + query_explanation
        elif comparing_setting == COMPARING_SETTING.NL_DESC_VS_EXPLANATION_ACTION_SCHEMA:
            query_content = query_content_dict['context'] if if_complex_context else SIMPLER_PROMPT_CONTEXT + query_content_dict['query']

        answer_content_lst = []
        for answer_action, answer_explanation in zip(answer_action_lst, answer_explanation_lst):
            if comparing_setting == COMPARING_SETTING.NL_DESC_VS_ACTION_SCHEMA:
                answer_content = get_action_schema_answer_str(answer_action)
            elif comparing_setting == COMPARING_SETTING.EXPLANATION_VS_ACTION_SCHEMA:
                answer_content = get_action_schema_answer_str(answer_action)
            elif comparing_setting == COMPARING_SETTING.NL_DESC_VS_EXPLANATION_ACTION_SCHEMA:
                answer_content = answer_explanation + "\n" + get_action_schema_answer_str(answer_action)

            answer_content_lst.append(answer_content)

        if alternative_embedding_model is None:
            query_embedding = cross_encoder_model.encode(query_content, convert_to_tensor=True)
            corpus_embeddings = cross_encoder_model.encode(answer_content_lst, convert_to_tensor=True)
            similarity_scores = cross_encoder_model.similarity(query_embedding, corpus_embeddings).detach().cpu().numpy().tolist()[0]

        new_row_dict = {
            "Domain": [query_domain_name] * len(answer_action_lst),
            "Action Name": [query_action_name] * len(answer_action_lst),
            "Cosine Sim Score": similarity_scores,
            "Pair Type": [pair_type_str] * len(answer_action_lst),
            "Comparison Mode": [comparing_setting.value] * len(answer_action_lst),
            "Query Content": [query_content] * len(answer_action_lst),
            "Answer Content": answer_content_lst,
            "Query Action Schema": [query_action_schema_str] * len(answer_action_lst),
            "Manipulation Details": manipulation_details_lst,
        }
        df = pd.concat([df, pd.DataFrame(new_row_dict)], ignore_index=True)

        return df