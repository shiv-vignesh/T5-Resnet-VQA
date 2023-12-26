import os, json 

from model.vit_vqa_model import VitVQAModel
from trainer.vit_vqa_trainer import ViTVQATrainer

import torch

def init_model(model_kwargs:dict, trainer_kwargs:dict, answer_spaces:list):
    ''' 
    Initialize Model for training. 
    '''

    #check if cuda is available else use cpu
    device = torch.device(trainer_kwargs["device"]) if torch.cuda.is_available() else torch.device("cpu")

    model = VitVQAModel(
        model_kwargs["vision_model_name"], #t5-base
        model_kwargs["language_model_name"], #ViT 
        answer_spaces=len(answer_spaces)
    )

    model.to(device)

    return model

if __name__ == "__main__":

    config_json = json.load(open(
        'vit_daquar_config.json'
    ))

    root_dir = config_json["dataset_kwargs"]["root_data_dir"]
    answers_spaces = config_json["dataset_kwargs"]["answer_spaces_file"]
    answers_spaces = open(f'{root_dir}/{answers_spaces}').readlines()
    answers_spaces = [ans.replace("\n","") for ans in answers_spaces]

    vqa_model = init_model(
        config_json["model_kwargs"],
        config_json["trainer_kwargs"], 
        answers_spaces
    )

    trainer = ViTVQATrainer(
        vqa_model, config_json["trainer_kwargs"],
        config_json["optimizer_kwargs"], config_json["lr_scheduler_kwargs"],
        config_json["callbacks_kwargs"], config_json["dataset_kwargs"]
    )

    trainer.train()
