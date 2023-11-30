import os, json 

from model.vqa_model import VQAModel
from trainer.trainer import VQATrainer

import torch

def init_model(model_kwargs:dict, trainer_kwargs:dict):

    '''
    #TODO
    load model from checkpoint code. 
    '''

    device = torch.device(trainer_kwargs["device"]) if torch.cuda.is_available() else torch.device("cpu")

    model = VQAModel(
        vision_model_name=model_kwargs["vision_model_name"],
        language_model_name=model_kwargs["language_model_name"],
        device=device 
    )

    model.to(device)

    return model

if __name__ == "__main__":

    config_json = json.load(open(
        'config.json'
    ))

    vqa_model = init_model(
        config_json["model_kwargs"],
        config_json["trainer_kwargs"]
    )
    
    trainer = VQATrainer(
        vqa_model, config_json["trainer_kwargs"],
        config_json["optimizer_kwargs"], config_json["lr_scheduler_kwargs"],
        config_json["callbacks_kwargs"], config_json["dataset_kwargs"]
    )

    trainer.train()
