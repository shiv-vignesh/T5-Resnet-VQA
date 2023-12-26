import os, json 

from model.faster_rcnn_vqa_model import FasterRcnnVQAModel
from model.resnet_vqa_model import ResnetVQAModel
from trainer.faster_rcnn_vqa_trainer import FasterRcnnVQATrainer

import torch

def init_model(model_kwargs:dict, trainer_kwargs:dict, answer_spaces:list):

    device = torch.device(trainer_kwargs["device"]) if torch.cuda.is_available() else torch.device("cpu")

    if model_kwargs["vision_model_name"] == "faster-rcnn":

        model = FasterRcnnVQAModel(
            "faster-rcnn",
            model_kwargs["language_model_name"],
            answer_spaces=len(answer_spaces)
        )

        output_dir = trainer_kwargs["output_dir"]
        best_model_path = f'{output_dir}/model_checkpoints/best-model.pt'

        if os.path.exists(f'{best_model_path}'):
            print(f'Loading Best-Model.pt from {best_model_path}')
            model.load_state_dict(torch.load(best_model_path))

        model.to(device)

        return model

    if model_kwargs["vision_model_name"] == "resnet34" or model_kwargs["vision_model_name"] == model_kwargs["vision_model_name"] == "resnet18" or model_kwargs["vision_model_name"] == "resnet50":

        model = ResnetVQAModel(
            model_kwargs["vision_model_name"],
            model_kwargs["language_model_name"],
            answer_spaces=len(answer_spaces)
        )

        output_dir = trainer_kwargs["output_dir"]
        best_model_path = f'{output_dir}/model_checkpoints/best-model.pt'

        if os.path.exists(f'{best_model_path}'):
            print(f'Loading Best-Model.pt from {best_model_path}')
            model.load_state_dict(torch.load(best_model_path))

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

    # config_json["model_kwargs"]["vision_model_name"] = "resnet50"

    vqa_model = init_model(
        config_json["model_kwargs"],
        config_json["trainer_kwargs"], 
        answers_spaces
    )


    trainer = FasterRcnnVQATrainer(
        vqa_model, config_json["trainer_kwargs"],
        config_json["optimizer_kwargs"], config_json["lr_scheduler_kwargs"],
        config_json["callbacks_kwargs"], config_json["dataset_kwargs"]
    )

    trainer.train()
