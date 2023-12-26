import os
import numpy as np
import torch 
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.transforms as transforms
from torch.utils.data import Dataset 

import transformers
from transformers import AutoTokenizer

from .enums import Enums

class Question: 

    def __init__(self, question_text:str, question_id:int, image_id:int):

        self.question_text = question_text
        self.question_id = question_id
        self.image_id = image_id

    def __str__(self) -> str:
        return f"Id: {self.question_id}, Text: {self.question_text}, Image_id: {self.image_id}"

class Annotation: 

    def __init__(self, question_id:int, image_id:int, question_type:str,
                 answers:list, answer_type:str):
        
        self.question_id = question_id
        self.image_id = image_id
        self.question_type = question_type
        self.answers = answers
        self.answer_type = answer_type

    def __str__(self) -> str:
        return f"Question-Id: {self.question_id}, Length of Answers: {len(self.answers)}, Question-Type: {self.question_type}"


class VQADataset(Dataset):

    def __init__(self, 
                annotations_json:dict,
                questions_json:dict,
                images_dir:str,
                type:str
                # image_ids_to_fn:dict
                ):     

        self.images_dir = images_dir
        self.type = type
        self.image_ids_to_fn = {}
        self.load_data(annotations_json, questions_json, images_dir)

    def load_data(self, annotations_json:dict, questions_json:dict, images_dir:str):
        
        self.questions = questions_json["questions"]
        self.annotations = annotations_json["annotations"]
        self.images_fns = os.listdir(images_dir)

        for image_fn in self.image_fns:
            if self.type == "train":
                image_id = image_fn.split('COCO_train2014_')[1].lstrip('0').split('.')[0]

            elif self.type == "val":
                image_id = image_fn.split('COCO_val2014_')[1].lstrip('0').split('.')[0]

            self.image_ids_to_fn[int(image_id)] = image_fn

    def __getitem__(self, idx):

        question = self.questions[idx]
        annotation = self.annotations[idx]

        question = Question(
            question["question"],
            question["question_id"],
            question["image_id"]
        )

        annotation = Annotation(
            annotation["question_id"],annotation["image_id"], annotation["question_type"],
            annotation["answers"], annotation["answer_type"]
        )

        image_id = question.image_id
        image_fn = self.image_ids_to_fn[image_id]

        return {
            "question": question,
            "annotation":annotation,
            "image_path":f'{self.images_dir}/{image_fn}'
        }

    def __len__(self):
        return len(self.questions)   
    

class BatchCollateFn(object):

    def __init__(self, 
                resizing_dimensions:tuple,
                interpolation_strategy:str,
                image_transforms:list,
                lang_model:str,
                eval_mode:bool=False
                 ):

        self.resizing_width, self.resizing_height = resizing_dimensions
        self.interpolation_strategy = interpolation_strategy

        self.create_transforms(image_transforms)

        self.tokenizer = AutoTokenizer.from_pretrained(lang_model)
        self.tokenizer.add_special_tokens({
            "additional_special_tokens":[Enums.QUESTION_SPECIAL_TOKEN, Enums.CONTEXT_SPECIAL_TOKEN, Enums.QUESTION_TYPE_SPECIAL_TOKEN]
        })

        self.eval_mode = eval_mode

        
    def create_transforms(self, image_transforms:list):        
        transforms_techniques = [Enums.TRANSFORM_STRATEGIES[transform] for transform in image_transforms]

        self.image_transformations = A.Compose(transforms_techniques)

    def collect_preprocessed_data(self, data_points):

        questions = [data["question"] for data in data_points]
        annotations = [data["annotation"] for data in data_points]
        images_file_paths = [data["image_path"] for data in data_points]

        image_tensors = []

        for fp in images_file_paths:
            image_arr = cv2.imread(fp)
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)

            if self.interpolation_strategy == "bilinear_interpolation":
                image_arr = cv2.resize(image_arr, (self.resizing_width, self.resizing_height), interpolation=cv2.INTER_LINEAR)
            
            elif self.interpolation_strategy == "lanczos_interpolation":
                image_arr = cv2.resize(image_arr, (self.resizing_width, self.resizing_height), interpolation=cv2.INTER_LANCZOS4)

            elif self.interpolation_strategy == "bicubic_interpolation":
                image_arr = cv2.resize(image_arr, (self.resizing_width, self.resizing_height), interpolation=cv2.INTER_CUBIC)

            image_tensor = self.image_transformations(image=image_arr)["image"]

            image_tensors.append(image_tensor)

        image_tensors = torch.stack(image_tensors, dim=0)
        ''' 
        Uncomment for just question as input to the T5-encoder.
        '''
        # question_texts = [f'{Enums.QUESTION_SPECIAL_TOKEN} {question.question_text}' for question in questions]
        # question_tensors = self.tokenizer(question_texts, return_tensors="pt", padding="longest") 
                
        annotations_ids = [] 
        question_type_ids = []

        question_types = []

        for annotation in annotations:
            answers = annotation.answers
            answers = [answer["answer"] for answer in answers]
            answer_input_ids = self.tokenizer(answers, return_tensors="pt", padding="max_length", truncation=True, max_length=Enums.MAX_LEN)["input_ids"]
            annotations_ids.append(answer_input_ids)

            question_type = annotation.question_type
            question_types.append(question_type)
            question_type_id = Enums.QUESTION_TYPE_TO_IDS[question_type]
            question_type_ids.append(question_type_id)

        annotations_ids = torch.stack(annotations_ids, dim=0) #[bs, 10, 512]; 10 - 10 answers per question and 512 is the max generative len. Each of 0-512 is a token id. 
        question_type_ids = torch.tensor(question_type_ids)

        question_texts = [f'{Enums.QUESTION_SPECIAL_TOKEN} {question.question_text} {Enums.QUESTION_TYPE_SPECIAL_TOKEN} {question_types[idx]}' for idx, question in enumerate(questions)]
        question_tensors = self.tokenizer(question_texts, return_tensors="pt", padding="longest")         

        if self.eval_mode:
            answers = [annotation.answers for annotation in annotations]
            return {
                "question_input_ids":question_tensors["input_ids"],
                "question_attention_masks":question_tensors["attention_mask"],
                "annotation_ids":annotations_ids,
                "image_tensors":image_tensors,
                "question_type_ids":question_type_ids,
                "answers":answers,
                "questions":questions
            }


        return {
            "question_input_ids":question_tensors["input_ids"],
            "question_attention_masks":question_tensors["attention_mask"],
            "annotation_ids":annotations_ids,
            "image_tensors":image_tensors,
            "question_type_ids":question_type_ids
        }

    def __call__(self, data_points):

        return self.collect_preprocessed_data(data_points)
        