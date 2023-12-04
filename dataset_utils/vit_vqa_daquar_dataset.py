import torch
import ast, os, random
import cv2

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset 
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor

from .enums import Enums

class Question: 
    def __init__(self, question_text:str, question_id:str, image_id:str):

        self.question_text = question_text
        self.question_id = question_id
        self.image_id = image_id

    def __str__(self) -> str:
        return f"Id: {self.question_id}, Text: {self.question_text}, Image_id: {self.image_id}"

class Annotation: 
    def __init__(self, question_id:str, image_id:str,
                 answers:list):
        
        self.question_id = question_id
        self.image_id = image_id

        answers = [answer.replace(" ","_") for answer in answers]
        self.answers = answers

    def __str__(self) -> str:
        return f"Question-Id: {self.question_id}, Length of Answers: {len(self.answers)}, Question-Type: {self.question_type}"

class DaquarDataset(Dataset):

    def __init__(
                self,
                root_dir:str, 
                csv_file_path:str,
                images_dir:str,
                type:str 
                 ):
        

        self.data = pd.read_csv(f'{root_dir}/{csv_file_path}')
        self.images_dir = f'{root_dir}/{images_dir}'
        self.type = type

        self.images_fns = os.listdir(self.images_dir)
        self.image_ids_to_fn = {}

        for image_fn in self.images_fns:
            image_id = image_fn.split('.')[0]
            self.image_ids_to_fn[image_id] = image_fn

    def __getitem__(self, idx):

        data_points = self.data[idx:idx+1]
        image_id = data_points["image_id"].item()

        question = Question(
            data_points["question"].item(),
            f'{data_points["image_id"].item()}_{idx}_Question',
            image_id            
        )
        
        annotation = Annotation(
            f'{data_points["image_id"].item()}__{idx}_Question',
            image_id,
            ast.literal_eval(data_points["answers_list"].item())
        )

        image_id = question.image_id
        image_fn = self.image_ids_to_fn[image_id]

        return {
            "question": question,
            "annotation":annotation,
            "image_path":f'{self.images_dir}/{image_fn}'
        }

    def __len__(self):
        return len(self.data) 
    

class DaquarVitT5CollateFn(object):
    def __init__(self, 
                image_model:str, 
                lang_model:str,
                answer_spaces:list, 
                eval_mode:bool=False
                 ):    
        
        self.tokenizer = AutoTokenizer.from_pretrained(lang_model)
        self.image_preprocessor = AutoImageProcessor.from_pretrained(image_model)

        self.eval_mode = eval_mode
        self.answer_spaces = answer_spaces
        self.answer_spaces = [answer.strip('\n') for answer in self.answer_spaces]

        self.tokenizer.add_special_tokens({
            "additional_special_tokens":[Enums.QUESTION_SPECIAL_TOKEN, Enums.ANSWER_SPECIAL_TOKEN, Enums.QUESTION_TYPE_SPECIAL_TOKEN]
        })

        question_types = [qt.replace(" ","_") for qt in Enums.QUESTION_TYPES.values()]

        self.tokenizer.add_special_tokens({
            "additional_special_tokens":question_types
        })


    def collect_preprocessed_data(self, data_points):

        questions = [data["question"] for data in data_points]
        annotations = [data["annotation"] for data in data_points]
        images_file_paths = [data["image_path"] for data in data_points]

        image_arrs = [cv2.imread(image_fp) for image_fp in images_file_paths]
        image_arrs = [cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB) for image_arr in image_arrs]

        pixel_values = self.image_preprocessor(
                images = image_arrs, return_tensors = "pt"
                )['pixel_values']
                        
        annotations_ids = []  

        random_answer_choices = []

        for annotation in annotations:
            ''' 
            randomly select a single choice from list of answers
            '''
            answer = random.choice(annotation.answers)
            random_answer_choices.append(self.answer_spaces.index(answer))


        annotations_ids = torch.tensor(random_answer_choices)
        question_texts = [f'{Enums.QUESTION_SPECIAL_TOKEN} {question.question_text}' for idx, question in enumerate(questions)]

        question_tensors = self.tokenizer(question_texts, return_tensors="pt", padding="longest")         

        decoder_question_texts = [f'{Enums.QUESTION_SPECIAL_TOKEN} {question.question_text} {Enums.ANSWER_SPECIAL_TOKEN}' for idx, question in enumerate(questions)]
        decoder_question_tensors = self.tokenizer(decoder_question_texts, return_tensors="pt", padding="longest")         

        if self.eval_mode:
            answers = [annotation.answers for annotation in annotations]
            return {
                "question_input_ids":question_tensors["input_ids"],
                "decoder_question_input_ids":decoder_question_tensors["input_ids"],
                "question_attention_masks":question_tensors["attention_mask"],
                "decoder_question_attention_masks":decoder_question_tensors["attention_mask"],
                "annotation_ids":annotations_ids, #list(tensors)
                "pixel_values":pixel_values,
                "question_type_ids":None,
                "answers":answers,
                "questions":questions
            }


        return {
            "question_input_ids":question_tensors["input_ids"],
            "decoder_question_input_ids":decoder_question_tensors["input_ids"],
            "question_attention_masks":question_tensors["attention_mask"],
            "decoder_question_attention_masks":decoder_question_tensors["attention_mask"],
            "annotation_ids":annotations_ids,
            "pixel_values":pixel_values,
            "question_type_ids":None,
        }

    def __call__(self, data_points):

        return self.collect_preprocessed_data(data_points)

