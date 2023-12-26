import torch
import ast, os, random
import cv2

import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms

from torch.utils.data import Dataset 
from transformers import AutoTokenizer

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
    

class DaquarFasterRcnnT5CollateFn(object):
    '''
    Custom Dataloader Class to preprocess DAQUAR dataset inputs for CNN based Multi-Modal Pipeline.
    '''
    def __init__(self,  
                interpolation_strategy:str,
                resizing_dimensions:tuple,
                lang_model:str,
                answer_spaces:list, 
                image_transforms,
                eval_mode:bool=False
                 ):    
        
        ''' 
        interpolation_strategy -  method used for image interpolation. Interpolation is the technique used to estimate pixel values when resizing or transforming images.
        resizing_dimensions - tuple of (H, W)
        lang_model - To create instance of Tokenizer
        answer_spaces - Unique Answer Spaces for Prediction
        eval_mode - if True, returns Image_Fns, Question         
        '''

        self.tokenizer = AutoTokenizer.from_pretrained(lang_model)
        self.interpolation_strategy = interpolation_strategy
        
        self.eval_mode = eval_mode
        self.answer_spaces = answer_spaces
        self.answer_spaces = [answer.strip('\n') for answer in self.answer_spaces]

        # Add Special Token to Tokenizer. While tokenizing inputs it will treat SPECIAL tokens differently.
        self.tokenizer.add_special_tokens({
            "additional_special_tokens":[Enums.QUESTION_SPECIAL_TOKEN, Enums.ANSWER_SPECIAL_TOKEN, Enums.QUESTION_TYPE_SPECIAL_TOKEN]
        })

        question_types = [qt.replace(" ","_") for qt in Enums.QUESTION_TYPES.values()]

        self.tokenizer.add_special_tokens({
            "additional_special_tokens":question_types
        })

        self.resizing_width, self.resizing_height = resizing_dimensions

        # self.image_transforms = image_transforms
        self.image_transformations = transforms.Compose([
                                transforms.ToTensor(),  # Convert to tensor
                                # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Normalize if needed
                            ])
        # self.create_transforms(image_transforms)

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

            # image_tensor = self.image_transformations(image=image_arr)["image"]
            image_tensor = self.image_transformations(image_arr)
            
            image_tensors.append(image_tensor)

        image_tensors = torch.stack(image_tensors, dim=0)
                        
        annotations_ids = []  
        random_answer_choices = []

        raw_answer_texts = []

        for annotation in annotations:
            ''' 
            randomly select a single choice from list of answers
            '''
            answer = random.choice(annotation.answers)
            random_answer_choices.append(self.answer_spaces.index(answer))
            
            raw_answer_texts.append(answer)
        
        answer_tensors = self.tokenizer(raw_answer_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=Enums.MAX_LEN)
        
        annotations_ids = torch.tensor(random_answer_choices)
        question_texts = [f'{Enums.QUESTION_SPECIAL_TOKEN} {question.question_text}' for idx, question in enumerate(questions)]

        question_tensors = self.tokenizer(question_texts, return_tensors="pt", padding="max_length", max_length=16, truncation=True)         

        decoder_question_texts = [f'{Enums.QUESTION_SPECIAL_TOKEN} {question.question_text} {Enums.ANSWER_SPECIAL_TOKEN}' for idx, question in enumerate(questions)]
        decoder_question_tensors = self.tokenizer(decoder_question_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=Enums.MAX_LEN)         

        if self.eval_mode:
            answers = [annotation.answers for annotation in annotations]
            return {
                "question_input_ids":question_tensors["input_ids"],
                "decoder_question_input_ids":decoder_question_tensors["input_ids"],
                "question_attention_masks":question_tensors["attention_mask"],
                "decoder_question_attention_masks":decoder_question_tensors["attention_mask"],
                "annotation_ids":annotations_ids, #list(tensors)
                "pixel_values":None,
                "image_tensors":image_tensors,
                "question_type_ids":None,
                "answers":answers,
                "questions":questions,
                "image_fns":images_file_paths,
                "answer_input_ids":answer_tensors["input_ids"],
                "answer_attention_masks":answer_tensors["attention_mask"]
            }


        return {
            "question_input_ids":question_tensors["input_ids"],
            "decoder_question_input_ids":decoder_question_tensors["input_ids"],
            "question_attention_masks":question_tensors["attention_mask"],
            "decoder_question_attention_masks":decoder_question_tensors["attention_mask"],
            "annotation_ids":annotations_ids,
            "pixel_values":None,
            "image_tensors":image_tensors,
            "question_type_ids":None,
            "answer_input_ids":answer_tensors["input_ids"],
            "answer_attention_masks":answer_tensors["attention_mask"]            
        }

    def __call__(self, data_points):

        return self.collect_preprocessed_data(data_points)

