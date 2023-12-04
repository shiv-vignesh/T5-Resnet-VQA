import torch
import json, os, random
import cv2

from PIL import Image

from torch.utils.data import Dataset 
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor

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
        answers = [answer.replace(" ","_") for answer in answers]
        self.answers = answers
        self.answer_type = answer_type

    def __str__(self) -> str:
        return f"Question-Id: {self.question_id}, Length of Answers: {len(self.answers)}, Question-Type: {self.question_type}"

class OKVQADataset(Dataset):

    def __init__(
                self,
                dataset_json_fp:str,
                images_dir:str,
                type:str 
                 ):
    
        self.images_dir = images_dir
        self.type = type
        self.image_ids_to_fn = {}

        self.dataset_json = json.load(open(dataset_json_fp))

        self.images_fns = os.listdir(images_dir)

        for image_fn in self.images_fns:
            if self.type == "train":
                image_id = image_fn.split('COCO_train2014_')[1].lstrip('0').split('.')[0]

            elif self.type == "val":
                image_id = image_fn.split('COCO_val2014_')[1].lstrip('0').split('.')[0]

            self.image_ids_to_fn[int(image_id)] = image_fn
        
        
    def __getitem__(self, idx):
        
        data_points = list(self.dataset_json[idx].values())[0]

        question = Question(
            data_points["question_text"],
            data_points["question_id"],
            data_points["image_id"],
            
        )

        annotation = Annotation(
            data_points["question_id"],data_points["image_id"], data_points["question_type"],
            data_points["unique_answers"], data_points["answer_type"]
        )

        image_id = question.image_id
        image_fn = self.image_ids_to_fn[image_id]

        return {
            "question": question,
            "annotation":annotation,
            "image_path":f'{self.images_dir}/{image_fn}'
        }

    def __len__(self):
        return len(self.dataset_json)  

class VitT5CollateFn(object):
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
        question_types = []
        question_type_ids = []
        random_answer_choices = []

        for annotation in annotations:
            ''' 
            randomly select a single choice from list of answers
            '''
            
            answer = random.choice(annotation.answers)
            random_answer_choices.append(self.answer_spaces.index(answer))

            question_type = annotation.question_type
            question_type_id = Enums.QUESTION_TYPE_TO_IDS[question_type]
            question_type_ids.append(question_type_id)

            question_type = Enums.QUESTION_TYPES[question_type]
            question_type = question_type.replace(" ","_")
            question_types.append(question_type)

        # annotations_ids = torch.stack(random_answer_choices, dim=0)

        annotations_ids = torch.tensor(random_answer_choices)
        question_texts = [f'{Enums.QUESTION_SPECIAL_TOKEN} {question.question_text} {Enums.QUESTION_TYPE_SPECIAL_TOKEN} {question_types[idx]}' for idx, question in enumerate(questions)]

        question_tensors = self.tokenizer(question_texts, return_tensors="pt", padding="longest")         

        decoder_question_texts = [f'{Enums.QUESTION_SPECIAL_TOKEN} {question.question_text} {Enums.QUESTION_TYPE_SPECIAL_TOKEN} {question_types[idx]} {Enums.ANSWER_SPECIAL_TOKEN}' for idx, question in enumerate(questions)]
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
                "question_type_ids":question_type_ids, #list(int)
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
            "question_type_ids":question_type_ids
        }

    def __call__(self, data_points):

        return self.collect_preprocessed_data(data_points)

