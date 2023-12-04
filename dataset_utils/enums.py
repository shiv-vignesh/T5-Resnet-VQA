import albumentations as A 
from albumentations.pytorch import ToTensorV2

class Enums: 

    QUESTION_TYPES = {
        "eight": "Plants and Animals",
        "nine": "Science and Technology",
        "four": "Sports and Recreation",
        "six": "Geography, History, Language and Culture",
        "two": "Brands, Companies and Products",
        "other": "Other",
        "one": "Vehicles and Transportation",
        "five": "Cooking and Food",
        "ten": "Weather and Climate",
        "seven": "People and Everyday life",
        "three": "Objects, Material and Clothing"
    }

    TOTAL_QUESTION_TYPES = len(list(QUESTION_TYPES.keys()))

    QUESTION_IDS_TO_TYPE = {}
    QUESTION_TYPE_TO_IDS = {}

    for idx, q_type in enumerate(QUESTION_TYPES):
        QUESTION_IDS_TO_TYPE[idx] = q_type
        QUESTION_TYPE_TO_IDS[q_type] = idx

    IMAGE_ID_IMAGE_FN = {}
    IMAGES_DIR = ""

    TRANSFORM_STRATEGIES = {
        "smallestMaxSize" : A.SmallestMaxSize(max_size=256),
        "ShiftScaleRotate" : A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        "RandomCrop" : A.RandomCrop(height=224, width=224),
        "RGBShift" : A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        "RandomBrightnessContrast" : A.RandomBrightnessContrast(p=0.5),
        "MultiplicativeNoise" : A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        "Normalize" : A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        "HueSaturationValue" : A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        "RandomBrightnessContrast" : A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        "ToTensorV2" : ToTensorV2(),
    }

    QUESTION_SPECIAL_TOKEN = "[Question]"
    CONTEXT_SPECIAL_TOKEN = "[CONTEXT]"
    QUESTION_TYPE_SPECIAL_TOKEN = "[QUESTION_TYPE]"
    ANSWER_SPECIAL_TOKEN = "[Answer]"

    MAX_LEN = 30 # estimated through EDA.  Longest Annotation in Training is 26; Longest Annotation in Validation is 23
    ANSWERS_PER_QUESTION = 10

    NUM_BEAMS = 3
    PAD_TOKEN_ID = 0
    EOS_TOKEN_ID = 1