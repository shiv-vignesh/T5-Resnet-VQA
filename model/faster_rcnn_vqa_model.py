import torch
import torch.nn as nn 
import torch.nn.functional as F

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from transformers import T5ForConditionalGeneration, T5ForQuestionAnswering
from .multi_head_vision_text_attn import SGA
from .multi_head_vision_text_attn import ImageConfiguration
from .multi_head_vision_text_attn import TextConfiguration

from collections import defaultdict

class AttentionPooler(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooler, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)  # Apply softmax along the sequence dimension
        )

    def forward(self, x):
        att_weights = self.attention(x).transpose(1, 2)  # Transpose for weighted sum
        pooled_output = torch.bmm(att_weights, x).squeeze(1)  # Weighted sum
        return pooled_output

class FasterRcnnVQAModel(nn.Module):

    def __init__(
            self, 
            vision_model_name:str,
            language_model_name:str,
            answer_spaces:int,
            fine_tune_lm_encoder:bool=True, 
            fine_tune_lm_decoder:bool=True,
            fine_tune_vision:bool=True,
            num_attention_blocks=3,
            device="cpu"):
        
        super().__init__()

        ''' 
        Implement temperature scaling. 
        '''

        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name

        
        if self.vision_model_name == "faster-rcnn":
            self.vision_model = fasterrcnn_resnet50_fpn(pretrained=True)
            self.vision_model = self.vision_model.backbone

        if self.language_model_name == "t5-base":
            self.lang_model = T5ForQuestionAnswering.from_pretrained(self.language_model_name)
            self.lang_model = self.lang_model.encoder

        self.upscale_layer = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=768,
            kernel_size=3,  # Adjust kernel size as needed
            stride=1,       # Adjust stride as needed
            padding=1       # Adjust padding as needed
        )

        __IMAGE_C = ImageConfiguration()
        _TEXT_C = TextConfiguration()

        self.sga_modules = nn.ModuleList([SGA(__IMAGE_C, _TEXT_C) for _ in range(num_attention_blocks)])

        self.classification_layer = nn.Linear(
            768, answer_spaces
        )

        self.attention_pooler = AttentionPooler(768)

        self.fine_tune_lm_encoder = fine_tune_lm_encoder
        self.fine_tune_lm_decoder = fine_tune_lm_decoder
        self.fine_tune_vision = fine_tune_vision

        self.device = device
        self.num_beams = 2
        self.max_answer_length = 5

        self.temperature_scaler = 1.5

    def forward(self, 
                question_input_ids:torch.tensor, 
                decoder_question_input_ids:torch.tensor,
                question_attention_masks:torch.tensor, 
                decoder_question_attention_masks:torch.tensor,
                annotation_ids:torch.tensor,
                image_tensors:torch.tensor,
                answer_input_ids:torch.tensor=None,
                pixel_values:torch.tensor=None,                                
                answer_attention_masks:torch.tensor=None,                
                question_type_ids:torch.tensor=None,
                ):
        
                
        if self.vision_model_name == "faster-rcnn":
            self.vision_model.eval()
            with torch.no_grad():
                # vision_embeddings = self.vision_model.backbone(image_tensors)['pool']
                vision_embeddings = self.vision_model(image_tensors)['pool']
        
        vision_embeddings = self.upscale_layer(vision_embeddings)
 
        text_encoder_outputs = self.lang_model(
            input_ids = question_input_ids,
            attention_mask=question_attention_masks
        ).last_hidden_state

        flatted_vision_embeddings = vision_embeddings.view(vision_embeddings.shape[0], vision_embeddings.shape[1], -1)
        flatted_vision_embeddings = flatted_vision_embeddings.permute(0, 2, 1)

        fused_embeddings = None

        for sga in self.sga_modules:
            fused_embeddings = sga(text_encoder_outputs, flatted_vision_embeddings)
            flatted_vision_embeddings = fused_embeddings

        #[bs, 1, 768]
        fused_embeddings = self.attention_pooler(fused_embeddings)

        lm_logits = self.classification_layer(fused_embeddings)
        # lm_logits = lm_logits/self.temperature_scaler
        lm_logits = nn.functional.log_softmax(lm_logits, dim=-1)

        if annotation_ids != None:        
            loss = nn.NLLLoss()(
                lm_logits, annotation_ids)

            return lm_logits, loss 

        else:
            return lm_logits, None

    def generate_answers(self,question_input_ids:torch.tensor, 
                decoder_question_input_ids:torch.tensor,
                question_attention_masks:torch.tensor, 
                decoder_question_attention_masks:torch.tensor,
                image_tensors:torch.tensor,
                annotation_ids:torch.tensor=None,                
                answer_input_ids:torch.tensor=None,
                pixel_values:torch.tensor=None,                                
                answer_attention_masks:torch.tensor=None,                
                question_type_ids:torch.tensor=None):
    
        image_feature_maps_dict = defaultdict()

        if self.vision_model_name == "faster-rcnn":
            self.vision_model.eval()
            with torch.no_grad():
                vision_embeddings = self.vision_model(image_tensors)

                for k, tensors in vision_embeddings.items():
                    image_feature_maps_dict[k] = tensors

                vision_embeddings = vision_embeddings['pool']
                vision_embeddings = self.upscale_layer(vision_embeddings)

        # text_encoder_outputs = self.lang_model.encoder(
        #     input_ids = question_input_ids,
        #     attention_mask=question_attention_masks
        # ).last_hidden_state

        text_encoder_outputs = self.lang_model(
            input_ids = question_input_ids,
            attention_mask=question_attention_masks
        ).last_hidden_state

        flatted_vision_embeddings = vision_embeddings.view(vision_embeddings.shape[0], vision_embeddings.shape[1], -1)
        flatted_vision_embeddings = flatted_vision_embeddings.permute(0, 2, 1)

        fused_embeddings = None

        for sga in self.sga_modules:
            fused_embeddings = sga(text_encoder_outputs, flatted_vision_embeddings)
            flatted_vision_embeddings = fused_embeddings

        #[bs, 1, 768]

        fused_embeddings = self.attention_pooler(fused_embeddings)

        lm_logits = self.classification_layer(fused_embeddings)
        lm_logits = nn.functional.log_softmax(lm_logits, dim=-1)

        if annotation_ids != None:        
            loss = nn.NLLLoss()(
                lm_logits, annotation_ids)

            return lm_logits, loss, image_feature_maps_dict

        else:
            return lm_logits, None, image_feature_maps_dict
