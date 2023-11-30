import torch
import torch.nn as nn 
import torch.nn.functional as F

from torchvision.models import resnet50, resnet34, resnet18

from transformers import T5ForQuestionAnswering, T5Config
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria

from dataset_utils.enums import Enums
from dataset_utils.utils import prepare_prediction_input_ids

class VisionModel(nn.Module):

    def __init__(self, model_name:str, fine_tune_vision:bool=True, device="cpu"):
        super().__init__()

        self.device = device
        self.fine_tune = fine_tune_vision

        if model_name=="resnet50":
            self.model = resnet50(pretrained=True) 
            self.linear = nn.Sequential(
                nn.Linear(2048, 768),
                nn.ReLU(),
                nn.Linear(768, 768)                 
            )

        if model_name == "resnet34":
            self.model = resnet34(pretrained=True)
            self.linear = nn.Sequential(
                nn.Linear(512, 768),
                nn.ReLU(),
                nn.Linear(768, 768)                 
            )

        if model_name == "resnet18":
            self.model = resnet18(pretrained=True)
            self.linear = nn.Sequential(
                nn.Linear(512, 768),
                nn.ReLU(),
                nn.Linear(768, 768)                 
            )

    def forward(self, image_tensor:torch.tensor, reduce_dim:bool=True):
        image_features = image_tensor
        for layer_name, resnet_block in self.model._modules.items():
            if layer_name not in ["avgpool","fc"]:
                image_features = resnet_block(image_features.float()) 

        if reduce_dim:
            bs, hidden_dim, _, _ = image_features.size() # size [bs, 2048, 7, 7] > resnet50; #size [bs, 512, 14] > resnet34 & resnet18
            image_features = image_features.view(bs, hidden_dim, -1) #size [bs, 2048, 14] > resnet50; size [bs, 512, 14] > resnet34 & resnet18
            image_features = image_features.permute(0, 2, 1) #size [bs, 14, 2048] > resnet50; size [bs, 14, 512] > resnet34 & resnet18
            image_features = self.linear(image_features) 

        return image_features
    
class LanguageModel(nn.Module):

    def __init__(self, model_name:str, fine_tune_lm_encoder:bool=True, fine_tune_lm_decoder:bool=True, device="cpu"):
        super().__init__()

        self.device = device
        self.fine_tune_encoder = fine_tune_lm_encoder
        self.fine_tune_decoder = fine_tune_lm_decoder

        if model_name == "t5-base":
            model = T5ForQuestionAnswering.from_pretrained(model_name)
            self.encoder = model.encoder
            self.decoder = model.decoder

    def forward_encoder(self, question_input_ids:torch.tensor, question_attention_masks):

        question_embeddings = self.encoder(
            input_ids = question_input_ids,
            attention_mask=question_attention_masks
        )
        
        return question_embeddings
    
    def forward_decoder(self, encoder_hidden_states:torch.tensor, 
                        decoder_input_ids:torch.tensor=None,
                        decoder_input_embeddings:torch.tensor=None):

        decoder_embeddings = self.decoder(
            encoder_hidden_states=encoder_hidden_states, 
            input_ids = decoder_input_ids,
            inputs_embeds = decoder_input_embeddings
        )

        return decoder_embeddings

'''
Remove Question Padding before concatenating the image embedding, 
after concatenation re-pad it to max_len before passing to decoder. 
'''

class VQAModel(nn.Module):

    def __init__(self, 
                vision_model_name:str,
                language_model_name:str,
                fine_tune_lm_encoder:bool=True, 
                fine_tune_lm_decoder:bool=True,
                fine_tune_vision:bool=True,
                device="cpu"
                ):
        super().__init__()

        self.vision_model = VisionModel(
            model_name=vision_model_name, 
            fine_tune_vision=fine_tune_vision, 
            device=device
            )
        self.language_model = LanguageModel(
            model_name=language_model_name, 
            fine_tune_lm_encoder=fine_tune_lm_encoder, 
            fine_tune_lm_decoder=fine_tune_lm_decoder, 
            device=device
            )

        self.question_type_classifier = nn.Sequential(
            nn.Linear(
            768, Enums.TOTAL_QUESTION_TYPES,
            nn.Softmax(Enums.TOTAL_QUESTION_TYPES)
        )
        ) #fill this up. (768, len(question_types))

        self.answer_classifier = nn.Sequential(
            nn.Linear(
                768, T5Config().vocab_size
            ) #fill this up. (768, vocab size)
        )

        self.qa_outputs = nn.Linear(
            768, T5Config().num_labels
        )
        self.device = device

    def forward(self, 
                question_input_ids:torch.tensor, 
                question_attention_masks:torch.tensor, 
                annotation_ids:torch.tensor,
                image_tensors:torch.tensor,
                question_type_ids:torch.tensor
                ):
        
        if not self.vision_model.fine_tune:
            with torch.no_grad():
                image_emebedings = self.vision_model(image_tensors) # [bs, 49, 768]
        
        else:
            image_emebedings = self.vision_model(image_tensors) # [bs, 49, 768]

        if not self.language_model.fine_tune_encoder:
            with torch.no_grad():
                text_emebedings = self.language_model.forward_encoder(
                    question_input_ids = question_input_ids,
                    question_attention_masks = question_attention_masks
                ).last_hidden_state # [bs, longest_ques_len_in_batch, 768]

        else:
            text_emebedings = self.language_model.forward_encoder(
                question_input_ids = question_input_ids,
                question_attention_masks = question_attention_masks
            ).last_hidden_state # [bs, longest_ques_len_in_batch, 768]            

        combined_embeddings = torch.cat(
            (text_emebedings, image_emebedings), dim=1
        ) # [bs, longest_ques_len_in_batch + 49, 768]

        if question_input_ids.shape[1] < Enums.MAX_LEN:
            zero_padding = torch.zeros(
                [question_input_ids.shape[0], Enums.MAX_LEN - question_input_ids.shape[1]]
            ).to(self.device)        

            zero_padding = zero_padding.to(question_input_ids.dtype)

            reshaped_question_input_ids = torch.cat(
                (question_input_ids, zero_padding), dim=1
            ) # [bs, Enums.Max_len, 768]

        else:
            reshaped_question_input_ids = question_input_ids[:,:Enums.MAX_LEN]

        decoder_input_ids = self.language_model.encoder._shift_right(reshaped_question_input_ids) # [bs, Enums.Max_len, 768]

        if not self.language_model.fine_tune_decoder:
            with torch.no_grad():
                decoder_embeddings = self.language_model.forward_decoder(
                    combined_embeddings,
                    decoder_input_ids=decoder_input_ids
                ).last_hidden_state # [bs, Enums.Max_len, 768]

        else:
            decoder_embeddings = self.language_model.forward_decoder(
                combined_embeddings,
                decoder_input_ids=decoder_input_ids
            ).last_hidden_state # [bs, Enums.Max_len, 768]            

        # batch_size, longest_len, hidden_dim = decoder_embeddings.size()

        reshaped_decoder_embedding = decoder_embeddings.unsqueeze(1).expand(
            -1, Enums.ANSWERS_PER_QUESTION, -1, -1
        ) #converting # [bs, Enums.Max_len, 768] to [bs, 10, Enums.Max_len, 768]. 10 is the number of answers per question. CrossEntropy Loss can be computed smoothly when predicting 10 
        # different answers, one-to-one loss computation.

        answer_logits = self.answer_classifier(
            reshaped_decoder_embedding
        ) #[bs, 10, Enums.Max_len, 32128], last dim is the vocab size of T5

        answer_logits = torch.softmax(answer_logits, dim=2)

        question_type_embedding = torch.mean(
            decoder_embeddings, dim=1
        )

        question_type_scores = self.question_type_classifier(
            question_type_embedding
        )

        answer_loss, question_type_loss = None, None 


        if self.training or question_type_ids!=None:
            question_type_loss = F.cross_entropy(
                question_type_scores.view(-1, question_type_scores.size(-1)), question_type_ids.view(-1)
            )

        if self.training or annotation_ids!=None:
            answer_loss = F.cross_entropy(
                answer_logits.view(-1, answer_logits.size(-1)), annotation_ids.view(-1), ignore_index=-100
            )
        
        return answer_logits, question_type_scores, answer_loss, question_type_loss                

    def beam_search(self, prediction:torch.tensor, pad_token_id:int, eos_token_id:int, top_k=3, min_prob_threshold=1e-5):
        ''' 
        BeamSearchScore process() and finalize()
        '''
        batch_size, seq_length, vocab_size = prediction.shape
        log_prob, indices = prediction[:, 0, :].topk(top_k, sorted=True)
        # log_prob = nn.functional.log_softmax(log_prob, dim=-1)
        indices = indices.unsqueeze(-1)
        for n1 in range(1, seq_length):
            log_prob_temp = log_prob.unsqueeze(-1) + prediction[:, n1, :].unsqueeze(1).repeat(1, top_k, 1)
            log_prob, index_temp = log_prob_temp.view(batch_size, -1).topk(top_k, sorted=True)         
            
            mask = log_prob < torch.log(torch.tensor(min_prob_threshold))
            index_temp[mask] = pad_token_id

            idx_begin = index_temp // vocab_size  # retrieve index of start sequence
            idx_concat = index_temp % vocab_size  # retrieve index of new token
            new_indices = torch.zeros((batch_size, top_k, n1+1), dtype=torch.int64)
            for n2 in range(batch_size):
                new_indices[n2, :, :-1] = indices[n2][idx_begin[n2]]
                new_indices[n2, :, -1] = idx_concat[n2]
            indices = new_indices
        return indices, log_prob      


    def generate_answers(self,
                question_input_ids:torch.tensor, 
                question_attention_masks:torch.tensor, 
                image_tensors:torch.tensor,
                question_type_ids:torch.tensor,
                pad_token_id, eos_token_id,
                num_beams = 2,
                top_k=10
                ):
        

        image_emebedings = self.vision_model(image_tensors) # [bs, 49, 768]
        text_emebedings = self.language_model.forward_encoder(
                        question_input_ids = question_input_ids,
                        question_attention_masks = question_attention_masks
                    ).last_hidden_state # [bs, longest_ques_len_in_batch, 768]            

        combined_embeddings = torch.cat(
            (text_emebedings, image_emebedings), dim=1
        ) # [bs, longest_ques_len_in_batch + 49, 768]     

        if question_input_ids.shape[1] < Enums.MAX_LEN:
            zero_padding = torch.zeros(
                [question_input_ids.shape[0], Enums.MAX_LEN - question_input_ids.shape[1]]
            ).to(self.device)        

            zero_padding = zero_padding.to(question_input_ids.dtype)

            reshaped_question_input_ids = torch.cat(
                (question_input_ids, zero_padding), dim=1
            ) # [bs, Enums.Max_len, 768]

        else:
            reshaped_question_input_ids = question_input_ids[:,:Enums.MAX_LEN]

        reshaped_decoder_input_ids = self.language_model.encoder._shift_right(reshaped_question_input_ids) # [bs, Enums.Max_len]

        ''' 
        Potential buggy code. TODO, Check later
        '''        
        # [bs*num_beams, Enums.Max_len]
        # reshaped_decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.shape[-1]).repeat(num_beams, 1)
        # batch_size, length, hidden_dim = combined_embeddings.size()
        
        # reshaped_combined_embeddings = combined_embeddings.unsqueeze(1).repeat(
        #     1, num_beams, 1, 1
        # ).view(batch_size * num_beams, length, hidden_dim)

        decoder_embeddings = self.language_model.forward_decoder(
            combined_embeddings,
            decoder_input_ids=reshaped_decoder_input_ids
        ).last_hidden_state # [bs*num_beams, Enums.Max_len, 768]  

        answer_logits = self.answer_classifier(
            decoder_embeddings
        )

        return self.beam_search(answer_logits.cpu(), 
                                pad_token_id,
                                eos_token_id,
                                top_k=top_k)