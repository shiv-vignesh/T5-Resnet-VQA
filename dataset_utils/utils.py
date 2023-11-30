import torch

from .enums import Enums

def prepare_prediction_input_ids(answer_logits:torch.tensor, vocab_size:int):

    batch_size, seq_len, hidden_dim = answer_logits.shape

    input_ids = torch.randint(0, vocab_size, (batch_size * Enums.NUM_BEAMS, seq_len + 1))
    input_ids[:, :seq_len] = torch.argmax(answer_logits, dim=-1).view(-1, seq_len)

    next_scores = torch.rand(batch_size, 2 * Enums.NUM_BEAMS)
    next_tokens = torch.randint(0, vocab_size, (batch_size, 2 * Enums.NUM_BEAMS))
    next_indices = torch.randint(0, Enums.NUM_BEAMS, (batch_size, 2 * Enums.NUM_BEAMS))    

    return input_ids, next_scores, next_tokens, next_indices

def convert_time_to_readable_format(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    if not hour:
        if not minutes:
            time = f"{seconds} Seconds"
        else:
            time = f"{minutes} minute/s and {seconds} seconds"
    else:
        time = f"{hour} hour/s {minutes} minute/s {seconds} seconds"

    return time
