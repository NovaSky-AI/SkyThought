from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from torch import nn
import numpy as np
from mat.envs.math.prompts import IN_CONTEXT_EXAMPLE

class MSProcessRM(nn.Module):

    def __init__(self, all_args):
        super().__init__()
        self.model_name_or_path = all_args.prm_model_name_or_path
        
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.pad_token_id = 0
        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1] # 12902
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map="auto",).eval()
        print(f"successfully loaded PRM.")
            

    @torch.no_grad()
    def get_reward(self, obs: list[np.ndarray[str]], actions: list[np.ndarray[str]]):
        inputs_for_prm = []
        for o, a in zip(obs.copy(), actions.copy()):
            o = o[0].replace(IN_CONTEXT_EXAMPLE, "")
            a = a[0].replace(self.step_tag, "").strip()
            inputs_for_prm.append(f"{o}{a} {self.step_tag}")
        input_ids = self.tokenizer(inputs_for_prm, return_tensors="pt", padding=True).to("cuda")
        logits = self.model(**input_ids).logits[:, :, self.candidate_tokens]
        score = logits.softmax(dim=-1)[:, :, 0]
        
        step_scores = []
        for i in range(np.shape(score)[0]):
            step_score = score[i][input_ids["input_ids"][i] == self.step_tag_id]
            last_step_score = step_score[-1]
            step_scores.append([last_step_score.item()])
        step_scores = np.array(step_scores)
        
        return step_scores