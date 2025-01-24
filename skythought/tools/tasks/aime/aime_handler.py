from datasets import load_dataset

from tasks.math.math_handler import MathTaskHandler
from util.model_utils import MODEL_TO_NAME


class AIMETaskHandler(MathTaskHandler):
    def __init__(self):
        self.dataset = "AI-MO/aimo-validation-aime"

    def generate_prompt(self, prompt, model):
        if MODEL_TO_NAME[model] == "Sky-T1-32B-Preview":
            return prompt + "\nReturn your final response within \\boxed{{}}"
        else:
            return "Return your final response within \\boxed{{}}. " + prompt

    @staticmethod
    def get_question_key():
        return "problem"

    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem["problem"], model)
            conversations.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ]
            )
        return conversations

    def load_and_filter_dataset(
        self, start, end, split="train", source=None, filter_difficulty=False, args=None
    ):
        dataset = load_dataset(self.dataset)
        train_data = dataset[split].to_pandas()
        filtered_data = train_data[train_data["url"].str.contains("2024", na=False)]
        return filtered_data.iloc[start:end] if end > 0 else filtered_data.iloc[start:]
