from datasets import load_dataset

from tasks.math.math_handler import MathTaskHandler


class AMC23TaskHandler(MathTaskHandler):
    def __init__(self):
        self.dataset = "AI-MO/aimo-validation-amc"

    @staticmethod
    def get_question_key():
        return "problem"

    def load_and_filter_dataset(
        self, start, end, split="train", source=None, filter_difficulty=False, args=None
    ):
        dataset = load_dataset(self.dataset)
        train_data = dataset[split].to_pandas()
        filtered_data = train_data[train_data["url"].str.contains("2023", na=False)]
        return filtered_data.iloc[start:end] if end > 0 else filtered_data.iloc[start:]
