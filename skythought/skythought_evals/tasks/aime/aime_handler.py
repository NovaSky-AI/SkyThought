from typing import Dict

from ..math.math_handler import MathTaskHandler


class AIMETaskHandler(MathTaskHandler):
    def generate_prompt(self, problem: Dict):
        return self.task_config.templating_parameters["template"].format(
            prompt=problem[self.question_key]
        )

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ):
        train_data = self.load_dataset(subset=subset, split=split).to_pandas()
        if self.task_config.preprocess_config:
            if "url" in self.task_config.preprocess_config:
                train_data = train_data[
                    train_data["url"].str.contains(
                        self.task_config.preprocess_config["url"], na=False
                    )
                ]
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]
