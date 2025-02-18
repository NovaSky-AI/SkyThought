from ..math.math_handler import MathTaskHandler


class AMC23TaskHandler(MathTaskHandler):
    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ):
        train_data = self.load_dataset(subset=subset, split=split).to_pandas()
        filtered_data = train_data[train_data["url"].str.contains("2023", na=False)]
        return filtered_data.iloc[start:end] if end > 0 else filtered_data.iloc[start:]
