import os
import datasets
import pandas as pd

# 指定本地数据集目录
local_data_dir = "/home/featurize/work/qzy/LLaMA-Factory/evaluation/ques"

_CITATION = """\
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
"""

_DESCRIPTION = """\
Measuring Massive Multitask Language Understanding by Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021).
"""

_HOMEPAGE = "https://github.com/hendrycks/test"
_LICENSE = "MIT"
_URL = "ques.zip"

task_list = [
    "Senior_Engineer_Examination",
]

class MMLUConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

class MMLU(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MMLUConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # 直接使用本地路径，而不是下载
        data_dir = local_data_dir
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", "test", f"{task_name}_test.csv"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", "val", f"{task_name}_val.csv"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", "dev", f"{task_name}_dev.csv"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = ["question", "A", "B", "C", "D", "answer"]

        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance

