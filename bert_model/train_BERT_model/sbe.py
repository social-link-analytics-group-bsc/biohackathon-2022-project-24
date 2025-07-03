# Loading script for the Sex Bias Extraction dataset.
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """ """

_DESCRIPTION = """
               """

_HOMEPAGE = """"""

_URL = "../../data/training_data/"
_TRAINING_FILE = "train.conll"
_DEV_FILE = "dev.conll"
_TEST_FILE = "test.conll"


class SBEConfig(datasets.BuilderConfig):
    """ Builder config for the Sex Bias Extraction dataset """

    def __init__(self, **kwargs):
        """BuilderConfig for SBE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SBEConfig, self).__init__(**kwargs)


class SBE(datasets.GeneratorBasedBuilder):
    """ SBE dataset."""

    BUILDER_CONFIGS = [
        SBEConfig(
            name="SBE",
            version=datasets.Version("2.0.0"),
            description="Sex Bias Extraction dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "sbe_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "n_fem",
                                "n_male",
                                "perc_fem",
                                "perc_male",
                                "sample",
                                "O",
                            ]
                        )
                    ),
                    "id": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            sbe_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "sbe_tags": sbe_tags,
                        }
                        guid += 1
                        tokens = []
                        sbe_tags = []
                else:
                    # SBE tokens are space separated
                    splits = line.split('\t')
                    tokens.append(splits[0])
                    sbe_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "tokens": tokens,
                "sbe_tags": sbe_tags,
                "id": str(guid),
            }
