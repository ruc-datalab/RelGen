import unittest
import pandas as pd

from relgen.utils.enum_type import SynthesisMethod
from relgen.data.metadata import Metadata
from relgen.data.dataset import Dataset
from relgen.synthesizer.diffusionsynthesizer import RelDDPMSynthesizer


class TestRelDDPMSynthesizer(unittest.TestCase):
    def test_relddpm_synthesizer_multi_model(self):
        metadata = Metadata()
        metadata.load_from_dict({
            "tables": {
                "t1": {
                    "primary_key": "id"
                },
                "t2": {
                    "primary_key": "id21"
                },
                "t3": {
                    "primary_key": "id31"
                },
            },
            "relationships": [
                {
                    "parent_table_name": "t1",
                    "child_table_name": "t2",
                    "parent_foreign_key": "id12",
                    "child_primary_key": "id21"
                },
                {
                    "parent_table_name": "t1",
                    "child_table_name": "t3",
                    "parent_foreign_key": "id13",
                    "child_primary_key": "id31"
                }
            ]
        })

        data = {
            "t1": pd.DataFrame({
                "id": [1, 2, 3, 4, 5, 6],
                "t1": ["a", "b", "c", "d", "e", "f"],
                "id12": [1, 2, 2, 3, 3, 3],
                "id13": [1, 1, 2, 2, 3, 3]
            }),
            "t2": pd.DataFrame({
                "t2": ["a", "b", "c"],
                "id21": [1, 2, 3]
            }),
            "t3": pd.DataFrame({
                "t3": ["a", "b", "c"],
                "id31": [1, 2, 3]
            }),
        }

        dataset = Dataset(metadata)
        dataset.fit(data)
        synthesizer = RelDDPMSynthesizer(dataset)
        synthesizer.fit(data, epochs=50, controller_steps=50)
        sampled_data = synthesizer.sample()

        self.assertEqual(list(sampled_data.keys()), ["t1", "t2", "t3"])
        print(sampled_data["t1"])
        print(sampled_data["t2"])
        print(sampled_data["t3"])

    def test_relddpm_synthesizer_single_model(self):
        metadata = Metadata()
        metadata.load_from_dict({
            "tables": {
                "t1": {
                    "primary_key": "id"
                },
                "t2": {
                    "primary_key": "id21"
                },
                "t3": {
                    "primary_key": "id31"
                },
            },
            "relationships": [
                {
                    "parent_table_name": "t1",
                    "child_table_name": "t2",
                    "parent_foreign_key": "id12",
                    "child_primary_key": "id21"
                },
                {
                    "parent_table_name": "t1",
                    "child_table_name": "t3",
                    "parent_foreign_key": "id13",
                    "child_primary_key": "id31"
                }
            ]
        })

        data = {
            "t1": pd.DataFrame({
                "id": [1, 2, 3, 4, 5, 6],
                "t1": ["a", "b", "c", "d", "e", "f"],
                "id12": [1, 2, 2, 3, 3, 3],
                "id13": [1, 1, 2, 2, 3, 3]
            }),
            "t2": pd.DataFrame({
                "t2": ["a", "b", "c"],
                "id21": [1, 2, 3]
            }),
            "t3": pd.DataFrame({
                "t3": ["a", "b", "c"],
                "id31": [1, 2, 3]
            }),
        }

        dataset = Dataset(metadata)
        dataset.fit(data)
        synthesizer = RelDDPMSynthesizer(dataset, method=SynthesisMethod.SINGLE_MODEL)
        synthesizer.fit(data, epochs=50, controller_steps=50)
        sampled_data = synthesizer.sample()

        self.assertEqual(list(sampled_data.keys()), ["t1", "t2", "t3"])
        print(sampled_data["t1"])
        print(sampled_data["t2"])
        print(sampled_data["t3"])


if __name__ == '__main__':
    unittest.main()
