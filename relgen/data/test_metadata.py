import unittest
from relgen.data.metadata import Metadata


class TestMetadata(unittest.TestCase):
    def test_validate_correct_relationships(self):
        metadata = Metadata()
        metadata.load_from_dict({
            "tables": {
                "t1": {},
                "t2": {},
                "t3": {},
                "t4": {},
                "t5": {},
            },
            "relationships": [
                {
                    "parent_table_name": "t1",
                    "child_table_name": "t2",
                    "parent_foreign_key": "id1",
                    "child_primary_key": "id2"
                },
                {
                    "parent_table_name": "t1",
                    "child_table_name": "t3",
                    "parent_foreign_key": "id1",
                    "child_primary_key": "id3"
                },
                {
                    "parent_table_name": "t3",
                    "child_table_name": "t4",
                    "parent_foreign_key": "id3",
                    "child_primary_key": "id4"
                },
                {
                    "parent_table_name": "t3",
                    "child_table_name": "t5",
                    "parent_foreign_key": "id3",
                    "child_primary_key": "id5"
                },
            ]
        })
        result = metadata.sorted_relationships
        self.assertEqual(result, [
            {
                "child_table_name": "t1",
            },
            {
                "parent_table_name": "t1",
                "child_table_name": "t2",
                "parent_foreign_key": "id1",
                "child_primary_key": "id2"
            },
            {
                "parent_table_name": "t1",
                "child_table_name": "t3",
                "parent_foreign_key": "id1",
                "child_primary_key": "id3"
            },
            {
                "parent_table_name": "t3",
                "child_table_name": "t4",
                "parent_foreign_key": "id3",
                "child_primary_key": "id4"
            },
            {
                "parent_table_name": "t3",
                "child_table_name": "t5",
                "parent_foreign_key": "id3",
                "child_primary_key": "id5"
            },
        ])

    def test_validate_incorrect_relationships_v1(self):
        metadata = Metadata()
        with self.assertRaises(ValueError):
            metadata.load_from_dict({
                "tables": {},
                "relationships": [
                    {
                        "parent_table_name": "t1",
                        "child_table_name": "t2",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    }
                ]
            })

    def test_validate_incorrect_relationships_v2(self):
        metadata = Metadata()
        with self.assertRaises(ValueError):
            metadata.load_from_dict({
                "tables": {
                    "t1": {},
                    "t2": {},
                    "t3": {},
                },
                "relationships": [
                    {
                        "parent_table_name": "t1",
                        "child_table_name": "t3",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    },
                    {
                        "parent_table_name": "t2",
                        "child_table_name": "t3",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    }
                ]
            })

    def test_validate_incorrect_relationships_v3(self):
        metadata = Metadata()
        with self.assertRaises(ValueError):
            metadata.load_from_dict({
                "tables": {
                    "t1": {},
                    "t2": {},
                    "t3": {},
                    "t4": {},
                },
                "relationships": [
                    {
                        "parent_table_name": "t1",
                        "child_table_name": "t2",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    },
                    {
                        "parent_table_name": "t1",
                        "child_table_name": "t3",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    },
                    {
                        "parent_table_name": "t2",
                        "child_table_name": "t4",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    },
                    {
                        "parent_table_name": "t3",
                        "child_table_name": "t4",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    }
                ]
            })

    def test_validate_incorrect_relationships_v4(self):
        metadata = Metadata()
        with self.assertRaises(ValueError):
            metadata.load_from_dict({
                "tables": {
                    "t1": {},
                    "t2": {},
                    "t3": {},
                },
                "relationships": [
                    {
                        "parent_table_name": "t1",
                        "child_table_name": "t2",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    },
                    {
                        "parent_table_name": "t2",
                        "child_table_name": "t3",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    },
                    {
                        "parent_table_name": "t3",
                        "child_table_name": "t1",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    }
                ]
            })

    def test_validate_incorrect_relationships_v5(self):
        metadata = Metadata()
        with self.assertRaises(ValueError):
            metadata.load_from_dict({
                "tables": {
                    "t1": {},
                    "t2": {},
                    "t3": {},
                    "t4": {},
                },
                "relationships": [
                    {
                        "parent_table_name": "t1",
                        "child_table_name": "t2",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    },
                    {
                        "parent_table_name": "t2",
                        "child_table_name": "t3",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    },
                    {
                        "parent_table_name": "t3",
                        "child_table_name": "t4",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    },
                    {
                        "parent_table_name": "t4",
                        "child_table_name": "t2",
                        "parent_foreign_key": "",
                        "child_primary_key": ""
                    }
                ]
            })


if __name__ == '__main__':
    unittest.main()
