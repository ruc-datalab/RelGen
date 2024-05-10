import os
from typing import Dict, List, Any
from queue import Queue

from relgen.data.utils import read_json


class Metadata:
    def __init__(self):
        self.tables = {}
        self.relationships = []
        self.relationships_matrix = {}
        self.sorted_relationships = {}

    def __str__(self):
        newline = os.linesep
        return f"tables:{newline}{self.tables}{newline}relationships:{newline}{self.relationships}"

    def load_from_dict(self, metadata_dict: Dict[str, Any]):
        self.tables = metadata_dict["tables"]
        if "relationships" in metadata_dict:
            self.relationships = metadata_dict["relationships"]
        self._update_relationships_matrix()
        self._update_sorted_relationships()
        self._validate_relationships()

    def load_from_json(self, filepath):
        metadata = read_json(filepath)
        return self.load_from_dict(metadata)

    def _update_relationships_matrix(self):
        for table in self.tables.keys():
            self.relationships_matrix[table] = {
                "parent_tables": [],
                "child_tables": []
            }
        for relationship in self.relationships:
            parent_table_name = relationship["parent_table_name"]
            child_table_name = relationship["child_table_name"]
            if parent_table_name not in self.tables.keys():
                raise ValueError("Table in relationships must be in `tables`")
            if child_table_name not in self.tables.keys():
                raise ValueError("Table in relationships must be in `tables`")
            self.relationships_matrix[parent_table_name]["child_tables"].append(relationship)
            self.relationships_matrix[child_table_name]["parent_tables"].append(relationship)

    def _validate_relationships(self):
        root = None
        for table_name, relationship in self.relationships_matrix.items():
            if len(relationship["parent_tables"]) == 0:
                if root is None:
                    root = table_name
                else:
                    raise ValueError("Relationships can only have one root")
        if root is None:
            raise ValueError("Relationships must have one root")
        table_queue = Queue()
        table_queue.put(root)
        visited_tables = set()
        visited_tables.add(root)
        while not table_queue.empty():
            parent_table_name = table_queue.get()
            for relationship in self.relationships_matrix[parent_table_name]["child_tables"]:
                child_table_name = relationship["child_table_name"]
                if child_table_name in visited_tables:
                    raise ValueError("Relationships can not have cycle and a table can only have one parent table")
                table_queue.put(child_table_name)
                visited_tables.add(child_table_name)

    def _update_sorted_relationships(self):
        sort_results = []
        in_degree = {}
        table_queue = Queue()
        for table_name, relationship in self.relationships_matrix.items():
            in_degree[table_name] = len(relationship["parent_tables"])
            if in_degree[table_name] == 0:
                table_queue.put(table_name)
                sort_results.append({
                    "child_table_name": table_name
                })
        while not table_queue.empty():
            parent_table_name = table_queue.get()
            for relationship in self.relationships_matrix[parent_table_name]["child_tables"]:
                child_table_name = relationship["child_table_name"]
                in_degree[child_table_name] -= 1
                if in_degree[child_table_name] == 0:
                    table_queue.put(child_table_name)
                    sort_results.append({
                        "parent_table_name": parent_table_name,
                        "child_table_name": child_table_name,
                        "parent_foreign_key": relationship["parent_foreign_key"],
                        "child_primary_key": relationship["child_primary_key"]
                    })
        self.sorted_relationships = sort_results
