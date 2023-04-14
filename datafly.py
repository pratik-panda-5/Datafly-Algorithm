import csv
import time
from io import StringIO
from queue import Queue

class Node:

    def __init__(self, data):
        self.data = data
        self.parent = None
        self.children = dict()

    def add_child(self, child):
        child.parent = self
        self.children[child.data] = child

class Tree:

    def __init__(self, root: Node):
        self.root = root
    def bfs_search(self, data, depth=None):
        visited, queue = set(), Queue()
        queue.put((self.root, 0))

        while not queue.empty():
            node, level = queue.get()
            if depth is not None and level > depth:
                break
            if depth is None:
                if node.data == data:
                    return node
            else:
                if level == depth and node.data == data:
                    return node
            for child in node.children.values():
                if child in visited:
                    continue
                queue.put((child, level + 1))

            visited.add(node)

        return None

    def _bfs_insert(self, child: Node, parent: Node) -> bool:
        node = self.bfs_search(parent.data)
        if node is not None:
            node.add_child(child)
            return True
        else:
            return False

    def insert(self, child: Node, parent: Node) -> bool:
        return self._bfs_insert(child, parent)

    def parent(self, data):
        node = self.bfs_search(data)

        if node is not None:
            return node.parent
        else:
            return None

class CsvDGH():

    def __init__(self, dgh_path):

        self.hierarchies = dict()
        self.gen_levels = dict()
        with open(dgh_path, 'r') as file:
            for line in file:
                csv_reader = csv.reader(StringIO(line))
                values = next(csv_reader)
                
                # If it doesn't exist a hierarchy with this root, add one:
                if values[-1] not in self.hierarchies:
                    self.hierarchies[values[-1]] = Tree(Node(values[-1]))
                    # Add the number of generalization levels:
                    self.gen_levels[values[-1]] = len(values) - 1
                # Populate hierarchy with the other values:
                self._insert_hierarchy(values[:-1], self.hierarchies[values[-1]])

    def generalize(self, value, gen_level=None):
        # Search across all hierarchies (slow if there are a lot of hierarchies):
        for hierarchy in self.hierarchies:

            # Try to find the node:
            if gen_level is None:
                node = self.hierarchies[hierarchy].bfs_search(value)
            else:
                node = self.hierarchies[hierarchy].bfs_search(
                    value,
                    self.gen_levels[hierarchy] - gen_level)     # Depth.

            if node is None:
                continue
            elif node.parent is None:
                # The value is a hierarchy root:
                return None
            else:
                return node.parent.data

        # The value is not found:
        raise KeyError(value)    
        
    @staticmethod
    def _insert_hierarchy(values, tree):
        current_node = tree.root

        for i, value in enumerate(reversed(values)):

            if value in current_node.children:
                current_node = current_node.children[value]
                continue
            else:
                # Insert the hierarchy from this node:
                for v in list(reversed(values))[i:]:
                    current_node.add_child(Node(v))
                    current_node = current_node.children[v]
                return True

        return False


class _Table:

    def __init__(self, pt_path: str, dgh_paths: dict):
        self.table = None
        self.attrs = dict()
        self._init_table(pt_path)
        self.dghs = dict()
        for attr in dgh_paths:
            self.add_dgh(dgh_paths[attr], attr)

    def __del__(self):
        self.table.close()

    def anonymize(self, qi_names: list, k: int, output_path: str, v=True):
        
        output = open(output_path, 'w')

        # Start reading the table file from the top:
        self.table.seek(0)
        # Dictionary whose keys are sequences of values for the Quasi Identifiers and whose values
        # are couples (n, s) where n is the number of occurrences of a sequence and s is a set
        # containing the indices of the rows in the original table file with those QI values:
        QI_freq = dict()
        # Dictionary whose keys are the indices in the QI attr names list, and whose values are
        # sets containing the corresponding domain elements:
        domains = dict()
        for i,atrribute in enumerate(qi_names):
            domains[i] = set()

        # Dictionary whose keys are the indices in the QI attr names list, and whose values are
        # the current levels of generalization, from 0 (not generalized):
        gen_levels = dict()
        for i, attr in enumerate(qi_names):
            gen_levels[i] = 0

        for i, row in enumerate(self.table):

            QI_seq = self._get_values(row, qi_names, i)

            # Skip if this row must be ignored:
            if QI_seq is None:
                continue
            else:
                QI_seq = tuple(QI_seq)

            if QI_seq in QI_freq:
                occurrences = QI_freq[QI_seq][0] + 1
                rows_set = QI_freq[QI_seq][1].union([i])
                QI_freq[QI_seq] = (occurrences, rows_set)
            else:
                # Initialize number of occurrences and set of row indices:
                QI_freq[QI_seq] = (1, set())
                QI_freq[QI_seq][1].add(i)

                # Update domain set for each attr in this sequence:
                for j, value in enumerate(QI_seq):
                    domains[j].add(value)

        while True:

            # Number of tuples which are not k-anonymous.
            count = 0

            for QI_seq in QI_freq:

                # Check number of occurrences of this sequence:
                if QI_freq[QI_seq][0] < k:
                    # Update the number of tuples which are not k-anonymous:
                    count += QI_freq[QI_seq][0]
            print(f"\n{count} rows are not k-anonymous.")

            # Limit the number of tuples to suppress to k:
            if count > k:

                # Get the attr whose domain has the max cardinality:
                max_cardinality, max_attr_idx = 0, None
                for attr_idx in domains:
                    if len(domains[attr_idx]) > max_cardinality:
                        max_cardinality = len(domains[attr_idx])
                        max_attr_idx = attr_idx

                # Index of the attr to generalize:
                attr_idx = max_attr_idx
                print(f"Attribute with most distinct values: {qi_names[attr_idx]}")
                # self._log("[LOG] Current attr with most distinct values is '%s'." %
                        #   qi_names[attr_idx], endl=True, enabled=v)

                # Generalize each value for that attr and update the attr set in the
                # domains dictionary:
                domains[attr_idx] = set()
                # Look up table for the generalized values, to avoid searching in hierarchies:
                generalizations = dict()

                # Note: using the list of keys since the dictionary is changed in size at runtime
                # and it can't be used an iterator:
                for j, QI_seq in enumerate(list(QI_freq)):
                    # print(f"Generalizing {qi_names[attr_idx]} for iteration {j}")
                    # self._log("[LOG] Generalizing attr '%s' for sequence %d... cock" %
                    #           (qi_names[attr_idx], j), endl=False, enabled=v)
                    # Get the generalized value:
                    if QI_seq[attr_idx] in generalizations:
                        # Find directly the generalized value in the look up table:
                        generalized_value = generalizations[attr_idx]
                    else:
                        # Get the corresponding generalized value from the attr DGH:
                        generalized_value = self.dghs[qi_names[attr_idx]].generalize(QI_seq[attr_idx], gen_levels[attr_idx])

                        if generalized_value is None:
                            # Skip if it's a hierarchy root:
                            continue

                        # Add to the look up table:
                        generalizations[attr_idx] = generalized_value

                    # Tuple with generalized value:
                    new_QI_seq = list(QI_seq)
                    new_QI_seq[attr_idx] = generalized_value
                    new_QI_seq = tuple(new_QI_seq)

                    # Check if there is already a tuple like this one:
                    if new_QI_seq in QI_freq:
                        # Update the already existing one:
                        # Update the number of occurrences:
                        occurrences = QI_freq[new_QI_seq][0] + QI_freq[QI_seq][0]
                        # Unite the row indices sets:
                        rows_set = QI_freq[new_QI_seq][1].union(QI_freq[QI_seq][1])
                        QI_freq[new_QI_seq] = (occurrences, rows_set)
                        # Remove the old sequence:
                        QI_freq.pop(QI_seq)
                    else:
                        # Add new tuple and remove the old one:
                        QI_freq[new_QI_seq] = QI_freq.pop(QI_seq)

                    # Update domain set with this attr value:
                    domains[attr_idx].add(QI_seq[attr_idx])

                print(f"Generalized {qi_names[attr_idx]}")
                # Update current level of generalization:
                gen_levels[attr_idx] += 1

            else:
                # Drop tuples which occur less than k times:
                for QI_seq, data in QI_freq.copy().items():
                    if data[0] < k:
                        QI_freq.pop(QI_seq)
                print(f"Suppressed {count} tuples.")

                # Start to read the table file from the start:
                self.table.seek(0)
                for i, row in enumerate(self.table):
                    table_row = self._get_values(row, list(self.attrs), i)

                    # Skip this row if it must be ignored:
                    if table_row is None:
                        continue

                    # Find sequence corresponding to this row index:
                    for QI_seq in QI_freq:
                        if i in QI_freq[QI_seq][1]:
                            line = self.set_values(table_row, QI_seq, qi_names)
                            print(line[:-2], file=output)
                            break
                break

        output.close()
        print("Final generalization levels:")
        for i in range(0, len(qi_names)):
            print("{:<10} : {:<5}".format(qi_names[i], gen_levels[i]))

    def _init_table(self, pt_path: str):
        self.table = open(pt_path, 'r')

    def _get_values(self, row: str, attrs: list, row_index=None):
        # Ignore empty lines:
        if row.strip() == '':
            return None

class CsvTable(_Table):

    def __init__(self, pt_path: str, dgh_paths: dict):

        super().__init__(pt_path, dgh_paths)

    def __del__(self):

        super().__del__()

    def anonymize(self, qi_names, k, output_path, v=False):

        super().anonymize(qi_names, k, output_path, v)

    def _init_table(self, pt_path):

        super()._init_table(pt_path)

        csv_reader = csv.reader(StringIO(next(self.table)))

        # Initialize the dictionary of table attrs:
        for i, attr in enumerate(next(csv_reader)):
            self.attrs[attr] = i

    def _get_values(self, row: str, attrs: list, row_index=None):

        super()._get_values(row, attrs, row_index)

        # Ignore the first line (which contains the attr names):
        if row_index is not None and row_index == 0:
            return None

        # Try to parse the row:
        csv_reader = csv.reader(StringIO(row))
        parsed_row = next(csv_reader)

        values = list()
        for attr in attrs:
            if attr in self.attrs:
                values.append(parsed_row[self.attrs[attr]])
            else:
                raise KeyError(attr)

        return values

    def set_values(self, row: list, values, attrs: list):
        for i, attr in enumerate(attrs):
            row[self.attrs[attr]] = values[i]

        values = StringIO()
        csv_writer = csv.writer(values)
        csv_writer.writerow(row)

        return values.getvalue()

    def add_dgh(self, dgh_path, attr):
        self.dghs[attr] = CsvDGH(dgh_path)


# Set Parameters
private_table = "example/db.csv"
quasi_identifier = ["Race","BirthDate","Gender","ZIP"]
domain_gen_hierarchies = ["example/race_generalization.csv","example/birth_generalization.csv","example/gender_generalization.csv","example/zip_generalization.csv"]
k = int(input())
output = "example/db_"+str(k)+"_anon.csv"

dgh_paths = dict()
for i, qi_name in enumerate(quasi_identifier):
    dgh_paths[qi_name] = domain_gen_hierarchies[i]
table = CsvTable(private_table, dgh_paths)
start = time.time()
table.anonymize(quasi_identifier, k, output, v=True)
end = (time.time()) - start
print(f"Time taken: {end:.3f} seconds")
