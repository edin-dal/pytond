import copy
from enum import Enum
from typing import List, Union

global_context = {'is_last_rule': False}

#### Enums ########################################
   
class BinOpType(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    AND = 5
    OR = 6
    LIKE = 7

    def __str__(self):
        match self.name:
            case "ADD":
                return "+"
            case "SUB":
                return "-"
            case "MUL":
                return "*"
            case "DIV":
                return "/"
            case "AND":
                return " AND "
            case "OR":
                return " OR "
            case "LIKE":
                return " LIKE "

######

class AggregateType(Enum):
    COUNT = 1
    SUM = 2
    MIN = 3
    MAX = 4
    AVG = 5
    COUNT_DISTINCT = 6

    def __str__(self):
        match self.name:
            case "COUNT":
                return "count"
            case "SUM":
                return "sum"
            case "MIN":
                return "min"
            case "MAX":
                return "max"
            case "AVG":
                return "avg"
            case 'COUNT_DISTINCT':
                return 'count_distinct'
            
    def to_sql(self):
        match self.name:
            case "COUNT":
                return "COUNT"
            case "SUM":
                return "SUM"
            case "MIN":
                return "MIN"
            case "MAX":
                return "MAX"
            case "AVG":
                return "AVG"
            case 'COUNT_DISTINCT':
                return 'COUNT DISTINCT'

######

class ThetaType(Enum):
    ASSIGN = 1
    EQ = 2
    NE = 3
    LT = 4
    LE = 5
    GT = 6
    GE = 7
    NOT = 8
    IN = 9

    def __str__(self):
        match self.name:
            case "ASSIGN":
                return "="
            case "EQ":
                return "="
            case "NE":
                return "!="
            case "LT":
                return "<"
            case "LE":
                return "<="
            case "GT":
                return ">"
            case "GE":
                return ">="
            case "NOT":
                return "NOT"
            case "IN":
                return " IN "

######

class OuterType(Enum):
    FULL = 1
    LEFT = 2
    RIGHT = 3

    def __str__(self):
        return self.name.lower()
    
    def to_sql(self):
        match self.name:
            case "FULL":
                return "FULL OUTER JOIN"
            case "LEFT":
                return "LEFT JOIN"
            case "RIGHT":
                return "RIGHT JOIN"

######

class ExternalType(Enum):
    YEAR = 1
    SLICE = 2
    UID = 3
    LOWER = 4
    UNIQUE = 5
    ISNULL = 6
    NOT = 7
    INT = 8

#### Constructs ###################################

class Base:
    def __init__(self):
        self.id = Base._id
        Base._id += 1
    _id = 1

    def __hash__(self) -> int:
        return str(self).__hash__()

###################################################

class Term(Base):
    def __init__(self):
        super().__init__()

###################################################

class Constant(Term):
    def __init__(self, value):
        super().__init__()
        self.value = value
    
###################################################
        
class Aggregate(Term):
    def __init__(self, type: AggregateType, term: 'Term'):
        super().__init__()
        self.type = type
        self.term = term
    def __str__(self):
        return f'{self.type}({self.term})'
    def to_sql(self):
        if self.type == AggregateType.COUNT_DISTINCT:
            return f'COUNT(DISTINCT({self.term.to_sql()}))'
        return f'{self.type.to_sql()}({self.term.to_sql()})'

###################################################

class BinOp(Term):
    def __init__(self, type: BinOpType, term1: 'Term', term2: 'Term'):
        super().__init__()
        self.type = type
        self.term1 = term1
        self.term2 = term2
    def __str__(self):
        res = f'({self.term1} '
        res += str(self.type).lower()
        res += f' {self.term2})'
        return res
    def to_sql(self):
        res = f'('
        res += ' ' + self.term1 if isinstance(self.term1, str) else self.term1.to_sql() + ' '
        res += str(self.type).upper()
        res += self.term2 if isinstance(self.term2, str) else self.term2.to_sql()
        res += ')'
        return res
    
###################################################
    
class If(Term):
    def __init__(self, condition: 'Term', term1: 'Term', term2: 'Term'):
        super().__init__()
        self.condition = condition
        self.term1 = term1
        self.term2 = term2
    def __str__(self):
        return f'if({self.condition}, {self.term1}, {self.term2})'
    def to_sql(self):
        return f'CASE WHEN ({self.condition.to_sql()}) THEN ({self.term1.to_sql()}) ELSE ({self.term2.to_sql()}) END'
    
###################################################
    
class External(Term):
    def __init__(self, type: ExternalType, args: List['Variable']):
        super().__init__()
        self.type = type
        self.args = args

    def __str__(self):
        res = f'{self.type}('
        for i in range(len(self.args)):
            if isinstance(self.args[i], list):
                res += '['
                for j in range(len(self.args[i])):
                    res += f'{self.args[i][j]}'
                    if j != len(self.args[i]) - 1:
                        res += ', '
                res += ']'
            else:
                res += str(self.args[i])
            if i != len(self.args) - 1:
                res += ', '
        res += ')'
        return res
    
    def to_sql(self):
        if self.type == ExternalType.YEAR:
            return f'EXTRACT(YEAR FROM {self.args[0].to_sql()})'
        elif self.type == ExternalType.SLICE:
            return f'substring({self.args[0].to_sql()} from {self.args[1].to_sql()} for {self.args[2].to_sql()})'
        elif self.type == ExternalType.UID:
            if self.args == None or len(self.args) == 0:
                return '(ROW_NUMBER() OVER())-1'
            else:
                res = f'(ROW_NUMBER() OVER (ORDER BY '
                for i in range(len(self.args[0])):
                    res += f'{self.args[0][i].to_sql()} '
                    res += 'ASC' if self.args[1][i] else 'DESC'
                    if i != len(self.args[0]) - 1:
                        res += ', '
                res += '))-1'
                return res
        elif self.type == ExternalType.LOWER:
            return f'LOWER({self.args[0].to_sql()})'
        elif self.type == ExternalType.UNIQUE:
            return f'DISTINCT({self.args[0].to_sql()})'
        elif self.type == ExternalType.ISNULL:
            return f'({self.args[0].to_sql()} IS NULL)'
        elif self.type == ExternalType.NOT:
            return f'(NOT {self.args[0].to_sql()})'
        elif self.type == ExternalType.INT:
            return f'FLOOR({self.args[0].to_sql()})'
        else:
            raise Exception('UDF not supported')

###################################################

class Integer(Constant):
    def __init__(self, value: int):
        super().__init__(value)
    def __str__(self):
        return str(self.value)
    def to_sql(self):
        return str(self.value)
    
###################################################
    
class String(Constant):
    def __init__(self, value: str):
        if not isinstance(value, str):
            raise Exception('String value must be a string')
        super().__init__(value)
    def __str__(self):
        return f"'{self.value}'"
    def to_sql(self):
        return f"'{self.value}'"

###################################################

class Date(Constant):
    def __init__(self, value: str):
        super().__init__(value)
    def __str__(self):
        return f'"{self.value}"'
    def to_sql(self):
        return f"date '{self.value}'"

###################################################
    
class Boolean(Constant):
    def __init__(self, value: bool):
        super().__init__(value)
    def __str__(self):
        return 'true' if self.value else 'false'
    def to_sql(self):
        return 'TRUE' if self.value else 'FALSE'
    
###################################################

class Float(Constant):
    def __init__(self, value: float):
        super().__init__(value)
    def __str__(self):
        return f'{self.value:.4f}'
    def to_sql(self):
        return f'{self.value:.4f}'
    
###################################################

class Null(Constant):
    def __init__(self):
        super().__init__(None)
    def __str__(self):
        return 'null'
    def to_sql(self):
        return 'NULL'

###################################################

class Atom(Base):
    def __init__(self):
        super().__init__()

###################################################

class ConstantRelation(Atom):
    def __init__(self, name, vars: list['Variable'], values: List[Union[Union[Integer, String, Boolean, Float], List[Union[Integer, String, Boolean, Float]]]]):
        super().__init__()
        self.name = name
        self.vars = vars
        self.values = values

        self.rows = 0
        self.cols = 0
        if len(values) > 0:
            if isinstance(values[0], list):
                self.rows = len(values)
                self.cols = len(values[0])
            else:
                self.rows = len(values)
                self.cols = 1

    def __str__(self):
        res = f'{self.name}({",".join(str(var) for var in self.vars)})['
        for i in range(self.rows):
            res += '['
            for j in range(self.cols):
                res += f'{self.values[i][j]}' if isinstance(self.values[i], list) else f'{self.values[i]}'
                if j != self.cols - 1:
                    res += ', '
            res += ']'
            if i != self.rows - 1:
                res += ', '
        res += ']'
        return res
    
    def to_sql(self):
        res = '(VALUES '
        for i in range(self.rows):
            res += '('
            for j in range(self.cols):
                res += f'{self.values[i][j]}' if isinstance(self.values[i], list) else f'{self.values[i]}'
                if j != self.cols - 1:
                    res += ', '
            res += ')'
            if i != self.rows - 1:
                res += ', '
        res += ')'
        return res

###################################################

class Theta(Atom):
    def __init__(self, type: ThetaType, term1: 'Term', term2: 'Term'):
        super().__init__()
        self.type = type
        self.term1 = term1
        self.term2 = term2

    def __str__(self):
        res = f'{self.term1}'
        if self.type == ThetaType.ASSIGN:
            res += f' <- '
        else:
            res += str(self.type).lower()
        res += f'{self.term2}'
        return res
    def to_sql(self):           
        if isinstance(self.term2, Null):
            if self.type == ThetaType.EQ:
                return f'({self.term1.to_sql()} IS NULL)'
            elif self.type == ThetaType.NE:
                return f'({self.term1.to_sql()} IS NOT NULL)'
        
        right_true = isinstance(self.term2, Boolean) and self.term2.value == True
        
        res = "("
        res += f'{self.term1.to_sql()}'
        res += f' ' if not right_true else ''
        res += f'{self.type}' if not right_true else ''
        res += f' {self.term2.to_sql()}' if not right_true else ''
        res += ")"
        
        return res

###################################################
    
class Exist(Atom):
    def __init__(self, body: List['Atom']):
        super().__init__()
        self.body = body
    def __str__(self):
        return f'exist({", ".join(str(x) for x in self.body)})'
    def to_sql(self):
        if not isinstance(self.body, list):
            raise Exception('Exist body must be a list of Atoms')

        rel_access = [atom for atom in self.body if isinstance(atom, RelationAccess)]
        condition = [atom for atom in self.body if isinstance(atom, Theta) and atom.type == ThetaType.EQ]
        condition_left = [atom.term1 for atom in condition]
        condition_right = [atom.term2 for atom in condition]

        res = 'SELECT * FROM '
        res += rel_access[0].name + ' AS ' + rel_access[0].name
        res += '('
        res += ', '.join([var.name for var in rel_access[0].vars])
        res += ')'
        res += ' WHERE '
        res += condition_left[0].to_sql()
        res += ' = '
        res += condition_right[0].to_sql()

        return f'EXISTS ({res})'
    
###################################################

class Outer(Atom):
    def __init__(self, type: OuterType, left_var: 'Variable', right_var: 'Variable'):
        super().__init__()
        self.type = type
        self.left_var = left_var
        self.right_var = right_var

    def __str__(self):
        return f'outer({self.type}, {self.left_var}, {self.right_var})'

###################################################

class Variable(Base):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    def __str__(self):
        return self.name
    def to_sql(self):
        return self.name

###################################################
    
class Relation(Base):
    def __init__(self, name: str, cols: List[str], types: List[Constant], index_col_idx: int, pks=None, fks=None, py_index=None):
        super().__init__()
        self.name = name
        self.cols = cols
        self.types = types
        self.index = index_col_idx
        self.pks = pks
        self.fks = fks
        self.py_index = py_index

###################################################

class RelationAccess(Atom):
    def __init__(self, name: str, vars: List[Variable]):
        super().__init__()
        self.name = name
        self.vars = vars
    def __str__(self):
        return f'{self.name}({", ".join(str(x) for x in self.vars)})'

###################################################
    
class Rule(Base):
    def __init__(self, rel_access: RelationAccess, group_vars: List[Variable], sort_vars: List[Variable], sort_order: List[bool], limit: int, body: List[Atom]):
        super().__init__()
        self.group_vars = group_vars
        self.sort_vars = sort_vars 
        self.sort_order = sort_order    # True for ASC, False for DESC
        self.limit = limit
        self.body = body
        self.rel_access = rel_access
        
    def __str__(self):
        res = f'{self.rel_access} '
        if self.group_vars:
            res += f'group({", ".join(str(x) for x in self.group_vars)}) '
        if self.sort_vars:
            res += f'sort([{", ".join(str(x) for x in self.sort_vars)}], [{", ".join(str(x).lower() for x in self.sort_order)}]) '
        if self.limit:
            res += f'limit({self.limit}) '
        res += f' :- {", ".join(str(x) for x in self.body)}'
        return res
    
    def to_sql(self, data_context: dict):

        # SQL Variables
        cols = []
        cols_aliases = []
        sql_assignments = {}

        rel_access = self.rel_access
        assignments = [self.body[i] for i in range(len(self.body)) if isinstance(self.body[i], Theta) and self.body[i].type == ThetaType.ASSIGN]
        rel = data_context[rel_access.name]

        ## Handling Ordering
        order_cols = self.sort_vars
        order_dirs = self.sort_order
        has_ordering = True if len(order_cols) > 0 else False
        has_limit = True if self.limit else False

        has_index = True if rel.index != None else False

        if has_ordering:
            if has_index:
                target_uid_generator_assignment = None
                for assignment in assignments:
                    if isinstance(assignment.term1, Variable) and assignment.term1.name == rel_access.vars[0].name:
                        if isinstance(assignment.term2, External) and assignment.term2.type == ExternalType.UID:
                            target_uid_generator_assignment = assignment
                            break
                if target_uid_generator_assignment:
                    target_uid_generator_assignment_body_copy = copy.deepcopy(target_uid_generator_assignment.term2)
                    target_uid_generator_assignment_body_copy.args = [order_cols, order_dirs]
                    sql_assignments[rel_access.vars[0].name] = target_uid_generator_assignment_body_copy.to_sql()

        ## Handling Assignments
        for assignment in assignments:
            if assignment.term1.name not in sql_assignments:
                sql_assignments[assignment.term1.name] = assignment.term2.to_sql()

        for var in rel_access.vars:
            if var.name in sql_assignments:
                cols.append(sql_assignments[var.name])
                cols_aliases.append(var.name)
            else:
                cols.append(var)
                cols_aliases.append(None)

        relations = [atom.name for atom in self.body if isinstance(atom, RelationAccess)]
        relations_cols = [[var.name for var in atom.vars] for atom in self.body if isinstance(atom, RelationAccess)]
        constant_relations = [atom for atom in self.body if isinstance(atom, ConstantRelation)]
        constant_relations_cols = [[var.name for var in atom.vars] for atom in constant_relations]
        conditions = [atom.to_sql() for atom in self.body if (isinstance(atom, Theta) and atom.type != ThetaType.ASSIGN) or isinstance(atom, Exist)]
        group_cols = [var.name for var in self.group_vars]
        outer = [atom for atom in self.body if isinstance(atom, Outer)]


        res = '\tSELECT '
        for i in range(len(cols)):
            if cols_aliases[i]:
                res += f'{cols[i]} AS {cols_aliases[i]}'
            else:
                res += str(cols[i])
            if i != len(cols) - 1:
                res += ', '
        res += '\n\tFROM '
        all_relation = relations + constant_relations
        all_relation_cols = relations_cols + constant_relations_cols
        
        relation_name_collision_counter = {}
        for i in range(len(all_relation)):

            if isinstance(all_relation[i], str):
                if all_relation[i] in relation_name_collision_counter:
                    relation_name_collision_counter[all_relation[i]] += 1
                else:
                    relation_name_collision_counter[all_relation[i]] = 0

                res += all_relation[i]
                alias = all_relation[i] + "_" + str(relation_name_collision_counter[all_relation[i]]) if relation_name_collision_counter[all_relation[i]] > 0 else all_relation[i]
                res += f' AS {alias}'
                res += f'({", ".join(all_relation_cols[i])})'
            elif isinstance(all_relation[i], ConstantRelation):
                res += all_relation[i].to_sql()
                res += f' AS {all_relation[i].name}'
                res += f'({", ".join(all_relation_cols[i])})'

            if i != len(all_relation) - 1:
                if outer != []:
                    res += '\n\t'
                    res += outer[0].type.to_sql()
                    res += ' '
                else:
                    res += ', '

        if outer != []:
            res += ' ON '
            res += f'{outer[0].left_var.to_sql()} = {outer[0].right_var.to_sql()}'


        if len(conditions) > 0:
            res += '\n\tWHERE ('
            res += ') AND ('.join(conditions)
            res += ')'

        if len(group_cols) > 0:
            res += '\n\tGROUP BY '
            res += ', '.join(group_cols)

        if has_limit:
            if has_ordering:
                if not global_context['is_last_rule']:
                    if not has_index:
                            res += '\n\tORDER BY '
                            for i in range(len(order_cols)):
                                res += f'{order_cols[i].name}'
                                res += ' ASC' if order_dirs[i] else ' DESC'
                                if i != len(order_cols) - 1:
                                    res += ', '
                            res += f'\n\tLIMIT {self.limit}'
            else:
                    res += f'\n\tORDER BY ' + rel.cols[0]
                    res += f'\n\tLIMIT {self.limit}'
        return res
    
###################################################
    
class Program(Base):
    def __init__(self, name: str, rules: List[Rule]):
        super().__init__()
        self.name = name
        self.rules = rules

        self.final_rel = None
        self.sql_final_order_columns = []
        self.sql_final_order_directions = []
        self.limit = None

    def __str__(self):
        return "\n".join(str(x) for x in self.rules)
    
    def to_sql(self, data_context: dict):
        global global_context

        rules = self.rules

        self.final_rel = data_context[rules[-1].rel_access.name]
        self.sql_final_order_columns = rules[-1].sort_vars
        self.sql_final_order_directions = rules[-1].sort_order
        self.limit = rules[-1].limit
        has_index = True if self.final_rel.index != None else False

        res = f'-- Query {self.name}\nWITH '
        for i in range(len(rules)):
            res += f'{rules[i].rel_access.name}('
            res += ', '.join([var.name for var in rules[i].rel_access.vars])
            res += f') AS (\n'
            if i == len(rules) - 1:
                global_context['is_last_rule'] = True
            res += rules[i].to_sql(data_context)
            global_context['is_last_rule'] = False
            if i != len(rules) - 1:
                res += '\n),\n'
            else:
                res += '\n)'

        cols_index_to_show = [i for i in range(len(self.final_rel.cols))]
        py_index = self.final_rel.py_index if self.final_rel.py_index else []
        cols_index_to_show = [i for i in cols_index_to_show if i not in py_index]
        cols_to_show = [self.final_rel.cols[i] for i in cols_index_to_show]
        res += '\nSELECT ' + ('0 AS ID, ' if not has_index else '')  + ', '.join(cols_to_show) + ' FROM ' + self.final_rel.name + ' AS ' + self.final_rel.name + '(' + ', '.join(self.final_rel.cols) + ')'

        if self.sql_final_order_columns:
            if not has_index:
                res += '\nORDER BY '
                for i in range(len(self.sql_final_order_columns)):
                    var = self.sql_final_order_columns[i].name
                    var_names = [var.name for var in self.rules[-1].rel_access.vars]
                    col_name = self.final_rel.cols[var_names.index(var)]
                    res += f'{col_name}'
                    res += ' ASC' if self.sql_final_order_directions[i] else ' DESC'
                    if i != len(self.sql_final_order_columns) - 1:
                        res += ', '
            else:
                res += ' ORDER BY ID ASC'
        else:
            if has_index:
                res += ' ORDER BY ID ASC'
                       
        if self.limit:
            res += f'\nLIMIT {self.limit}'

        return res

###################################################
