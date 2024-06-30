import os
import ast
import copy
from pytond.tondir import *
from pytond.optimizer import optimize

##################################################################################
# TEMPLATE FOR DECORATOR
# @pytond(input_indexes={}, input_layouts={}, pivot_column_values={})
##################################################################################
debug_mode = False
##################################################################################

var_counter = {'x': 1, 'v': 1, 'V': 1}
anf_result_cache = []
tondir_global_context = {}

#### Helper Functions ###########################################################

def reset_tondir_global_context():
    global tondir_global_context
    tondir_global_context = {
        'database': {},
        'var_mapping': {},
        'created_rules': [],
        'decorator_args': {}
    }

def print_ast(node):
    print(ast.dump(node), end='\n')    

def fresh_var(type='x'):
        global var_counter
        res = f'{type}{var_counter[type]}'
        var_counter[type] += 1
        return res

def fresh_var_reset():
    global var_counter
    var_counter = {'x': 1, 'V': 1, 'v': 1}

def n_fresh_ir_vars(n):
    if n == 1:
        return Variable(fresh_var())
    return [Variable(fresh_var()) for _ in range(n)]

def store_in_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def check_anf_compliant_types(arg):
    if type(arg) in [ast.Name, ast.Constant, str, int, float, bool] or \
        isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) and isinstance(arg.operand, ast.Constant):
        return True
    return False

def create_intermediate_assignment(expr):
    global anf_result_cache
    var = fresh_var('v')
    anf_result_cache.append(ast.Assign(targets=[ast.Name(id=var, ctx=ast.Store())], value=expr))
    return  ast.Name(id=var, ctx=ast.Load())

def is_ast_list_of_constants(expr):
    if not isinstance(expr, ast.List):
        return False
    res = True
    for el in expr.elts:
        if isinstance(el, ast.Constant) or \
            isinstance(el, ast.UnaryOp) and isinstance(el.op, ast.USub) and isinstance(el.operand, ast.Constant):
            continue
        else:
            res = False
            break
    return res

def is_ast_dict_of_constants(expr):
    if not isinstance(expr, ast.Dict):
        return False
    res = True
    for key, value in zip(expr.keys, expr.values):
        if isinstance(key, ast.Constant) or \
            isinstance(key, ast.UnaryOp) and isinstance(key.op, ast.USub) and isinstance(key.operand, ast.Constant) or \
            isinstance(value, ast.Constant) or \
            isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub) and isinstance(value.operand, ast.Constant):
            continue
        else:
            res = False
            break
    return res

def get_index_of_column(str_list, col):
    for i in range(len(str_list)):
        if str_list[i] == col:
            return i
    return -1

def get_index_of_ir_var(var_list, var):
    for i in range(len(var_list)):
        if var_list[i].name == var.name:
            return i
    return -1

def get_ir_typed_constant(value):
    if isinstance(value, str):
        if len(value.split('-')) == 3:
            return Date(value)
        return String(value)
    elif isinstance(value, int):
        return Integer(value)
    elif isinstance(value, float):
        return Float(value)
    elif isinstance(value, bool):
        return Boolean(value)
    else:
        raise Exception('Invalid constant type')

def get_mapped_rel_name(name):
    global tondir_global_context
    if name in tondir_global_context['var_mapping']:
        return tondir_global_context['var_mapping'][name]
    return name

def get_mapped_var(col_name, cols_list, vars_list):
    if col_name in cols_list:
        return vars_list[get_index_of_column(cols_list, col_name)]
    raise Exception('Column not found in list', col_name, cols_list)

def get_or_create_index_var(relation, relation_access, current_body):
    index_var = None
    if relation.index is None:
        atom = Theta(ThetaType.ASSIGN, n_fresh_ir_vars(1), External(ExternalType.UID, []))
        current_body.append(atom)
        index_var = atom.term1
    else:
        index_var = relation_access.vars[relation.index]
    return current_body, index_var

def create_rule(relation_access, groupby, sort_vars, sort_order, sort_limit, body):
    global tondir_global_context
    rule = Rule(relation_access, groupby, sort_vars, sort_order, sort_limit, body)
    tondir_global_context['created_rules'].append(rule)
    return rule

def prepare_body_relation(rel_name, body):

    body_rel = tondir_global_context['database'][get_mapped_rel_name(rel_name)]

    if body_rel.index is not None:
        body_rel_access = RelationAccess(body_rel.name, n_fresh_ir_vars(len(body_rel.cols)))
        body.append(body_rel_access)

    # Create a new relation and with index
    else:
        # Create a new relation and rule
        body_rel_access = RelationAccess(body_rel.name, n_fresh_ir_vars(len(body_rel.cols)))
        index_var = n_fresh_ir_vars(1)
        new_rel = Relation(fresh_var('V'), ['ID'] + body_rel.cols, None, 0)
        tondir_global_context['database'][new_rel.name] = new_rel
        new_rel_access = RelationAccess(new_rel.name, [index_var] + body_rel_access.vars)
        pks_of_base_rel = body_rel.pks
        create_rule(new_rel_access, [], [], [], None, [body_rel_access, Theta(ThetaType.ASSIGN, index_var, External(ExternalType.UID, []))])
        tondir_global_context['var_mapping'][rel_name] = new_rel.name

        new_new_rel_access = RelationAccess(new_rel.name, n_fresh_ir_vars(len(new_rel.cols)))
        body.append(new_new_rel_access)
        body_rel = new_rel
        body_rel_access = new_new_rel_access

    return body, body_rel, body_rel_access, body_rel_access.vars[0]

def check_skipped_call_rule_generation(call_node_func_name, call_node_attr_name, call_node_args):
    if call_node_attr_name == 'DataFrame' and call_node_args == []:
        return True

#### Primary Functions ##########################################################

def ast_expression_to_tondir(node, context):
    
    if isinstance(node, ast.IfExp):
        return If(ast_expression_to_tondir(node.test, context), ast_expression_to_tondir(node.body, context), ast_expression_to_tondir(node.orelse, context))
    elif isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Constant) and isinstance(node.value, ast.Name) and context['var_mapping'][node.value.id] is not None:
            col_idx = get_index_of_column(context['rel'].cols, node.slice.value)
            if col_idx == -1:
                raise Exception('Column not found in relation')
            return context['rel_access'].vars[col_idx]
    elif isinstance(node, ast.Compare):
        op = None
        if isinstance(node.ops[0], ast.Eq):
            op = ThetaType.EQ
        elif isinstance(node.ops[0], ast.NotEq):
            op = ThetaType.NE
        elif isinstance(node.ops[0], ast.Lt):
            op = ThetaType.LT
        elif isinstance(node.ops[0], ast.LtE):
            op = ThetaType.LE
        elif isinstance(node.ops[0], ast.Gt):
            op = ThetaType.GT
        elif isinstance(node.ops[0], ast.GtE):
            op = ThetaType.GE
        elif isinstance(node.ops[0], ast.In):
            op = ThetaType.IN
        else:   
            print_ast(node)
            raise Exception('Invalid comparison operation')

        left = ast_expression_to_tondir(node.left, context)
        right = ast_expression_to_tondir(node.comparators[0], context)
        if op != ThetaType.IN:
            return Theta(op, left, right)
        else:
            if isinstance(left, String):
                left.value = '%' + left.value + '%'
                return BinOp(BinOpType.LIKE, right, left)
            else:
                raise Exception('Invalid IN operation')
    elif isinstance(node, ast.Constant):
        return get_ir_typed_constant(node.value)
    elif isinstance(node, ast.BinOp):
        op = None
        if isinstance(node.op, ast.Add):
            op = BinOpType.ADD
        elif isinstance(node.op, ast.Sub):
            op = BinOpType.SUB
        elif isinstance(node.op, ast.Mult):
            op = BinOpType.MUL
        elif isinstance(node.op, ast.Div):
            op = BinOpType.DIV
        elif isinstance(node.op, ast.BitAnd):
            op = BinOpType.AND
        elif isinstance(node.op, ast.BitOr):
            op = BinOpType.OR
        else:
            raise Exception('Invalid binary operation')

        left = ast_expression_to_tondir(node.left, context)
        right = ast_expression_to_tondir(node.right, context)
        return BinOp(op, left, right)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'startswith':
                if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                    return BinOp(BinOpType.LIKE, ast_expression_to_tondir(node.func.value, context), "'" + node.args[0].value + "%'")
                else:
                    raise Exception('Invalid function call')
            if node.func.attr == 'lower':
                return External(ExternalType.LOWER, [ast_expression_to_tondir(node.func.value, context)])
            else:
                raise Exception('Invalid function call')
    elif isinstance(node, ast.Name):
        if tondir_global_context['var_mapping'][node.id] is not None:
            return Variable(tondir_global_context['var_mapping'][node.id])
        else:
            return Variable(node.id)
    else:
        print_ast(node)
        raise Exception('Invalid expression')

def line_anf_to_tondir(node, context):

    # Contextual Fields to Be Filled
    ## t_rule_body
    ## t_rule_rel_cols
    ## t_rule_rel_vars
    ## t_rule_rel_index
    ## t_rule_groupby
    ## t_rule_sort_vars
    ## t_rule_sort_order
    ## t_rule_sort_limit

    if isinstance(node, ast.Name):
        if context['p_target_type'] == 'sub':

            ## t_rule_body
            left_rel_name = context['p_target_var_name']
            left_col_name = context['p_target_sub_var_name']
            body = []
            body, body_rel_right, body_rel_access_right, index_right = prepare_body_relation(node.id, body)

            # Normal Case
            if get_mapped_rel_name(left_rel_name) in tondir_global_context['database']:

                body, body_rel_left, body_rel_access_left, index_left = prepare_body_relation(left_rel_name, body)

                ## Join Condition
                body.append(Theta(ThetaType.EQ, index_left, index_right))

                ## Assigning Value
                target_var_idx = get_index_of_column(body_rel_left.cols, left_col_name)
                is_a_replace = left_col_name in body_rel_left.cols 
                new_var = n_fresh_ir_vars(1)
                body.append(Theta(ThetaType.ASSIGN, new_var, body_rel_access_right.vars[1]))

                context['t_rule_body'] = body

                ## t_rule_rel_cols
                context['t_rule_rel_cols'] = body_rel_left.cols + [left_col_name] if not is_a_replace else body_rel_left.cols

                ## t_rule_rel_vars
                if not is_a_replace:
                    context['t_rule_rel_vars'] = body_rel_access_left.vars + [new_var]
                else:
                    context['t_rule_rel_vars'] = body_rel_access_left.vars[:target_var_idx] + [new_var] + body_rel_access_left.vars[target_var_idx + 1:]

                ## t_rule_rel_index
                context['t_rule_rel_index'] = 0

                ## t_rule_groupby
                ## t_rule_sort_vars
                ## t_rule_sort_order
                ## t_rule_sort_limit
                # NO ACTION

            # targeting pd.DataFrame() case
            else:

                # Assigning Value
                value_var = n_fresh_ir_vars(1)
                body.append(Theta(ThetaType.ASSIGN, value_var, body_rel_access_right.vars[1]))

                context['t_rule_body'] = body

                ## t_rule_rel_cols
                context['t_rule_rel_cols'] = ['ID'] + [left_col_name]

                ## t_rule_rel_vars
                context['t_rule_rel_vars'] = body_rel_access_right.vars

                ## t_rule_rel_index
                context['t_rule_rel_index'] = 0

                ## t_rule_groupby
                ## t_rule_sort_vars
                ## t_rule_sort_order
                ## t_rule_sort_limit
                # NO ACTION

        else:
            context['skip_rule_generation'] = True
            tondir_global_context['var_mapping'][context['p_target_var_name']] = get_mapped_rel_name(node.id)

    elif isinstance(node, ast.Attribute):

        target_col_name = node.attr
        if (target_col_name in ['str', 'dt']):
            context['skip_rule_generation'] = True
            tondir_global_context['var_mapping'][context['p_target_var_name']] = get_mapped_rel_name(node.value.id)
            return

        ## preparing t_rule_body
        body = []
        body, body_rel, body_rel_access, index = prepare_body_relation(node.value.id, body)

        reserved_names = ['year']
        if (target_col_name in reserved_names and target_col_name not in body_rel.cols):
            if target_col_name == 'year':
                
                value_var = n_fresh_ir_vars(1)
                body.append(Theta(ThetaType.ASSIGN, value_var, External(ExternalType.YEAR, [body_rel_access.vars[1]])))

            else:
                raise Exception('Invalid attribute')

        context['t_rule_body'] = body

        ## preparing t_rule_rel_cols
        context['t_rule_rel_cols'] = ['ID', target_col_name]

        ## preparing t_rule_rel_vars
        if target_col_name not in reserved_names:
            target_var = get_mapped_var(target_col_name, body_rel.cols, body_rel_access.vars)
        else:
            target_var = value_var
 
        context['t_rule_rel_vars'] = [index, target_var]

        ## preparing t_rule_rel_index
        context['t_rule_rel_index'] = 0

        ## preparing t_rule_groupby
        ## preparing t_rule_sort_vars
        ## preparing t_rule_sort_order
        ## preparing t_rule_sort_limit
        # NO ACTION
   
    elif isinstance(node, ast.Compare):

        # v cmp v
        if isinstance(node.left, ast.Name) and isinstance(node.comparators[0], ast.Name):

            ## t_rule_body

            body = []
            body, body_rel_left, body_rel_access_left, index_left = prepare_body_relation(node.left.id, body)
            body, body_rel_right, body_rel_access_right, index_right = prepare_body_relation(node.comparators[0].id, body)

            ## Join Condition
            body.append(Theta(ThetaType.EQ, index_left, index_right))

            ## COMPARISON BODY HERE
            if node.ops[0].__class__.__name__ in ['Lt', 'LtE', 'Gt', 'GtE', 'Eq', 'NotEq']:
                compare_type = None
                if node.ops[0].__class__.__name__ == 'Lt':
                    compare_type = ThetaType.LT
                elif node.ops[0].__class__.__name__ == 'LtE':
                    compare_type = ThetaType.LE
                elif node.ops[0].__class__.__name__ == 'Gt':
                    compare_type = ThetaType.GT
                elif node.ops[0].__class__.__name__ == 'GtE':
                    compare_type = ThetaType.GE
                elif node.ops[0].__class__.__name__ == 'Eq':
                    compare_type = ThetaType.EQ
                elif node.ops[0].__class__.__name__ == 'NotEq':
                    compare_type = ThetaType.NE

                value_var = n_fresh_ir_vars(1)
                body.append(Theta(ThetaType.ASSIGN, value_var, Theta(compare_type, body_rel_access_left.vars[1], body_rel_access_right.vars[1])))

            context['t_rule_body'] = body

            ## t_rule_rel_cols
            context['t_rule_rel_cols'] = ['ID', 'Value']

            ## t_rule_rel_vars
            context['t_rule_rel_vars'] = [index_left, value_var]

            ## t_rule_rel_index
            context['t_rule_rel_index'] = 0

            ## t_rule_groupby
            ## t_rule_sort_vars
            ## t_rule_sort_order
            ## t_rule_sort_limit
            # NO ACTION

        # v cmp non-v or non-v cmp v
        elif isinstance(node.left, ast.Name) or isinstance(node.comparators[0], ast.Name):

            ## t_rule_body
            var = node.left if isinstance(node.left, ast.Name) else node.comparators[0]
            non_var = node.comparators[0] if isinstance(node.left, ast.Name) else node.left

            body = []
            body, body_rel, body_rel_access, index = prepare_body_relation(var.id, body)

            ## COMPARISON BODY HERE
            if node.ops[0].__class__.__name__ in ['Lt', 'LtE', 'Gt', 'GtE', 'Eq', 'NotEq']:
                if isinstance(non_var, ast.Constant):
                    compare_type = None
                    if node.ops[0].__class__.__name__ == 'Lt':
                        compare_type = ThetaType.LT
                    elif node.ops[0].__class__.__name__ == 'LtE':
                        compare_type = ThetaType.LE
                    elif node.ops[0].__class__.__name__ == 'Gt':
                        compare_type = ThetaType.GT
                    elif node.ops[0].__class__.__name__ == 'GtE':
                        compare_type = ThetaType.GE
                    elif node.ops[0].__class__.__name__ == 'Eq':
                        compare_type = ThetaType.EQ
                    elif node.ops[0].__class__.__name__ == 'NotEq':
                        compare_type = ThetaType.NE

                    const = get_ir_typed_constant(non_var.value)
                    value_var = body_rel_access.vars[1]
                    new_value_var = n_fresh_ir_vars(1)
                    assignment = Theta(ThetaType.ASSIGN, new_value_var, Theta(compare_type, value_var, const))
                    body.append(assignment)

            context['t_rule_body'] = body

            ## t_rule_rel_cols
            context['t_rule_rel_cols'] = ['ID', 'Value']

            ## t_rule_rel_vars
            context['t_rule_rel_vars'] = [index, new_value_var]

            ## t_rule_rel_index
            context['t_rule_rel_index'] = 0

            ## t_rule_groupby
            ## t_rule_sort_vars
            ## t_rule_sort_order
            ## t_rule_sort_limit
            # NO ACTION

        # constant compare constant
        else:
            context['skip_rule_generation'] = True
    
    elif isinstance(node, ast.Call):
        # x.func()
        if isinstance(node.func, ast.Attribute):
            ## t_rule_body
            body = []

            if check_skipped_call_rule_generation(node.func.value.id, node.func.attr, node.args):
                context['skip_rule_generation'] = True
                return
                
            if not node.func.value.id in ['pd', 'np']:
                body, body_rel, body_rel_access, index = prepare_body_relation(node.func.value.id, body)

            if node.func.attr == 'isnull':
                value_var = n_fresh_ir_vars(1)
                body.append(Theta(ThetaType.ASSIGN, value_var, External(ExternalType.ISNULL, [body_rel_access.vars[1]])))
                context['t_rule_body'] = body
                ## t_rule_rel_cols
                context['t_rule_rel_cols'] = ['ID', 'Value']
                ## t_rule_rel_vars
                context['t_rule_rel_vars'] = [index, value_var]
                ## t_rule_rel_index
                context['t_rule_rel_index'] = 0
                ## t_rule_groupby
                ## t_rule_sort_vars
                ## t_rule_sort_order
                ## t_rule_sort_limit
                # NO ACTION
            elif node.func.attr == 'groupby':

                # Reset index
                new_index = n_fresh_ir_vars(1)
                body.append(Theta(ThetaType.ASSIGN, new_index, External(ExternalType.UID, [])))

                context['t_rule_body'] = body

                # t_rule_groupby
                cols = []
                for arg in node.args:
                    for col in arg.elts:
                        if isinstance(col, ast.Constant):
                            cols.append(col.value)
                        else:
                            raise Exception('Invalid groupby argument')
                
                mapped_groupby_vars = [get_mapped_var(col, body_rel.cols, body_rel_access.vars) for col in cols]
                context['t_rule_groupby'] = mapped_groupby_vars

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = ['ID'] + cols

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = [new_index] + mapped_groupby_vars

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.attr == 'agg':

                # find the related rule
                rule_idx = None
                rule = None
                for i in range(len(tondir_global_context['created_rules'])):
                    if tondir_global_context['created_rules'][i].rel_access.name == get_mapped_rel_name(node.func.value.id):
                        rule_idx = i
                        rule = tondir_global_context['created_rules'][i]
                        break
                if rule_idx is None:
                    raise Exception('Rule not found for aggregation')
                
                previous_head_rel = tondir_global_context['database'][rule.rel_access.name]
                previous_head_rel_access = rule.rel_access
                previous_body_rel = None
                previous_body_rel_access = None
                for item in rule.body:
                    if item.__class__.__name__ == 'RelationAccess':
                        previous_body_rel = tondir_global_context['database'][item.name]
                        previous_body_rel_access = item
                        break

                # Aggregation
                if tondir_global_context['created_rules'][i].group_vars == []:
                    raise Exception('Pure aggregation not implemented')

                # Groupby / Aggregation
                else:
                    inp_cols_names = []
                    agg_funcs = []
                    out_col_names = []
                    for kw in node.keywords:
                        inp_cols_names.append(kw.value.elts[0].value)
                        out_col_names.append(kw.arg)
                        if isinstance(kw.value.elts[1], ast.Constant):
                            agg_funcs.append(kw.value.elts[1].value)
                        elif isinstance(kw.value.elts[1], ast.Lambda):
                            if isinstance(kw.value.elts[1].body, ast.Call):
                                if isinstance(kw.value.elts[1].body.func, ast.Attribute):
                                    if kw.value.elts[1].body.func.attr == 'nunique':
                                        agg_funcs.append('count_distinct')
                                    else:
                                        raise Exception('Invalid aggregation function')
                                else:
                                    raise Exception('Invalid aggregation function')
                        else:
                            raise Exception('Invalid aggregation function')


                    # Aggregation
                    tmp_body = []
                    tmp_cols = []
                    tmp_vars = []

                    for i in range(len(inp_cols_names)):
                        col = inp_cols_names[i]
                        agg_func = agg_funcs[i]
                        col_var = get_mapped_var(col, previous_body_rel.cols, previous_body_rel_access.vars)
                        agg_var = n_fresh_ir_vars(1)
                        if agg_func == 'sum':
                            tmp_body.append(Theta(ThetaType.ASSIGN, agg_var, Aggregate(AggregateType.SUM, col_var)))
                        elif agg_func == 'count':
                            tmp_body.append(Theta(ThetaType.ASSIGN, agg_var, Aggregate(AggregateType.COUNT, col_var)))
                        elif agg_func == 'mean':
                            tmp_body.append(Theta(ThetaType.ASSIGN, agg_var, Aggregate(AggregateType.AVG, col_var)))
                        elif agg_func == 'min':
                            tmp_body.append(Theta(ThetaType.ASSIGN, agg_var, Aggregate(AggregateType.MIN, col_var)))
                        elif agg_func == 'max':
                            tmp_body.append(Theta(ThetaType.ASSIGN, agg_var, Aggregate(AggregateType.MAX, col_var)))
                        elif agg_func == 'count_distinct':
                            tmp_body.append(Theta(ThetaType.ASSIGN, agg_var, Aggregate(AggregateType.COUNT_DISTINCT, col_var)))
                        else:
                            raise Exception('Invalid aggregation function', agg_func)

                        tmp_cols.append(out_col_names[i])
                        tmp_vars.append(agg_var)

                    body = rule.body + tmp_body
                    context['t_rule_body'] = body

                    # t_rule_rel_cols

                    context['t_rule_rel_cols'] = previous_head_rel.cols + tmp_cols

                    # t_rule_rel_vars
                    context['t_rule_rel_vars'] = previous_head_rel_access.vars + tmp_vars

                    # t_rule_rel_index
                    context['t_rule_rel_index'] = 0

                    # t_rule_groupby
                    context['t_rule_groupby'] = rule.group_vars

                    tondir_global_context['created_rules'].remove(rule)

                    # t_rule_sort_vars
                    # t_rule_sort_order
                    # t_rule_sort_limit
                    # NO ACTION
            elif node.func.attr == 'sort_values':
                context['t_rule_body'] = body



                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit

                sort_cols = []
                sort_dirs = []
                sort_vars = []
                for kw in node.keywords:
                    if kw.arg == 'by':
                        for col in kw.value.elts:
                            sort_cols.append(col.value)
                    elif kw.arg == 'ascending':
                        for dir in kw.value.elts:
                            sort_dirs.append(dir.value)

                for i in range(len(sort_cols)):
                    sort_vars.append(get_mapped_var(sort_cols[i], body_rel.cols, body_rel_access.vars))
                    sort_dirs[i] = True if sort_dirs[i] else False

                # Reset index
                body.append(Theta(ThetaType.ASSIGN, body_rel_access.vars[0], External(ExternalType.UID, [])))

                context['t_rule_sort_vars'] = sort_vars
                context['t_rule_sort_order'] = sort_dirs

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = body_rel.cols

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = body_rel_access.vars

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_groupby
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.attr in ['sum', 'max', 'mean']:

                is_horizontal_agg = len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 1 

                # Aggregation
                sum_var = n_fresh_ir_vars(1)

                agg_type = None
                if node.func.attr == 'sum':
                    agg_type = AggregateType.SUM
                elif node.func.attr == 'max':
                    agg_type = AggregateType.MAX
                elif node.func.attr == 'mean':
                    agg_type = AggregateType.AVG
                else:
                    raise Exception('Invalid aggregation function')

                if is_horizontal_agg:
                    expr = None

                    target_col_idxs = [i for i in range(len(body_rel.cols))]
                    target_col_idxs.remove(0)
                    if body_rel.py_index is not None:
                        for i in body_rel.py_index:
                            target_col_idxs.remove(i)

                    if len(target_col_idxs) == 0:
                        raise Exception('No column to aggregate')
                    elif len(target_col_idxs) == 1:
                        expr = body_rel_access.vars[target_col_idxs[0]]
                    else:
                        expr = BinOp(BinOpType.ADD, body_rel_access.vars[target_col_idxs[0]], body_rel_access.vars[target_col_idxs[1]])
                        for i in range(2, len(target_col_idxs)):
                            expr = BinOp(BinOpType.ADD, expr, body_rel_access.vars[i])

                    body.append(Theta(ThetaType.ASSIGN, sum_var, expr))

                else:
                    body.append(Theta(ThetaType.ASSIGN, sum_var, Aggregate(agg_type, body_rel_access.vars[1])))

                if not is_horizontal_agg:
                    # ID Reset
                    new_id = n_fresh_ir_vars(1)
                    body.append(Theta(ThetaType.ASSIGN, new_id, External(ExternalType.UID, [])))

                context['t_rule_body'] = body

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = ['ID', 'Value']

                # t_rule_rel_vars
                if is_horizontal_agg:
                    context['t_rule_rel_vars'] = [body_rel_access.vars[0], sum_var]
                else:
                    context['t_rule_rel_vars'] = [new_id, sum_var]

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_rel_py_index
                context['t_rule_rel_py_index'] = body_rel.py_index

                # t_rule_groupby
                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.attr == 'merge':
                
                left_rel = body_rel
                left_rel_access = body_rel_access
                left_index = index

                right_rel_name = node.args[0].id
                body, right_rel, right_rel_access, right_index = prepare_body_relation(right_rel_name, body)

                # Extract Keywords Arguments

                how = 'inner'
                left_on = []
                right_on = []

                for kw in node.keywords:
                    if kw.arg == 'how':
                        how = kw.value.value
                    elif kw.arg == 'left_on':
                        if isinstance(kw.value, ast.Constant):
                            left_on.append(kw.value.value)
                        else:
                            for el in kw.value.elts:
                                left_on.append(el.value)
                    elif kw.arg == 'right_on':
                        if isinstance(kw.value, ast.Constant):
                            right_on.append(kw.value.value)
                        else:
                            for el in kw.value.elts:
                                right_on.append(el.value)

                if how == 'inner':

                    # Join Condition
                    joined_vars_left = []
                    joined_vars_right = []
                    joined_cols_left = []
                    joined_cols_right = []

                    lefts_for_id_sorting = []
                    for i in range(len(left_on)):
                        left_var = get_mapped_var(left_on[i], left_rel.cols, left_rel_access.vars)
                        right_var = get_mapped_var(right_on[i], right_rel.cols, right_rel_access.vars)
                        body.append(Theta(ThetaType.EQ, left_var, right_var))
                        joined_vars_left.append(left_var)
                        joined_vars_right.append(right_var)
                        joined_cols_left.append(left_on[i])
                        joined_cols_right.append(right_on[i])
                        lefts_for_id_sorting.append(left_var)

                    # Reset Index
                    index_var = n_fresh_ir_vars(1)
                    body.append(Theta(ThetaType.ASSIGN, index_var, External(ExternalType.UID, [])))

                    context['t_rule_body'] = body

                    # t_rule_rel_cols

                    final_cols = ['ID']
                    final_vars = [index_var]

                    left_covered_cols = [left_rel.cols[left_rel.index]]
                    right_covered_cols = [right_rel.cols[right_rel.index]]

                    for i in range(len(joined_cols_left)):
                        if joined_cols_left[i] == joined_cols_right[i] == 'ID':
                            continue
                        if joined_cols_left[i] == joined_cols_right[i]:
                            final_cols.append(joined_cols_left[i])
                            final_vars.append(joined_vars_left[i])
                        else:
                            final_cols.append(joined_cols_left[i])
                            final_cols.append(joined_cols_right[i])
                            final_vars.append(joined_vars_left[i])
                            final_vars.append(joined_vars_right[i])
                        left_covered_cols.append(joined_cols_left[i])
                        right_covered_cols.append(joined_cols_right[i])

                    common_cols = list(set(left_rel.cols).intersection(set(right_rel.cols)))

                    for i in range(len(left_rel.cols)):
                        if left_rel.cols[i] not in left_covered_cols:
                            final_cols.append(left_rel.cols[i] if left_rel.cols[i] not in common_cols else left_rel.cols[i] + '_x')
                            final_vars.append(left_rel_access.vars[i])

                    for i in range(len(right_rel.cols)):
                        if right_rel.cols[i] not in right_covered_cols:
                            final_cols.append(right_rel.cols[i] if right_rel.cols[i] not in common_cols else right_rel.cols[i] + '_y')
                            final_vars.append(right_rel_access.vars[i])

                    context['t_rule_rel_cols'] = final_cols
                    context['t_rule_rel_vars'] = final_vars

                    # t_rule_rel_index
                    context['t_rule_rel_index'] = 0

                    # t_rule_groupby
                    # t_rule_sort_vars
                    context['t_rule_sort_vars'] = lefts_for_id_sorting
                    # t_rule_sort_order
                    context['t_rule_sort_order'] = [True for i in range(len(lefts_for_id_sorting))]
                    # t_rule_sort_limit
                    # NO ACTION

                    if len(left_on) == 1 and len(right_on) == 1 and left_rel.pks is not None and right_rel.pks is not None:
                        left_on_col_name = left_on[0]
                        right_on_col_name = right_on[0]
                        left_pk = left_rel.pks[0] if len(left_rel.pks) == 1 else None
                        right_pk = right_rel.pks[0] if len(right_rel.pks) == 1 else None
                        if left_on_col_name == left_pk and right_on_col_name == right_pk:
                            uid_generation_thetas = [x for x in body if isinstance(x, Theta) and x.type == ThetaType.ASSIGN and isinstance(x.term2, External) and x.term2.type == ExternalType.UID]
                            body = [x for x in body if x not in uid_generation_thetas]
                            body.append(Theta(ThetaType.ASSIGN, index_var, left_var))
                            context['t_rule_body'] = body

                elif how == 'cross':

                    # Reset Index
                    index_var = n_fresh_ir_vars(1)
                    body.append(Theta(ThetaType.ASSIGN, index_var, External(ExternalType.UID, [])))

                    context['t_rule_body'] = body

                    # t_rule_rel_cols
                    # t_rule_rel_vars

                    final_cols = ['ID']
                    final_vars = [index_var]

                    for i in range(len(left_rel.cols)):
                        if left_rel.cols[i] not in right_rel.cols:
                            final_cols.append(left_rel.cols[i])
                        else:
                            final_cols.append(left_rel.cols[i] + '_x')
                        final_vars.append(left_rel_access.vars[i])

                    for i in range(len(right_rel.cols)):
                        if right_rel.cols[i] not in left_rel.cols:
                            final_cols.append(right_rel.cols[i])
                        else:
                            final_cols.append(right_rel.cols[i] + '_y')
                        final_vars.append(right_rel_access.vars[i])

                    context['t_rule_rel_cols'] = final_cols
                    context['t_rule_rel_vars'] = final_vars

                    # t_rule_rel_index
                    context['t_rule_rel_index'] = 0

                    # t_rule_groupby
                    # t_rule_sort_vars
                    # t_rule_sort_order
                    # t_rule_sort_limit
                    # NO ACTION

                elif how in ['left', 'right', 'outer']:

                    # Reset Index
                    index_var = n_fresh_ir_vars(1)
                    body.append(Theta(ThetaType.ASSIGN, index_var, External(ExternalType.UID, [])))

                    # column args
                    left_cols = []
                    right_cols = []
                    for kw in node.keywords:
                        if kw.arg == 'left_on':
                            for col in kw.value.elts:
                                left_cols.append(col.value)
                        elif kw.arg == 'right_on':
                            for col in kw.value.elts:
                                right_cols.append(col.value)

                    

                    if not (len(left_cols) == len(right_cols) == 1):
                        raise Exception('Outer join with multiple columns not implemented yet')
                    
                    left_col = left_cols[0]
                    right_col = right_cols[0]
                    left_var = get_mapped_var(left_col, left_rel.cols, left_rel_access.vars)
                    right_var = get_mapped_var(right_col, right_rel.cols, right_rel_access.vars)

                    # indicator arg
                    indicator = False
                    for kw in node.keywords:
                        if kw.arg == 'indicator':
                            indicator = kw.value.value

                    type = None
                    if how == 'left':
                        type = OuterType.LEFT
                    elif how == 'right':
                        type = OuterType.RIGHT
                    elif how == 'outer':
                        type = OuterType.FULL

                    atom = Outer(type, left_var, right_var)
                    body.append(atom)

                    if indicator:
                        indicator_var = n_fresh_ir_vars(1)
                        body.append(Theta(
                            ThetaType.ASSIGN, 
                            indicator_var, 
                            If(
                                Theta(ThetaType.EQ, left_var, Null()),
                                String('right_only'),
                                If(
                                    Theta(ThetaType.EQ, right_var, Null()),
                                    String('left_only'),
                                    String('both')
                                )
                            )
                        )) 

                    context['t_rule_body'] = body

                    # t_rule_rel_cols
                    # t_rule_rel_vars

                    final_cols = ['ID']
                    final_vars = [index_var]
                    if indicator:
                        final_cols.append('_merge')
                        final_vars.append(indicator_var)


                    for i in range(len(left_rel.cols)):
                        if left_rel.cols[i] not in right_rel.cols:
                            final_cols.append(left_rel.cols[i])
                        else:
                            final_cols.append(left_rel.cols[i] + '_x')
                        final_vars.append(left_rel_access.vars[i])

                    for i in range(len(right_rel.cols)):
                        if right_rel.cols[i] not in left_rel.cols:
                            final_cols.append(right_rel.cols[i])
                        else:
                            final_cols.append(right_rel.cols[i] + '_y')
                        final_vars.append(right_rel_access.vars[i])

                    context['t_rule_rel_cols'] = final_cols
                    context['t_rule_rel_vars'] = final_vars

                    # t_rule_rel_index
                    context['t_rule_rel_index'] = 0

                    # t_rule_groupby
                    # t_rule_sort_vars
                    # t_rule_sort_order
                    # t_rule_sort_limit
                    # NO ACTION

                else:
                    raise Exception('Merge type not implemented yet')
            elif node.func.attr == 'head':
                
                # find the related rule
                rule_idx = None
                rule = None
                for i in range(len(tondir_global_context['created_rules'])):
                    if tondir_global_context['created_rules'][i].rel_access.name == get_mapped_rel_name(node.func.value.id):
                        rule_idx = i
                        rule = tondir_global_context['created_rules'][i]
                        break

                if rule_idx is None:
                    raise Exception('Rule not found for head')

                # independent limit
                if rule.sort_vars == []:
                    
                    # Reset index
                    body.append(Theta(ThetaType.ASSIGN, body_rel_access.vars[0], External(ExternalType.UID, [])))

                    context['t_rule_body'] = body

                    # t_rule_rel_cols
                    context['t_rule_rel_cols'] = body_rel.cols

                    # t_rule_rel_vars
                    context['t_rule_rel_vars'] = body_rel_access.vars

                    # t_rule_rel_index
                    context['t_rule_rel_index'] = 0

                    # t_rule_sort_limit
                    context['t_rule_sort_limit'] = node.args[0].value

                    # t_rule_groupby
                    # t_rule_sort_vars
                    # t_rule_sort_order
                    # NO ACTION

                else:
                
                    body = rule.body
                    
                    # Reset index
                    body.append(Theta(ThetaType.ASSIGN, rule.rel_access.vars[0], External(ExternalType.UID, [])))

                    context['t_rule_body'] = body

                    rule_rel = tondir_global_context['database'][rule.rel_access.name]

                    # t_rule_rel_cols
                    context['t_rule_rel_cols'] = rule_rel.cols

                    # t_rule_rel_vars
                    context['t_rule_rel_vars'] = rule.rel_access.vars

                    # t_rule_rel_index
                    context['t_rule_rel_index'] = 0
                    
                    # t_rule_sort_vars
                    context['t_rule_sort_vars'] = rule.sort_vars

                    # t_rule_sort_order
                    context['t_rule_sort_order'] = rule.sort_order

                    # t_rule_sort_limit
                    context['t_rule_sort_limit'] = node.args[0].value

                    tondir_global_context['created_rules'].remove(rule)

                    # t_rule_groupby
                    # NO ACTION
            elif node.func.attr == 'contains':

                assignment_var = n_fresh_ir_vars(1)
                regex = node.args[0].value
                regex = regex[1:-1].replace('.*?', '%')

                body.append(Theta(ThetaType.ASSIGN, assignment_var, BinOp(BinOpType.LIKE, body_rel_access.vars[1], String(regex))))

                context['t_rule_body'] = body

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = ['ID', 'Value']

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = [index, assignment_var]

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_groupby
                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.attr == 'isin':

                body, body_rel_inner, body_rel_access_inner, index_inner = prepare_body_relation(node.args[0].id, body)
                body = body[:-1]

                # Exists Condition
                exist = Exist([body_rel_access_inner, Theta(ThetaType.EQ, body_rel_access.vars[1], body_rel_access_inner.vars[1])])
                assignment_var = n_fresh_ir_vars(1)
                body.append(Theta(ThetaType.ASSIGN, assignment_var, exist))

                context['t_rule_body'] = body

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = ['ID', 'Value']

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = [index, assignment_var]

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_groupby
                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.attr == 'slice':

                # Assignment Expression
                assignment_var = n_fresh_ir_vars(1)
                body.append(Theta(ThetaType.ASSIGN, assignment_var, External(ExternalType.SLICE, [body_rel_access.vars[1], Integer(node.args[0].value), Integer(int(node.args[1].value)+1)])))

                context['t_rule_body'] = body

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = ['ID', 'Value']

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = [index, assignment_var]

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_groupby
                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.attr == 'apply':

                if isinstance(node.args[0], ast.Lambda):
                    if len(node.keywords) == 1:
                        if node.keywords[0].arg == 'axis':
                            if node.keywords[0].value.value == 1:
                                if context['p_target_type'] == 'sub':
                                    lambda_expr = node.args[0]
                                    lambda_arg = lambda_expr.args.args[0].arg
                                    lambda_body = lambda_expr.body
                                    expr_context = {'var_mapping': { lambda_arg: body_rel_access.name }, 'rel': body_rel, 'rel_access': body_rel_access}
                                    expr_ir = ast_expression_to_tondir(lambda_body, expr_context)

                                    assign_var = n_fresh_ir_vars(1)
                                    body.append(Theta(ThetaType.ASSIGN, assign_var, expr_ir))

                                    context['t_rule_body'] = body

                                    # t_rule_rel_cols
                                    context['t_rule_rel_cols'] = body_rel.cols + [context['p_target_sub_var_name']]

                                    # t_rule_rel_vars
                                    context['t_rule_rel_vars'] = body_rel_access.vars + [assign_var]

                                    # t_rule_rel_index
                                    context['t_rule_rel_index'] = body_rel.index

                                    # t_rule_groupby
                                    # t_rule_sort_vars
                                    # t_rule_sort_order
                                    # t_rule_sort_limit
                                    # NO ACTION
                                else:
                                    raise Exception('Target Type Not Supported Yet')
                            else:
                                raise Exception('Invalid axis')
                        else:
                            raise Exception('Invalid keyword argument')
                    else:
                        raise Exception('Invalid number of keyword arguments')
                else:
                    raise Exception('Invalid lambda expression')
            elif node.func.attr == 'drop':
                if len(node.keywords) == 1:
                    if node.keywords[0].arg == 'columns':
                        drop_columns = node.keywords[0].value.elts
                        drop_columns_real_indexes = []
                        for col in drop_columns:
                            if isinstance(col, ast.Constant):
                                if col.value != 'ID':
                                    drop_columns_real_indexes.append(body_rel.cols.index(col.value))
                            else:
                                raise Exception('Invalid column argument')
                            
                        new_cols = []
                        new_vars = []
                        for i in range(len(body_rel.cols)):
                            if i not in drop_columns_real_indexes:
                                new_cols.append(body_rel.cols[i])
                                new_vars.append(body_rel_access.vars[i])

                        context['t_rule_body'] = body

                        # t_rule_rel_cols
                        context['t_rule_rel_cols'] = new_cols

                        # t_rule_rel_vars
                        context['t_rule_rel_vars'] = new_vars

                        # t_rule_rel_index
                        context['t_rule_rel_index'] = 0

                        # t_rule_groupby
                        # t_rule_sort_vars
                        # t_rule_sort_order
                        # t_rule_sort_limit
                        # NO ACTION

                    else:
                        raise Exception('Invalid keyword argument')
            elif node.func.attr == 'rename':

                if len(node.keywords) == 1:
                    if node.keywords[0].arg == 'columns':
                        source_cols = [key.value for key in node.keywords[0].value.keys]
                        target_cols = [value.value for value in node.keywords[0].value.values]
                        new_cols = []

                        for col in body_rel.cols:
                            if col in source_cols:
                                new_cols.append(target_cols[source_cols.index(col)])
                            else:
                                new_cols.append(col)

                        context['t_rule_body'] = body

                        # t_rule_rel_cols
                        context['t_rule_rel_cols'] = new_cols

                        # t_rule_rel_vars
                        context['t_rule_rel_vars'] = body_rel_access.vars

                        # t_rule_rel_index
                        context['t_rule_rel_index'] = 0

                        # t_rule_groupby
                        # t_rule_sort_vars
                        # t_rule_sort_order
                        # t_rule_sort_limit
                        # NO ACTION
                    else:
                        raise Exception('Invalid keyword argument')
            elif node.func.attr == 'to_numpy':
                tondir_global_context['database'][get_mapped_rel_name(node.func.value.id)] = body_rel
                tondir_global_context['var_mapping'][context['p_target_var_name']] = body_rel_access.name
                context['skip_rule_generation'] = True
            elif node.func.attr == 'array':
                if len(node.args) == 1:
                    if is_ast_list_of_constants(node.args[0]):
                        line_anf_to_tondir(node.args[0], context)
                    elif isinstance(node.args[0], ast.Name):
                        body, body_rel, body_rel_access, index = prepare_body_relation(node.args[0].id, body)
                        tondir_global_context['var_mapping'][context['p_target_var_name']] = body_rel_access.name
                        context['skip_rule_generation'] = True
                    else:
                        raise Exception('Non-constant elements not supported yet')
            elif node.func.attr == 'einsum':

                expr = node.args[0].value
                expr_tokens = expr.split('->')
                left_tokens = expr_tokens[0].split(',')
                right_tokens = expr_tokens[1].split(',')
                input_layout = 'dense'

                decorator = tondir_global_context['decorator_args'].get('input_layouts', None)
                decorator_dict = {}
                if decorator is not None:
                    for i in range(len(decorator.keys)):
                        decorator_dict[decorator.keys[i].value] = decorator.values[i].value
                if decorator_dict != {}:
                    input_layout = decorator_dict[node.args[1].id]

                if len(left_tokens) == 2 and len(right_tokens) == 1:
                    input_rel_names = [node.args[1].id, node.args[2].id]
                    input_rels = [tondir_global_context['database'][get_mapped_rel_name(name)] for name in input_rel_names]
                    
                    body = []
                    body, body_rel_left, body_rel_access_left, index_left = prepare_body_relation(input_rel_names[0], body)
                    body, body_rel_right, body_rel_access_right, index_right = prepare_body_relation(input_rel_names[1], body)
                    body_rel_left_cols = body_rel_left.cols
                    body_rel_right_cols = body_rel_right.cols

                    # 'ij,kj->i' Kernel (ij,1j->i is currently implemented)
                    if len(left_tokens[0]) == 2 and len(left_tokens[1]) == 1 \
                        and len(right_tokens[0]) == 1 and left_tokens[0][0] == right_tokens[0][0] \
                        and left_tokens[0][1] == left_tokens[1][0]:

                        if input_layout == 'sparse':
                            raise Exception('Support for sparse layout is not implemeted yet')

                        right_rule = None
                        for rule in tondir_global_context['created_rules']:
                            if rule.rel_access.name == get_mapped_rel_name(input_rel_names[1]):
                                right_rule = rule
                                break
                        if right_rule is not []:
                            constant_rel = None
                            for item in right_rule.body:
                                if isinstance(item, ConstantRelation):
                                    constant_rel = item
                                    break
                            if constant_rel is not None:
                                # transpose the second input staticaly since it is constant
                                needed_cols_count = len(constant_rel.values)
                                body_1 = []
                                rel1_col_names = ['ID'] + ['c' + str(i) for i in range(needed_cols_count)]
                                index_var_1 = n_fresh_ir_vars(1)
                                rel1_col_vars = n_fresh_ir_vars(needed_cols_count)
                                crel1 = ConstantRelation(fresh_var('V'), rel1_col_vars, [constant_rel.values])
                                body_1.append(crel1)
                                body_1.append(Theta(ThetaType.ASSIGN, index_var_1, External(ExternalType.UID, [])))
                                rel1 = Relation(fresh_var('V'), rel1_col_names, [Integer] + [Float] * needed_cols_count, 0)
                                tondir_global_context['database'][rel1.name] = rel1
                                rel1_access = RelationAccess(rel1.name, [index_var_1] + rel1_col_vars)
                                r1 = create_rule(rel1_access, [], [], [], [], body_1)
                                rel1_access_new_vars = n_fresh_ir_vars(needed_cols_count+1)
                                body_rel_right = rel1
                                body_rel_access_right = RelationAccess(rel1.name, rel1_access_new_vars)
                            else:
                                raise Exception('Constant relation not found')
                            
                            body_2 = []
                            body_2.append(body_rel_access_left)
                            body_2.append(body_rel_access_right)
                            index_var_2 = n_fresh_ir_vars(1)
                            body_2.append(Theta(ThetaType.ASSIGN, index_var_2, External(ExternalType.UID, [])))

                            rel2_col_names = ['ID', 'c0']
                            rel2_col_types = [Integer, Float]
                            c0_var = n_fresh_ir_vars(1)
                            rel2_col_vars = [index_var_2, c0_var] 

                            expr = None  
                            for j in range(len(body_rel_left_cols)-1, 0, -1):
                                expr = BinOp(BinOpType.ADD,
                                                BinOp(BinOpType.MUL, 
                                                    body_rel_access_left.vars[j], 
                                                    body_rel_access_right.vars[j]
                                                )
                                                , expr if expr is not None else Integer(0)
                                            )
                            expr = Aggregate(AggregateType.SUM, expr)
                            expr = Theta(ThetaType.ASSIGN, c0_var, expr)
                            body_2.append(expr)

                            rel2 = Relation(fresh_var('V'), rel2_col_names, rel2_col_types, 0)
                            tondir_global_context['database'][rel2.name] = rel2
                            rel2_access = RelationAccess(rel2.name, rel2_col_vars)

                            # r2 = create_rule(rel2_access, [body_rel_access_left.vars[0]], [], [], [], body_2)

                            rel_access_final_vars = rel2_access
                            body = body_2
                            context['t_rule_body'] = body

                            # t_rule_rel_cols
                            context['t_rule_rel_cols'] = rel2.cols

                            # t_rule_rel_vars
                            context['t_rule_rel_vars'] = rel2_col_vars

                            # t_rule_rel_index
                            context['t_rule_rel_index'] = 0

                            # t_rule_groupby
                            context['t_rule_groupby'] = [body_rel_access_left.vars[0]]
                            # t_rule_sort_vars
                            context['t_rule_sort_vars'] = [body_rel_access_left.vars[0]]
                            # t_rule_sort_order
                            context['t_rule_sort_order'] = [True]
                            # t_rule_sort_limit
                            # NO ACTION


                    # 'ij,ik->jk' Kernel
                    elif len(left_tokens[0]) == 2 and len(left_tokens[1]) == 2 \
                        and len(left_tokens[0]) == 2 and len(left_tokens[1]) == 2 and len(right_tokens[0]) == 2 \
                        and left_tokens[0][0] == left_tokens[1][0] and left_tokens[0][1] == right_tokens[0][0] \
                        and left_tokens[1][1] == right_tokens[0][1]:

                        if input_layout == 'dense':

                            # Joining two relations and computing the result
                            body.append(Theta(ThetaType.EQ, body_rel_access_left.vars[0], body_rel_access_right.vars[0]))
                            index_var = n_fresh_ir_vars(1)
                            body.append(Theta(ThetaType.ASSIGN, index_var, External(ExternalType.UID, []))) 

                            rel1_col_names = ['ID']
                            re1l_col_types = [Integer]
                            re1l_col_vars = [index_var]
                            for i in range(1, len(body_rel_left_cols)):
                                for j in range(1, len(body_rel_right_cols)):
                                    tmp_var = n_fresh_ir_vars(1)
                                    body.append(Theta(ThetaType.ASSIGN, tmp_var, \
                                                    Aggregate(AggregateType.SUM, \
                                                        BinOp(BinOpType.MUL, \
                                                            body_rel_access_left.vars[i], \
                                                            body_rel_access_right.vars[j] \
                                                        )
                                                    )
                                                )
                                            )
                                    rel1_col_names.append('c' + str(i-1) + "_" + str(j-1))
                                    re1l_col_types.append(Float)
                                    re1l_col_vars.append(tmp_var)
                                                    
                            rel1 = Relation(fresh_var('V'), rel1_col_names, re1l_col_types, 0)
                            tondir_global_context['database'][rel1.name] = rel1
                            rel1_access = RelationAccess(rel1.name, re1l_col_vars)
                            r1 = create_rule(rel1_access, [], [], [], [], body)

                            # Creating constant relation for transposition
                            body_2 = []

                            rel2_col_names = ['value']
                            rel2_constants = []
                            for i in range(len(body_rel_right.cols)-1):
                                rel2_constants.append([i])
                            rel2_var = n_fresh_ir_vars(1)
                            body_2.append(
                                ConstantRelation(fresh_var('V'),
                                                [rel2_var],
                                                rel2_constants))
                            
                            rel2 = Relation(fresh_var('V'), rel2_col_names, [Integer], None)
                            tondir_global_context['database'][rel2.name] = rel2

                            rel_access_2 = RelationAccess(rel2.name, [rel2_var])

                            r2 = create_rule(rel_access_2, [], [], [], [], body_2)

                            # Transposing the results
                            body_3 = []

                            rel1_vars = n_fresh_ir_vars(len(rel1.cols))
                            if not isinstance(rel1_vars, list):
                                rel1_vars = [rel1_vars]
                            rel2_vars = n_fresh_ir_vars(len(rel2.cols))
                            if not isinstance(rel2_vars, list):
                                rel2_vars = [rel2_vars]

                            body_3.append(RelationAccess(rel1.name, rel1_vars))
                            body_3.append(RelationAccess(rel2.name, rel2_vars))
                            index_var = n_fresh_ir_vars(1)
                            # body_3.append(Theta(ThetaType.ASSIGN, index_var, External(ExternalType.UID, [])))

                            rel3_col_names = ['ID']
                            rel3_col_types = [Integer]
                            rel3_col_vars = [rel2_vars[0]]

                            left_real_cols_count = len(body_rel_left_cols) - 1
                            right_real_cols_count = len(body_rel_right_cols) - 1

                            for k in range(right_real_cols_count):
                                new_col_var = n_fresh_ir_vars(1)
                                expr = None
                                for j in range(left_real_cols_count-1, -1, -1):
                                    expr = If(Theta(ThetaType.EQ, 
                                            rel2_vars[0], Integer(j)), 
                                            rel1_vars[1 + right_real_cols_count * k + j],
                                            Integer(0) if expr is None else expr)
                                body_3.append(Theta(ThetaType.ASSIGN, new_col_var, expr))
                                rel3_col_names.append('c' + str(k))
                                rel3_col_types.append(Float)
                                rel3_col_vars.append(new_col_var)

                            rel3 = Relation(fresh_var('V'), rel3_col_names, rel3_col_types, 0)
                            tondir_global_context['database'][rel3.name] = rel3
                            rel_access_3 = RelationAccess(rel3.name, rel3_col_vars)

                            r3 = create_rule(rel_access_3, [], [], [], [], body_3)

                            rel_access_final_vars = n_fresh_ir_vars(len(rel3.cols))
                            rel_access_final = RelationAccess(rel3.name, rel_access_final_vars)
                            body = [rel_access_final]
                            context['t_rule_body'] = body

                            # t_rule_rel_cols
                            context['t_rule_rel_cols'] = rel3.cols

                            # t_rule_rel_vars
                            context['t_rule_rel_vars'] = rel_access_final.vars

                            # t_rule_rel_index
                            context['t_rule_rel_index'] = 0
                            # t_rule_groupby
                            # t_rule_sort_vars
                            context['t_rule_sort_vars'] = [rel_access_final_vars[0]]
                            # t_rule_sort_order
                            context['t_rule_sort_order'] = [True]
                            # t_rule_sort_limit
                            # NO ACTION

                        elif input_layout == 'sparse':

                            # Index
                            index_var = n_fresh_ir_vars(1)
                            body.append(Theta(ThetaType.ASSIGN, index_var, External(ExternalType.UID, [])))

                            # Join Condition
                            body.append(Theta(ThetaType.EQ, body_rel_access_left.vars[1], body_rel_access_right.vars[1]))

                            # Assignment Expression
                            assignment_var = n_fresh_ir_vars(1)
                            body.append(Theta(ThetaType.ASSIGN, assignment_var,
                                                Aggregate(AggregateType.SUM,
                                                    BinOp(BinOpType.MUL, body_rel_access_left.vars[3], body_rel_access_right.vars[3])
                                                )
                                            )
                                        )
                            
                            context['t_rule_body'] = body

                            # t_rule_rel_cols
                            context['t_rule_rel_cols'] = ['ID', 'rid', 'cid', 'value']

                            # t_rule_rel_vars
                            context['t_rule_rel_vars'] = [index_var, body_rel_access_left.vars[2], body_rel_access_right.vars[2], assignment_var]

                            # t_rule_rel_index
                            context['t_rule_rel_index'] = 0

                            # t_rule_groupby
                            context['t_rule_groupby'] = [body_rel_access_left.vars[2], body_rel_access_right.vars[2]]

                            # t_rule_sort_vars
                            context['t_rule_sort_vars'] = [body_rel_access_left.vars[2], body_rel_access_right.vars[2]]

                            # t_rule_sort_order
                            context['t_rule_sort_order'] = [True, True]

                            # t_rule_sort_limit
                            # NO ACTION

                    else:
                        raise Exception('Not implemented einsum expression.')
            elif node.func.attr == 'replace':

                mapping = {}
                for i in range(len(node.args[0].keys)):
                    mapping[String(node.args[0].keys[i].value)] = String(node.args[0].values[i].value)

                expr = None
                for key in mapping:
                    if expr is None:
                        expr = If(Theta(ThetaType.EQ, body_rel_access.vars[1], (key)), (mapping[key]), body_rel_access.vars[1])
                    else:
                        expr = If(Theta(ThetaType.EQ, body_rel_access.vars[1], (key)), (mapping[key]), expr)

                assignment_var = n_fresh_ir_vars(1)

                body.append(Theta(ThetaType.ASSIGN, assignment_var, expr))

                context['t_rule_body'] = body

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = ['ID', 'Value']

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = [index, assignment_var]

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_groupby
                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.attr == 'unique':

                # Reset index
                body.append(Theta(ThetaType.ASSIGN, body_rel_access.vars[0], External(ExternalType.UID, [])))

                context['t_rule_body'] = body

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = body_rel.cols

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = body_rel_access.vars

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_groupby
                context['t_rule_groupby'] = [body_rel_access.vars[1]]
                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.attr == 'pivot_table':

                # column args
                index = None
                columns = None
                values = None
                for kw in node.keywords:
                    if not isinstance(kw.value, (ast.Constant, ast.Name)):
                        raise Exception('Invalid pivot_table arguments') 
                    if kw.arg == 'index':
                        index = kw.value.value
                    elif kw.arg == 'columns':
                        columns = kw.value.value
                    elif kw.arg == 'values':
                        values = kw.value.value
                    elif kw.arg == 'aggfunc':
                        aggfunc = kw.value.id
                        agg_op = None
                        if aggfunc == 'sum':
                            agg_op = AggregateType.SUM
                        else:
                            raise Exception('Aggregation function not implemented.')


                if index is None or columns is None or values is None:
                    raise Exception('Invalid pivot_table arguments')

                # Compute Pivot Table

                # New columns from decorator args
                new_cols = []
                new_vars = []

                decorator = tondir_global_context['decorator_args']['pivot_column_values']
                # conver decorator to a dictionary
                decorator_dict = {}
                for i in range(len(decorator.keys)):
                    decorator_dict[decorator.keys[i].value] = []
                    for j in range(len(decorator.values[i].elts)):
                        decorator_dict[decorator.keys[i].value].append(decorator.values[i].elts[j].value)

                # Create new columns
                new_cols = decorator_dict[columns]

                # Create new vars
                cond_var = body_rel_access.vars[body_rel.cols.index(columns)]
                val_var = body_rel_access.vars[body_rel.cols.index(values)]
                for i in range(len(new_cols)):
                    new_vars.append(n_fresh_ir_vars(1))
                    expr = Theta(ThetaType.ASSIGN,
                                 new_vars[i],
                                 Aggregate(agg_op,
                                            If(Theta(ThetaType.EQ, cond_var, String(new_cols[i])),
                                                  val_var,
                                                  Integer(0)
                                              )
                                        )
                                )
                    body.append(expr)

                # Reset index
                body.append(Theta(ThetaType.ASSIGN, body_rel_access.vars[0], External(ExternalType.UID, [])))

                context['t_rule_body'] = body

                # Grouping
                group_col_idx = body_rel.cols.index(index)
                group_col_var = body_rel_access.vars[group_col_idx]
                context['t_rule_groupby'] = [group_col_var]
                context['t_rule_rel_cols'] = ['ID', index] + new_cols
                context['t_rule_rel_vars'] = [body_rel_access.vars[0], group_col_var] + new_vars
                context['t_rule_rel_index'] = 0

                context['t_rule_rel_py_index'] = [1]

                # t_rule_sort_vars
                context['t_rule_sort_vars'] = [group_col_var]
                # t_rule_sort_order
                context['t_rule_sort_order'] = [True]
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.attr == 'div':

                body_inner = []
                body_inner, body_rel_inner, body_rel_access_inner, index_inner = prepare_body_relation(node.args[0].id, body_inner)

                body.append(body_rel_access_inner)
                body.append(Theta(ThetaType.EQ, body_rel_access.vars[0], body_rel_access_inner.vars[0]))

                new_vars = []
                for i in range(1, len(body_rel.cols)):
                    if i not in body_rel.py_index:
                        new_var = n_fresh_ir_vars(1)
                        new_vars.append(new_var)
                        body.append(Theta(ThetaType.ASSIGN,
                                        new_var, 
                                        BinOp(BinOpType.DIV,
                                                    BinOp(BinOpType.ADD, body_rel_access.vars[i], Float(0.0)),
                                                    body_rel_access_inner.vars[1]
                                                )
                                        )
                                    )

                context['t_rule_body'] = body

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = body_rel.cols

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = [body_rel_access.vars[0]] + [body_rel_access.vars[i] for i in body_rel.py_index] + new_vars

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_rel_py_index
                context['t_rule_rel_py_index'] = body_rel.py_index

                # t_rule_groupby
                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.attr == 'dropna':

                cols = []
                col_vars = []
                for kw in node.keywords:
                    if kw.arg == 'subset':
                        for col in kw.value.elts:
                            cols.append(col.value)
                            col_vars.append(body_rel_access.vars[body_rel.cols.index(col.value)])

                for i in range(len(cols)):
                    if body_rel.cols[i] not in cols:
                        body.append(Theta(ThetaType.NE, col_vars[i], Null()))

                context['t_rule_body'] = body

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = body_rel.cols

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = body_rel_access.vars

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_groupby
                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION

            else:
                raise Exception('Function call not implemented yet')

        # func()
        elif isinstance(node.func, ast.Name):
            
            body = []
            body, body_rel, body_rel_access, index = prepare_body_relation(node.args[0].id, body)

            if  node.func.id == 'len':
                                # Reset index
                body.append(Theta(ThetaType.ASSIGN, body_rel_access.vars[0], External(ExternalType.UID, [])))

                # Count
                count_var = n_fresh_ir_vars(1)
                body.append(Theta(ThetaType.ASSIGN, count_var, Aggregate(AggregateType.COUNT, body_rel_access.vars[0])))

                context['t_rule_body'] = body

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = ['ID', 'Value']

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = [index, count_var]

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_groupby
                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION
            elif node.func.id == 'int':

                new_var = n_fresh_ir_vars(1)

                # Convert to Integer
                expr = Theta(ThetaType.ASSIGN, 
                                new_var, 
                                External(ExternalType.INT, [body_rel_access.vars[1]])
                            )
                body.append(expr)

                context['t_rule_body'] = body

                # t_rule_rel_cols
                context['t_rule_rel_cols'] = ['ID', 'Value']

                # t_rule_rel_vars
                context['t_rule_rel_vars'] = [index, new_var]

                # t_rule_rel_index
                context['t_rule_rel_index'] = 0

                # t_rule_groupby
                # t_rule_sort_vars
                # t_rule_sort_order
                # t_rule_sort_limit
                # NO ACTION

            else:
                raise Exception('Function call not implemented yet')

    elif isinstance(node, ast.UnaryOp):
        # op x
        if isinstance(node.operand, ast.Name):
            ## t_rule_body
            body = []
            body, body_rel, body_rel_access, index = prepare_body_relation(node.operand.id, body)

            # ~x
            if node.op.__class__.__name__ == 'Invert':
                value_var = n_fresh_ir_vars(1)
                body.append(Theta(ThetaType.ASSIGN, value_var, External(ExternalType.NOT, [body_rel_access.vars[1]])))
            
            context['t_rule_body'] = body

            ## t_rule_rel_cols
            context['t_rule_rel_cols'] = ['ID', 'Value']

            ## t_rule_rel_vars
            context['t_rule_rel_vars'] = [index, value_var]
            
            ## t_rule_rel_index
            context['t_rule_rel_index'] = 0

            ## t_rule_groupby
            ## t_rule_sort_vars
            ## t_rule_sort_order
            ## t_rule_sort_limit
            # NO ACTION

    elif isinstance(node, ast.BinOp):
        # x op x
        if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):

            ## t_rule_body
            body = []
            
            body, body_rel_left, body_rel_access_left, index_left = prepare_body_relation(node.left.id, body)
            body, body_rel_right, body_rel_access_right, index_right = prepare_body_relation(node.right.id, body)

            ## Join Condition
            body.append(Theta(ThetaType.EQ, index_left, index_right))

            value_var = n_fresh_ir_vars(1)
            ## BinOp BODY HERE
            op = None
            if node.op.__class__.__name__ in ['BitAnd']:
                op = BinOpType.AND
            elif node.op.__class__.__name__ in ['BitOr']:
                op = BinOpType.OR
            elif node.op.__class__.__name__ in ['Mult']:
                op = BinOpType.MUL
            elif node.op.__class__.__name__ in ['Sub']:
                op = BinOpType.SUB
            elif node.op.__class__.__name__ in ['Div']:
                op = BinOpType.DIV
            else:
                raise Exception('Invalid BinOp')

            body.append(Theta(ThetaType.ASSIGN, value_var, BinOp(op, body_rel_access_left.vars[1], body_rel_access_right.vars[1])))

            context['t_rule_body'] = body

            ## t_rule_rel_cols
            context['t_rule_rel_cols'] = ['ID', 'Value']

            ## t_rule_rel_vars
            context['t_rule_rel_vars'] = [index_left, value_var]

            ## t_rule_rel_index
            context['t_rule_rel_index'] = 0

            ## t_rule_groupby
            ## t_rule_sort_vars
            ## t_rule_sort_order
            ## t_rule_sort_limit
            # NO ACTION

        # x op non-x or non-x op x
        elif isinstance(node.left, ast.Name) or isinstance(node.right, ast.Name):

            ## t_rule_body
            body = []
            var = node.left if isinstance(node.left, ast.Name) else node.right
            non_var = node.right if isinstance(node.left, ast.Name) else node.left
            is_var_left = isinstance(node.left, ast.Name)

            body_rel = tondir_global_context['database'][get_mapped_rel_name(var.id)]
            body_rel_access = RelationAccess(body_rel.name, n_fresh_ir_vars(len(body_rel.cols)))
            body.append(body_rel_access)
            body, index = get_or_create_index_var(body_rel, body_rel_access, body)

            ## BinOp BODY HERE
            if node.op.__class__.__name__ in ['Add', 'Sub', 'Mult', 'Div', 'Mod', 'Pow']:
                if isinstance(non_var, ast.Constant):
                    binop_type = None
                    if node.op.__class__.__name__ == 'Add':
                        binop_type = BinOpType.ADD
                    elif node.op.__class__.__name__ == 'Sub':
                        binop_type = BinOpType.SUB
                    elif node.op.__class__.__name__ == 'Mult':
                        binop_type = BinOpType.MUL
                    elif node.op.__class__.__name__ == 'Div':
                        binop_type = BinOpType.DIV

                    const = get_ir_typed_constant(non_var.value)
                    value_var = n_fresh_ir_vars(1)
                    if is_var_left:
                        body.append(Theta(ThetaType.ASSIGN, value_var, BinOp(binop_type, body_rel_access.vars[1], const)))
                    else:
                        body.append(Theta(ThetaType.ASSIGN, value_var, BinOp(binop_type, const, body_rel_access.vars[1])))

            context['t_rule_body'] = body

            ## t_rule_rel_cols
            context['t_rule_rel_cols'] = ['ID', 'Value']

            ## t_rule_rel_vars
            context['t_rule_rel_vars'] = [index, value_var]

            ## t_rule_rel_index
            context['t_rule_rel_index'] = 0

            ## t_rule_groupby
            ## t_rule_sort_vars
            ## t_rule_sort_order
            ## t_rule_sort_limit
            # NO ACTION


            
        # constant op
        else:
            context['skip_rule_generation'] = True

    elif isinstance(node, ast.Subscript):
        # x1[x2]
        if isinstance(node.slice, ast.Name):
            
            # t_rule_body
            body = []
            body, body_rel_outer, body_rel_access_outer, index_outer = prepare_body_relation(node.value.id, body)
            body, body_rel_inner, body_rel_access_inner, index_inner = prepare_body_relation(node.slice.id, body)

            # Reset Index
            body.append(Theta(ThetaType.ASSIGN, body_rel_access_outer.vars[0], External(ExternalType.UID, [])))

            # Join Condition
            body.append(Theta(ThetaType.EQ, index_outer, index_inner))

            # Filter Condition
            body.append(Theta(ThetaType.EQ, body_rel_access_inner.vars[1], Boolean(True)))

            context['t_rule_body'] = body

            # t_rule_rel_cols
            context['t_rule_rel_cols'] = body_rel_outer.cols

            # t_rule_rel_vars
            context['t_rule_rel_vars'] = body_rel_access_outer.vars

            # t_rule_rel_index
            context['t_rule_rel_index'] = 0

            # t_rule_groupby
            # t_rule_sort_vars
            context['t_rule_sort_vars'] = [index_outer]
            # t_rule_sort_order
            context['t_rule_sort_order'] = [True]
            # t_rule_sort_limit
            # NO ACTION

        # x1[[s1, s2, ...]] or x1[s]
        elif isinstance(node.slice, ast.List) or isinstance(node.slice, ast.Constant):

            # t_rule_body
            body = []
            body, body_rel, body_rel_access, index = prepare_body_relation(node.value.id, body)
            context['t_rule_body'] = body

            # t_rule_rel_cols

            projected_cols = []
            if isinstance(node.slice, ast.List):
                for el in node.slice.elts:
                    projected_cols.append(el.value)
            else:
                projected_cols.append(node.slice.value)
            context['t_rule_rel_cols'] = ['ID'] + projected_cols

            # Projecting
            projected_vars = []
            for col in projected_cols:
                projected_vars.append(get_mapped_var(col, body_rel.cols, body_rel_access.vars))
            context['t_rule_rel_vars'] = [index] + projected_vars

            # t_rule_rel_index
            context['t_rule_rel_index'] = 0

            # t_rule_groupby
            # t_rule_sort_vars
            # t_rule_sort_order
            # t_rule_sort_limit
            # NO ACTION

        # x1.iloc[x2:x3]
        elif isinstance(node.slice, ast.Slice) and isinstance(node.value, ast.Attribute) and node.value.attr == 'iloc':
            
            # t_rule_body
            body = []
            body, body_rel, body_rel_access, index = prepare_body_relation(node.value.value.id, body)

            # Reset Index
            new_var = n_fresh_ir_vars(1)
            body.append(Theta(ThetaType.ASSIGN, new_var, External(ExternalType.UID, [])))

            body_lower, lower_rel, lower_rel_access, lower_index = prepare_body_relation(node.slice.lower.id, body)
            body_upper, upper_rel, upper_rel_access, upper_index = prepare_body_relation(node.slice.upper.id, body)

            # Filter
            body.append(Theta(ThetaType.GE, body_rel_access.vars[0], lower_rel_access.vars[1]))
            body.append(Theta(ThetaType.LT, body_rel_access.vars[0], upper_rel_access.vars[1]))

            context['t_rule_body'] = body

            # t_rule_rel_cols
            context['t_rule_rel_cols'] = body_rel.cols

            # t_rule_rel_vars
            context['t_rule_rel_vars'] = [new_var] + body_rel_access.vars[1:]

            # t_rule_rel_index
            context['t_rule_rel_index'] = 0

            # t_rule_groupby
            # t_rule_sort_vars
            context['t_rule_sort_vars'] = [index]
            # t_rule_sort_order
            context['t_rule_sort_order'] = [True]
            # t_rule_sort_limit
            # NO ACTION

        else:
            raise Exception('Invalid Subscript')

    elif isinstance(node, ast.List):
        
        if len(node.elts) == 1 and isinstance(node.elts[0], ast.Name):
            context['skip_rule_generation'] = True
            tondir_global_context['var_mapping'][context['p_target_var_name']] = get_mapped_rel_name(node.elts[0].id)
        
        elif is_ast_list_of_constants(node): 

            # t_rule_body
            body = []
            
            rows_count = len(node.elts)
            cols_count = 1

            if not isinstance(node.elts[0], ast.Constant):
                raise Exception('Two Dimensional List not supported')
            
            # Create Relation
            index_var = n_fresh_ir_vars(1)
            cr_vars = n_fresh_ir_vars(cols_count)
            values = []
            for i in range(rows_count):
                if isinstance(node.elts[i], ast.Constant):
                    values.append(get_ir_typed_constant(node.elts[i].value))
                else:
                    values.append(get_ir_typed_constant(-1*node.elts[i].operand.value))
            crel = ConstantRelation(fresh_var('V'), [cr_vars], values)
            body.append(crel)

            # Index
            body.append(Theta(ThetaType.ASSIGN, index_var, External(ExternalType.UID, [])))

            context['t_rule_body'] = body

            # t_rule_cols
            context['t_rule_rel_cols'] = ['ID']
            for i in range(cols_count):
                context['t_rule_rel_cols'].append('C' + str(i))

            # t_rule_rel_vars
            context['t_rule_rel_vars'] = [index_var, cr_vars]

            # t_rule_rel_index
            context['t_rule_rel_index'] = 0

            # t_rule_groupby
            # t_rule_sort_vars
            # t_rule_sort_order
            # t_rule_sort_limit
            # NO ACTION

        else:
            raise Exception('Invalid List')

    elif isinstance(node, ast.Constant):

        if context['p_target_type'] != 'subsub':
            raise Exception('Constant assignment not supported')
        
        # handling x[y][z] = constant
        body = []
        body, body_rel_main, body_rel_access_main, index_main = prepare_body_relation(context['p_target_var_name'], body)
        body, body_rel_filter, body_rel_access_filter, index_filter = prepare_body_relation(context['p_target_sub_filter_var_name'], body)

        # Join Condition
        body.append(Theta(ThetaType.EQ, index_main, index_filter))

        # Does Condtional Column Exist Before?
        col_exists = context['p_target_sub_var_name'] in body_rel_main.cols
        col_idx = body_rel_main.cols.index(context['p_target_sub_var_name'])
        value_var = n_fresh_ir_vars(1)
        expr = Theta(ThetaType.ASSIGN, 
                        value_var,
                        If(Theta(ThetaType.EQ, body_rel_access_filter.vars[1], Boolean(True)),
                            get_ir_typed_constant(node.value),
                            body_rel_access_main.vars[col_idx] if col_exists else Null() 
                        )
                    )
        body.append(expr)

        # UID
        index_var = n_fresh_ir_vars(1)
        body.append(Theta(ThetaType.ASSIGN, index_var, External(ExternalType.UID, [])))
                    
        context['t_rule_body'] = body

        # t_rule_rel_cols
        context['t_rule_rel_cols'] = body_rel_main.cols
        if col_exists:
            context['t_rule_rel_cols'].remove(body_rel_main.cols[col_idx])
        context['t_rule_rel_cols'].append(context['p_target_sub_var_name'])

        # t_rule_rel_vars
        context['t_rule_rel_vars'] = [index_var] + body_rel_access_main.vars[1:]
        if col_exists:
            context['t_rule_rel_vars'].remove(body_rel_access_main.vars[col_idx])
        
        context['t_rule_rel_vars'].append(value_var)

        # t_rule_rel_index
        context['t_rule_rel_index'] = 0

        # t_rule_groupby
        # t_rule_sort_vars
        # t_rule_sort_order
        # t_rule_sort_limit
        # NO ACTION

    elif isinstance(node, ast.ListComp):

        # t_rule_body
        body = []

        # Creating Relation
        generator = node.generators[0]
        rel_name = generator.iter.id
        rel_var_name = generator.target.id
        body, body_rel, body_rel_access, index = prepare_body_relation(tondir_global_context['var_mapping'][rel_name], body)
        tondir_global_context['var_mapping'][rel_var_name] = body_rel_access.vars[1].name

        # Applying the comprehension

        res_var = n_fresh_ir_vars(1)
        comp_expr = ast_expression_to_tondir(node.elt, context)
        body.append(Theta(ThetaType.ASSIGN, res_var, comp_expr))

        # # Resetting the index
        # body.append(Theta(ThetaType.ASSIGN, body_rel_access.vars[0], External(ExternalType.UID, [])))

        context['t_rule_body'] = body

        # t_rule_rel_cols
        context['t_rule_rel_cols'] = body_rel.cols

        # t_rule_rel_vars
        context['t_rule_rel_vars'] = [body_rel_access.vars[0], res_var]

        # t_rule_rel_index
        context['t_rule_rel_index'] = 0

        # t_rule_groupby
        # t_rule_sort_vars
        # t_rule_sort_order
        # t_rule_sort_limit
        # NO ACTION

        del tondir_global_context['var_mapping'][rel_var_name]
    
    else:
        raise Exception('Invalid expression')

def function_anf_to_tondir(name, anf):
    global tondir_global_context

    program = Program(name, [])

    for item in anf:

        # Defining local context
        local_context = {
            'p_assignment_left': None,
            'p_assignment_right': None,
            'p_target_type': None, # 'var' or 'sub' or 'subsub'
            'p_target_var_name': None,
            'p_target_sub_var_name': None,
            'p_target_sub_filter_var_name': None,
            't_rule_rel_cols': [],
            't_rule_rel_vars': [],
            't_rule_groupby': [],
            't_rule_sort_vars': [],
            't_rule_sort_order': [],
            't_rule_sort_limit': None,
            't_rule_body': [],
            't_rule_rel_index': None,
            't_rule_rel_py_index': None,
            'skip_rule_generation': False,
        }

        # When the anf expression is an assignment (usual case)
        if isinstance(item, ast.Assign):
            local_context['p_assignment_left'] = item.targets[0]
            left = item.targets[0]
            local_context['p_assignment_right'] = item.value
            right = item.value

            if debug_mode:
                print('\033[94m' + 'LEFT:\t' + '\033[0m', end=' ')
                print_ast(left)
                print('\033[94m' + 'RIGHT:\t' + '\033[0m', end=' ')
                print_ast(right)

            # Assessing left side of assignment
            if isinstance(left, ast.Name):
                local_context['p_target_type'] = 'var'
                local_context['p_target_var_name'] = left.id
            elif isinstance(left, ast.Subscript):
                if isinstance(left.value, ast.Name):
                    local_context['p_target_type'] = 'sub'
                    local_context['p_target_var_name'] = left.value.id
                    local_context['p_target_sub_var_name'] = left.slice.value
                elif isinstance(left.value, ast.Subscript):
                    local_context['p_target_type'] = 'subsub'
                    local_context['p_target_var_name'] = left.value.value.id
                    local_context['p_target_sub_var_name'] = left.value.slice.value
                    local_context['p_target_sub_filter_var_name'] = left.slice.id
            else:
                print_ast(left)
                raise Exception('Non-defined assignment LEFT!!!')
            
            # Assessing right side of assignment
            line_anf_to_tondir(right, local_context)

            # Skipping Rule Generation
            if debug_mode:
                print('\033[91m' + 'Rule Generation Skipped' + '\033[0m', end='\n\n') if local_context['skip_rule_generation'] else None
            
            if local_context['skip_rule_generation']:
                continue

            # Creating TondIR Relation and Updating Database
            rel = Relation(fresh_var('V'),
                            local_context['t_rule_rel_cols'],
                            None,
                            local_context['t_rule_rel_index'],
                            py_index=local_context['t_rule_rel_py_index'])

            tondir_global_context['database'][rel.name] = rel

            # Creating TondIR rule
            rule = create_rule(RelationAccess(rel.name, local_context['t_rule_rel_vars']),
                                local_context['t_rule_groupby'],
                                local_context['t_rule_sort_vars'],
                                local_context['t_rule_sort_order'],
                                local_context['t_rule_sort_limit'],
                                local_context['t_rule_body'])
            program.rules = tondir_global_context['created_rules']


            # Updating relation var mapping            
            tondir_global_context['var_mapping'][local_context['p_target_var_name']] = rel.name

            # print(tondir_global_context['var_mapping'])
            # print(tondir_global_context['database'])
            
            if debug_mode:
                print('\033[92m' + 'Rule:\t', rule, '\033[0m', end='\n\n')

        # When the anf expression is a return statement (final case)
        elif isinstance(item, ast.Return):
            if debug_mode:
                print('\033[94m' + 'RETURN:\t' + '\033[0m', program.final_rel, end='\n\n')
                          
        else:
            print_ast(item)
            raise Exception('Invalid ANF expression')

    return program

def ast_to_anf(expr):
    global anf_result_cache
    # print("Converting -> ANF: \n", ast.dump(expr), end='\n\n')

    expr_copy = copy.deepcopy(expr)
    
    if isinstance(expr, ast.Assign):
        if len(expr.targets) > 1:
            raise Exception('Multiple targets in assignment not supported')
        target = expr.targets[0]
        if isinstance(target, ast.Name) or \
            (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)) \
            or (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Subscript)):
            # v = ...
            if isinstance(target, ast.Name):
                expr_copy.value = ast_to_anf(expr_copy.value)
            elif isinstance(target, ast.Subscript):
                # v[x] = ...
                if isinstance(target.slice, ast.Constant):
                    if not (isinstance(expr.value, ast.Call) and expr.value.func.attr == 'apply'): # Ignoring apply lambda case
                        expr_copy.value = create_intermediate_assignment(ast_to_anf(expr_copy.value))
                # v[x][cond] = ...
                elif isinstance(target.value, ast.Subscript):
                    filter = create_intermediate_assignment(ast_to_anf(target.slice))
                    expr_copy.targets[0].slice = filter
                    expr_copy.value = ast_to_anf(expr_copy.value)
            else:
                raise Exception('Invalid target in assignment 2')
        else:
            raise Exception('Invalid target in assignment 1')
    
    elif isinstance(expr, ast.Subscript):

        is_iloc_case = isinstance(expr.value, ast.Attribute) and expr.value.attr == 'iloc'

        if not check_anf_compliant_types(expr.value):
            if not is_iloc_case:
                expr_copy.value = create_intermediate_assignment(ast_to_anf(expr_copy.value))

        if not check_anf_compliant_types(expr.slice):
            if not is_ast_list_of_constants(expr.slice):
                if not is_iloc_case:
                    expr_copy.slice = create_intermediate_assignment(ast_to_anf(expr_copy.slice))
                else:
                    expr_copy.slice = ast_to_anf(expr_copy.slice)

    elif isinstance(expr, ast.UnaryOp):
        if not check_anf_compliant_types(expr.operand):
            expr_copy.operand = create_intermediate_assignment(ast_to_anf(expr_copy.operand))
    
    elif isinstance(expr, ast.BinOp):
        if not check_anf_compliant_types(expr.left):
            expr_copy.left = create_intermediate_assignment(ast_to_anf(expr_copy.left))
        if not check_anf_compliant_types(expr.right):
            expr_copy.right = create_intermediate_assignment(ast_to_anf(expr_copy.right))
    
    elif isinstance(expr, ast.Compare):
        if not check_anf_compliant_types(expr.left):
            expr_copy.left = create_intermediate_assignment(ast_to_anf(expr_copy.left))
        if not check_anf_compliant_types(expr.comparators[0]):
            expr_copy.comparators[0] = create_intermediate_assignment(ast_to_anf(expr_copy.comparators[0]))
    
    elif isinstance(expr, ast.Attribute):
        if not check_anf_compliant_types(expr.value):
            expr_copy.value = create_intermediate_assignment(ast_to_anf(expr_copy.value))
    
    elif isinstance(expr, ast.Call):
        for i, arg in enumerate(expr.args):
            if (not check_anf_compliant_types(arg)) and (not isinstance(arg, ast.Lambda)):
                if not(is_ast_list_of_constants(arg) and not (isinstance(expr.func, ast.Attribute) and expr.func.attr == 'isin')):
                    if not is_ast_dict_of_constants(arg):
                        expr_copy.args[i] = create_intermediate_assignment(ast_to_anf(arg))
        if isinstance(expr.func, ast.Attribute):
            if not check_anf_compliant_types(expr.func.value):
                expr_copy.func.value = create_intermediate_assignment(ast_to_anf(expr_copy.func.value))

    elif isinstance(expr, ast.List):
        for i, el in enumerate(expr.elts):
            if not check_anf_compliant_types(el):
                expr_copy.elts[i] = create_intermediate_assignment(ast_to_anf(el))
    
    elif isinstance(expr, ast.Return):
        if not check_anf_compliant_types(expr.value):
            expr_copy.value = create_intermediate_assignment(ast_to_anf(expr_copy.value))
    
    elif isinstance(expr, ast.Slice):
        if expr.lower is not None:
            if not check_anf_compliant_types(expr.lower):
                expr_copy.lower = create_intermediate_assignment(ast_to_anf(expr_copy.lower))
        if expr.upper is not None:
            if not check_anf_compliant_types(expr.upper):
                expr_copy.upper = create_intermediate_assignment(ast_to_anf(expr_copy.upper))
        if expr.step is not None:
            if not check_anf_compliant_types(expr.step):
                expr_copy.step = create_intermediate_assignment(ast_to_anf(expr_copy.step))
 
    elif isinstance(expr, ast.Constant):
        pass

    elif isinstance(expr, ast.Dict):
        pass

    elif isinstance(expr, ast.Lambda):
        anf_result_cache.append(expr)

    elif isinstance(expr, ast.ListComp):
        if len(expr.generators) > 1:
            raise Exception('Multiple generators in ListComp not supported')
    elif isinstance(expr, ast.Name):
        pass
    else:
        print_ast(expr)
        raise Exception('Invalid expression')    

    return expr_copy

def get_decorated_functions(file_path):
    with open(file_path, 'r') as file:
        source = file.read()
        tree = ast.parse(source)
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Name) and decorator.id == 'pytond') or \
                    (isinstance(decorator, ast.Call) and decorator.func.id == 'pytond'):
                        functions[node.name] = {}
                        if isinstance(decorator, ast.Call):
                            tmp = decorator.keywords
                            functions[node.name]['decorator_args'] = {x.arg: x.value for x in tmp}
                        else:
                            functions[node.name]['decorator_args'] = {}
                        functions[node.name]['args'] = node.args
                        functions[node.name]['ast'] = node.body
                        functions[node.name]['source'] = ast.unparse(node.body)
                        functions[node.name]['tondir'] = None
                        functions[node.name]['sql'] = None
                        functions[node.name]['sql_optimized'] = None
                        break
        print('>>> Python -> AST |', len(functions), 'functions [', ', '.join(functions.keys()), ']')
        return functions

def python_to_ast(source):
    global anf_result_cache
    funcs = get_decorated_functions(source)
    for func in funcs:
        fresh_var_reset()
        funcs[func]['ast_anf'] = []
        for node in funcs[func]['ast']:
            anf_res = ast_to_anf(node)
            funcs[func]['ast_anf'].extend(anf_result_cache)
            funcs[func]['ast_anf'].append(anf_res)
            anf_result_cache = []
        for i in range(len(funcs[func]['ast_anf'])):
            funcs[func]['ast_anf'][i] = ast.fix_missing_locations(funcs[func]['ast_anf'][i])
        funcs[func]['source_anf'] = ast.unparse(funcs[func]['ast_anf'])
        print(">>> AST -> ANF\t|", func)
    return funcs

def python_to_sql(bench_dir, source, used_database, pytond_context, verbose=False):
    file_name = source[0:-3]
    path = os.path.abspath(os.path.dirname(bench_dir)) + '/benchmark/outputs/' + file_name
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                os.remove(os.path.join(path, file))

    funcs = python_to_ast(bench_dir + '/workloads/' + source)
    for f in funcs:
        fresh_var_reset()
        ########################
        store_in_file(f'{path}/{f}.py', funcs[f]['source'])
        if verbose:
            print('\n\nAST (Original):\n')
            print(funcs[f]['source'])
            print('\n\n')
        ########################
        store_in_file(f'{path}/{f}_anf.py', funcs[f]['source_anf'])
        if verbose:
            print('AST (ANF):\n')
            print(funcs[f]['source_anf'])
            print('\n\n')
        ########################
        reset_tondir_global_context()
        tondir_global_context['decorator_args'] = funcs[f]['decorator_args']
        tondir_global_context['database'] = pytond_context['database'][used_database]
        funcs[f]['tondir'] = function_anf_to_tondir(f, funcs[f]['ast_anf'])
        store_in_file(f'{path}/{f}.tnd', str(funcs[f]['tondir']))
        print('>>> ANF -> TondIR |', f)
        if verbose:
            print('TondIR:\n')
            print(funcs[f]['tondir'])
            print('\n\n')
        ########################
        funcs[f]['sql'] = funcs[f]['tondir'].to_sql(tondir_global_context['database'])
        store_in_file(f'{path}/{f}.sql', funcs[f]['sql'])
        if verbose:
            print('SQL:\n')
            print(funcs[f]['sql'])
            print('\n\n')
        print(">>> TondIR -> SQL |", f)
        ########################
        funcs[f]['tondir_optimized'] = optimize(funcs[f]['tondir'], pytond_context['database'][used_database])
        store_in_file(f'{path}/{f}_opt.tnd', str(funcs[f]['tondir_optimized']))
        if verbose:
            print('TondIR (Optimized):\n')
            print(funcs[f]['tondir_optimized'])
            print('\n\n')
        ########################
        funcs[f]['sql_optimized'] = funcs[f]['tondir_optimized'].to_sql(tondir_global_context['database'])
        store_in_file(f'{path}/{f}_opt.sql', funcs[f]['sql_optimized'])
        if verbose:
            print('SQL (Optimized):\n')
            print(funcs[f]['sql_optimized'])
            print('\n\n')
        ########################
            
##################################################################################
