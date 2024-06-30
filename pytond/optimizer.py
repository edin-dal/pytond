import copy
from pytond.tondir import *

fresh_name_counter = 0
optimization_context = {}

DEBUG_MODE = False

def print_program_rule_by_rule(program):
    count = 0
    for r in optimization_context['current_program'].rules:
        print('\nrule#', count, r, '\n')
        count += 1

def traverse_tondir_expr(expr, func, visit_stack):
    visit_stack.append(expr)
    func(visit_stack)
    
    if isinstance(expr, Program):
        for rule in expr.rules:
            traverse_tondir_expr(rule, func, visit_stack)
    elif isinstance(expr, Rule):
        visit_stack.append("head_rel_access")
        traverse_tondir_expr(expr.rel_access, func, visit_stack)
        visit_stack.pop()
        visit_stack.append("head_group_vars")
        traverse_tondir_expr(expr.group_vars, func, visit_stack)
        visit_stack.pop()
        visit_stack.append("head_sort_vars")
        traverse_tondir_expr(expr.sort_vars, func, visit_stack)
        visit_stack.pop()
        visit_stack.append("body")
        traverse_tondir_expr(expr.body, func, visit_stack)
        visit_stack.pop()
    elif isinstance(expr, RelationAccess):
        traverse_tondir_expr(expr.vars, func, visit_stack)
    elif isinstance(expr, Variable):
        pass
    elif isinstance(expr, Outer):
        traverse_tondir_expr(expr.left_var, func, visit_stack)
        traverse_tondir_expr(expr.right_var, func, visit_stack)
    elif isinstance(expr, Exist):
        traverse_tondir_expr(expr.body, func, visit_stack)
    elif isinstance(expr, Theta):
        traverse_tondir_expr(expr.term1, func, visit_stack)
        traverse_tondir_expr(expr.term2, func, visit_stack)
    elif isinstance(expr, ConstantRelation):
        traverse_tondir_expr(expr.vars, func, visit_stack)
        traverse_tondir_expr(expr.values, func, visit_stack)
    elif isinstance(expr, Constant):
        pass
    elif isinstance(expr, External):
        for arg in expr.args:
            traverse_tondir_expr(arg, func, visit_stack)
    elif isinstance(expr, list):
        for item in expr:
                traverse_tondir_expr(item, func, visit_stack)
    elif isinstance(expr, If):
        traverse_tondir_expr(expr.condition, func, visit_stack)
        traverse_tondir_expr(expr.term1, func, visit_stack)
        traverse_tondir_expr(expr.term2, func, visit_stack)
    elif isinstance(expr, BinOp):
        traverse_tondir_expr(expr.term1, func, visit_stack)
        traverse_tondir_expr(expr.term2, func, visit_stack)
    elif isinstance(expr, Aggregate):
        traverse_tondir_expr(expr.term, func, visit_stack)
    elif isinstance(expr, (int, float, str, bool, type(None))):
        pass
    else:
        print(expr)
        raise Exception("Traversal Error: Unknown Expression Type: ", type(expr))
    
    visit_stack.pop()

def fresh_name(prefeix):
    global fresh_name_counter
    fresh_name_counter += 1
    return prefeix + "_" + str(fresh_name_counter)

def preprocessing(program):
    # removing unnecessary sort
    for rid in range(len(program.rules)):
        rule = program.rules[rid]
        if rule.sort_vars != []:
            if len(rule.sort_vars) == 1:
                sort_var = rule.sort_vars[0]
                related_uid_generator_assignment = None
                for x in rule.body:
                    if isinstance(x, Theta) and x.type == ThetaType.ASSIGN:
                        if isinstance(x.term2, External) and x.term2.type == ExternalType.UID:
                            if x.term1.name == sort_var.name:
                                related_uid_generator_assignment = x
                                break
                if related_uid_generator_assignment is not None:
                    program.rules[rid].sort_vars = []
                    program.rules[rid].sort_order = None 
                    program.rules[rid].body = [x for x in program.rules[rid].body if x != related_uid_generator_assignment]

    # removing unnecessary final ID ordering
    last_rule = program.rules[-1]
    last_rule_rel = optimization_context['data_context'][last_rule.rel_access.name]
    if last_rule.sort_vars != []:
        remove_id  = True
        if len(last_rule.sort_vars) == 1:
            sort_var_names = [x.name for x in last_rule.sort_vars]
            last_rule_rel_access_var_names = [x.name for x in last_rule.rel_access.vars]
            if sort_var_names[0] in last_rule_rel_access_var_names:
                sort_var_index = last_rule_rel_access_var_names.index(sort_var_names[0])
                sorted_col_name = last_rule_rel.cols[sort_var_index]
                if sorted_col_name == 'ID':
                    remove_id = False
            else:
                remove_id = False
        if remove_id:
            if 'ID' in last_rule_rel.cols:
                last_rule_rel.cols = last_rule_rel.cols[1:]
                last_rule_rel.index = None
                last_rule.rel_access.vars = last_rule.rel_access.vars[1:]

    sorted_rules = [rid for rid in range(len(program.rules)) if program.rules[rid].sort_vars != []]
    if len(sorted_rules) == 0:
        if last_rule_rel.cols[0] == 'ID':
            last_rule_rel.cols = last_rule_rel.cols[1:]
            last_rule_rel.index = None
            last_rule.rel_access.vars = last_rule.rel_access.vars[1:]

def find_breaker_rules(program):

    breaker_rules = set()

    #### helper visitor functions #################

    ## aggregate visitor
    def aggregate_visitor_func(visit_stack):
        last_expr = visit_stack[-1]
        if isinstance(last_expr, Aggregate):
            optimization_context['agg_found'] = True

    ## distinct visitor
    def distinct_visitor_func(visit_stack):
        last_expr = visit_stack[-1]
        if isinstance(last_expr, External) and last_expr.type == ExternalType.UNIQUE:
            optimization_context['distinct_found'] = True

    ## outer-join visitor
    def outer_visitor_func(visit_stack):
        last_expr = visit_stack[-1]
        if isinstance(last_expr, Outer):
            optimization_context['outer_found'] = True

    ## exist visitor
    def exist_visitor_func(visit_stack):
        last_expr = visit_stack[-1]
        if isinstance(last_expr, Exist):
            for x in last_expr.body:
                if isinstance(x, RelationAccess):
                    optimization_context['exist_inner_rel_access'].append(x)
                    break
    
    ##############################################

    for i in range(len(program.rules)):
        optimization_context['agg_found'] = False
        optimization_context['group_found'] = False
        optimization_context['distinct_found'] = False
        optimization_context['sort_found'] = False
        optimization_context['outer_found'] = False
        optimization_context['exist_inner_rel_access'] = []

        rule = program.rules[i]

        traverse_tondir_expr(rule, aggregate_visitor_func, [])
        traverse_tondir_expr(rule, distinct_visitor_func, [])
        traverse_tondir_expr(rule, outer_visitor_func, [])
        traverse_tondir_expr(rule, exist_visitor_func, [])

        if  optimization_context['agg_found'] or \
            optimization_context['group_found'] or \
            optimization_context['distinct_found'] or \
            optimization_context['outer_found']:
            breaker_rules.add(i)

        if rule.group_vars != []:
            optimization_context['group_found'] = True
            breaker_rules.add(i)

        if rule.sort_vars != []:
            correct_breaker = True
            if i == (len(program.rules)-1):
                correct_breaker = False

            if correct_breaker:
                optimization_context['sort_found'] = True
                breaker_rules.add(i)


        if optimization_context['exist_inner_rel_access'] != []:
            for rel in optimization_context['exist_inner_rel_access']:
                rel_name = rel.name
                for j in range(0, i):
                    if program.rules[j].rel_access.name == rel_name:
                        breaker_rules.add(j)

        if optimization_context['outer_found']:
            two_rel_accesses = []
            for x in rule.body:
                if isinstance(x, RelationAccess):
                    two_rel_accesses.append(x)
            if len(two_rel_accesses) == 2:
                for rel in two_rel_accesses:
                    rel_name = rel.name
                    for j in range(0, i):
                        if program.rules[j].rel_access.name == rel_name:
                            breaker_rules.add(j)

    del optimization_context['agg_found']
    del optimization_context['group_found']
    del optimization_context['distinct_found']
    del optimization_context['sort_found']
    del optimization_context['outer_found']
    del optimization_context['exist_inner_rel_access']

    res = list(breaker_rules)
    return res

def find_dependency_graph(program):

    ## dependency visitor helper function ###############################
    def find_dependency_visitor_func(visit_stack):
        if len(visit_stack) < 2:
            return
        parent_expr = visit_stack[-2]
        last_expr = visit_stack[-1]
        if isinstance(last_expr, RelationAccess):
            # and isinstance(parent_expr, str) and parent_expr == "body":
            if isinstance(parent_expr, str) and parent_expr == "head_rel_access":
                return
            rel_name = last_expr.name
            for i in range(len(program.rules)):
                if program.rules[i].rel_access.name == rel_name:
                    optimization_context['dependency_graph_tmp'].append(i)
                optimization_context['dependency_graph_tmp'] = list(set(optimization_context['dependency_graph_tmp']))

    #####################################################################

    dependency_graph = {}
    for i in range(len(program.rules)):
        dependency_graph[i] = []
        optimization_context['dependency_graph'] = dependency_graph



    for i in range(len(program.rules)):
        optimization_context['dependency_graph_tmp'] = []
        rule = program.rules[i]
        traverse_tondir_expr(rule, find_dependency_visitor_func, [])
        optimization_context['dependency_graph'][i] = optimization_context['dependency_graph_tmp']

    del optimization_context['dependency_graph_tmp']
    del optimization_context['dependency_graph']

    return dependency_graph

def inverse_dep_graph(dependency_graph):
    inverse_depency_graph = {}
    rule_ids = sorted(list(dependency_graph.keys()))
    last_rule_id = rule_ids[-1]
    
    for i in dependency_graph.keys():
        inverse_depency_graph[i] = []

    for i, lst in dependency_graph.items():
        for j in lst:
            inverse_depency_graph[j].append(i)

    for i in inverse_depency_graph.keys():
        inverse_depency_graph[i] = list(set(inverse_depency_graph[i]))

    del inverse_depency_graph[last_rule_id]
    return inverse_depency_graph

def hash_program(program):
    hash_str = hash(str(program))
    return hash_str

def group_elimination(rule_id):
    global optimization_context

    rule = optimization_context['current_program'].rules[rule_id]

    if rule.group_vars == []:
        return False
    
    if len(rule.group_vars) != 1:
        # print("Group-By Elimination: Multiple Group-By Variables are not supported")
        return False
    
    group_var = rule.group_vars[0]

    body_rel_accesses = []
    for x in rule.body:
        if isinstance(x, RelationAccess):
            body_rel_accesses.append(x)

    if len(body_rel_accesses) != 2:
        return False
    
    # Condition
    found = False
    if group_var == body_rel_accesses[0].vars[0]:
        for i in range(len(optimization_context['current_program'].rules)):
            if optimization_context['current_program'].rules[i].rel_access.name == body_rel_accesses[1].name:
                body = optimization_context['current_program'].rules[i].body
                constant_rels = [x for x in body if isinstance(x, ConstantRelation)]
                if len(constant_rels) == 1:
                    uids = [x for x in body if isinstance(x, Theta) and x.type == ThetaType.ASSIGN \
                            and isinstance(x.term2, External) and x.term2.type == ExternalType.UID]
                    if len(uids) == 1:
                        if isinstance(constant_rels[0].values, list):
                            if len(constant_rels[0].values) != 0:
                                if isinstance(constant_rels[0].values[0], list):
                                    if len(constant_rels[0].values[0]) != 0:
                                        if isinstance(constant_rels[0].values[0][0], Constant):
                                            found = True
                                            break

    if not found:
        return False

    optimization_context['current_program'].rules[rule_id].group_vars = []  

    for i in range(len(rule.body)):
        if isinstance(rule.body[i], Theta) and rule.body[i].type == ThetaType.ASSIGN:
            if isinstance(rule.body[i].term2, Aggregate):
                optimization_context['current_program'].rules[rule_id].body[i].term2 = rule.body[i].term2.term

    return True

def self_join_elimination(rule_id):
    global optimization_context

    rule = optimization_context['current_program'].rules[rule_id]

    body_rel_access_indices = []
    for i in range(len(rule.body)):
        if isinstance(rule.body[i], RelationAccess):
            body_rel_access_indices.append(i)

    if len(body_rel_access_indices) < 2:
        return False
    
    rel_usage_count = {}
    for i in range(len(body_rel_access_indices)):
        rel_access = rule.body[body_rel_access_indices[i]]
        if rel_access.name not in rel_usage_count:
            rel_usage_count[rel_access.name] = [body_rel_access_indices[i]]
        else:
            rel_usage_count[rel_access.name].append(body_rel_access_indices[i])

    for _, indices in rel_usage_count.items():
        if len(indices) < 2:
            continue

        self_join_handled = False
        vars_to_be_replaced = rule.body[indices[1]].vars
        vars_for_replacement = rule.body[indices[0]].vars
        mapping = {}
        for i in range(len(vars_to_be_replaced)):
            mapping[vars_to_be_replaced[i].name] = vars_for_replacement[i].name

        related_rel_0 = optimization_context['data_context'][rule.body[indices[0]].name]
        related_rel_1 = optimization_context['data_context'][rule.body[indices[1]].name]
        index_0 = related_rel_0.index
        index_1 = related_rel_1.index
        if index_0 is None or index_1 is None:
            return False
        
        id_var_0 = rule.body[indices[0]].vars[index_0].name
        id_var_1 = rule.body[indices[1]].vars[index_1].name

        for i in range(len(rule.body)):
            if isinstance(rule.body[i], Theta) and rule.body[i].type == ThetaType.EQ:
                if isinstance(rule.body[i].term1, Variable) and isinstance(rule.body[i].term2, Variable):
                    if (rule.body[i].term1.name == id_var_0 and rule.body[i].term2.name == id_var_1) or (rule.body[i].term1.name == id_var_1 and rule.body[i].term2.name == id_var_0):
                        # Removing Extra Relation Access and ID Comparison
                        optimization_context['current_program'].rules[rule_id].body = [rule.body[j] for j in range(len(rule.body)) if j not in [indices[1], i]]
                        self_join_handled = True
                        break

        if self_join_handled:
            def var_replace_visior_func(visit_stack):
                if isinstance(visit_stack[-1], Variable):
                    if visit_stack[-1].name in mapping:
                        visit_stack[-1].name = mapping[visit_stack[-1].name]
            traverse_tondir_expr(optimization_context['current_program'].rules[rule_id], var_replace_visior_func, [])
            return True

    return False

def is_var_used_in_rule(var, rule):
    global optimization_context

    def var_used_visitor_func(visit_stack):
        if len(visit_stack) < 2:
            return
        
        last_expr = visit_stack[-1]
        parent_expr = visit_stack[-2]
        parent_parent_expr = None
        parent_parent_parent_expr = None
        parent_parent_parent_parent_expr = None
        if len(visit_stack) > 2:
            parent_parent_expr = visit_stack[-3]
        if len(visit_stack) > 3:
            parent_parent_parent_expr = visit_stack[-4]
        if len(visit_stack) > 4:
            parent_parent_parent_parent_expr = visit_stack[-5]

        if isinstance(last_expr, Variable) and last_expr.name == var.name:
            
            if isinstance(parent_expr, list) and \
                isinstance(parent_parent_expr, RelationAccess) and \
                isinstance(parent_parent_parent_expr, str) and \
                parent_parent_parent_expr == "head_rel_access":
                optimization_context['var_used_in_head_rel_access'] = True
            
            elif isinstance(parent_expr, list) and \
                isinstance(parent_parent_expr, Variable) and \
                isinstance(parent_parent_parent_expr, str) and \
                parent_parent_parent_expr == "head_group_vars":
                optimization_context['var_used_in_head_group_vars'] = True

            elif isinstance(parent_expr, list) and \
                isinstance(parent_parent_expr, Variable) and \
                isinstance(parent_parent_parent_expr, str) and \
                parent_parent_parent_expr == "head_sort_vars":
                optimization_context['var_used_in_head_sort_vars'] = True

            elif  isinstance(parent_parent_expr, Theta) and \
                parent_parent_expr.type == ThetaType.EQ and \
                parent_parent_expr.term1 == last_expr:
                optimization_context['var_used_in_body_theta_left'] = True

            else:
                exception = False

                if isinstance(parent_expr, list) and \
                    isinstance(parent_parent_expr, RelationAccess) and \
                    isinstance(parent_parent_parent_expr, list) and \
                    isinstance(parent_parent_parent_parent_expr, str) and \
                    parent_parent_parent_parent_expr == "body":
                    exception = True

                if not exception:
                    optimization_context['var_used_in_body_other'] = True

    optimization_context['var_used_in_head_rel_access'] = False
    optimization_context['var_used_in_head_group_vars'] = False
    optimization_context['var_used_in_head_sort_vars'] = False
    optimization_context['var_used_in_body_theta_left'] = False
    optimization_context['var_used_in_body_other'] = False

    traverse_tondir_expr(rule, var_used_visitor_func, [])
    
    used_in_head_rel_access = optimization_context['var_used_in_head_rel_access']
    used_in_head_group_vars = optimization_context['var_used_in_head_group_vars']
    used_in_head_sort_vars = optimization_context['var_used_in_head_sort_vars']
    used_in_body_theta_left = optimization_context['var_used_in_body_theta_left']
    used_in_body_other = optimization_context['var_used_in_body_other']

    final_result = False

    if  (used_in_head_rel_access or \
        used_in_head_group_vars or \
        used_in_head_sort_vars or \
        used_in_body_other) and \
        not used_in_body_theta_left:
        final_result = True
    
    if used_in_body_theta_left and used_in_body_other:
        final_result = True
    

    del optimization_context['var_used_in_head_rel_access']
    del optimization_context['var_used_in_head_group_vars']
    del optimization_context['var_used_in_head_sort_vars']
    del optimization_context['var_used_in_body_theta_left']
    del optimization_context['var_used_in_body_other']

    return final_result

def unused_var_elimination(rule_id):
    global optimization_context

    removal_result = {}
    rule = optimization_context['current_program'].rules[rule_id]

    # Find relation accesses in the rule
    rel_access_ids_in_rule_body = []
    for i in range(len(rule.body)):
        if isinstance(rule.body[i], RelationAccess):
            # Exclude the database relation accesses
            if rule.body[i].name.startswith("V"):
                rel_access_ids_in_rule_body.append(i)

    # Trying to eliminate unused variables in each relation access
    for rel_access_id in rel_access_ids_in_rule_body:
        rel_access = optimization_context['current_program'].rules[rule_id].body[rel_access_id]

        vars_to_be_eliminated = []
        for v in range(len(rel_access.vars)):
            var = rel_access.vars[v]

            used = is_var_used_in_rule(var, rule)

            if used:
                continue
            # See if it is used in the other user rules
            rel_access_original = None
            for r in range(len(optimization_context['current_program'].rules)):
                if optimization_context['current_program'].rules[r].rel_access.name == rel_access.name:
                    rel_access_original = optimization_context['current_program'].rules[r].rel_access
                    break
            if rel_access_original is None:
                raise Exception("Unused Var Elimination: Original Relation Access Not Found")
            
            users = optimization_context['inv_dep_graph'][r]
            used_in_any_user = False
            for u in users:
                u_rule = optimization_context['current_program'].rules[u]
                relevant_rel_accesses = []
                for j in range(len(optimization_context['current_program'].rules[u].body)):
                    if isinstance(optimization_context['current_program'].rules[u].body[j], RelationAccess):
                        if optimization_context['current_program'].rules[u].body[j].name == rel_access.name:
                            relevant_rel_accesses.append(optimization_context['current_program'].rules[u].body[j])
                if relevant_rel_accesses == []:
                    raise Exception("Unused Var Elimination: Original Relation Access Not Found in User Rule")
                for relevant_rel_access in relevant_rel_accesses:
                    if is_var_used_in_rule(relevant_rel_access.vars[v], u_rule):
                        used_in_any_user = True
                        break
            if used_in_any_user:
                continue
            else:
                vars_to_be_eliminated.append(v)

        if len(vars_to_be_eliminated) > 0:
            removal_result[rel_access_original.name] = [rel_access_original.vars[l].name for l in vars_to_be_eliminated]
            
            # Eliminate the variables from the original rel and rel access
            related_rel = optimization_context['data_context'][rel_access_original.name]

            if related_rel.index in vars_to_be_eliminated:
                if (rule_id != (len(optimization_context['current_program'].rules)-1)) and (rule_id != len(optimization_context['current_program'].rules)-2):
                    related_rel.index = None
                    rule.sort_vars = []
                    rule.sort_order = None

            new_vars = [rel_access_original.vars[l] for l in range(len(rel_access_original.vars)) if l not in vars_to_be_eliminated]
            rel_access_original.vars = new_vars
            related_rel.cols = [related_rel.cols[l] for l in range(len(related_rel.cols)) if l not in vars_to_be_eliminated]

            # Eliminate the variables from all the user rules
            for u in users:
                relevant_rel_access_ids = []
                for j in range(len(optimization_context['current_program'].rules[u].body)):
                    if isinstance(optimization_context['current_program'].rules[u].body[j], RelationAccess) and \
                        optimization_context['current_program'].rules[u].body[j].name == rel_access.name:
                        relevant_rel_access_ids.append(j)
                for j in relevant_rel_access_ids:
                    relevant_rel_access = optimization_context['current_program'].rules[u].body[j]
                    new_vars = [relevant_rel_access.vars[l] for l in range(len(relevant_rel_access.vars)) if l not in vars_to_be_eliminated]
                    relevant_rel_access.vars = new_vars

    return removal_result

def build_program_stats(program):
    # Find Breaker Rules
    breakers = find_breaker_rules(optimization_context['current_program'])
    optimization_context['breakers'] = breakers
    # print(">>> Breakers: ", breakers)
    ####################################################################
    # Dependency Graph
    dep_graph = find_dependency_graph(optimization_context['current_program'])
    optimization_context['dep_graph'] = dep_graph
    # print(">>> Dependency Graph: ", dep_graph)
    ####################################################################
    # Inverse Dependency Graph
    inv_dep_graph = inverse_dep_graph(dep_graph)
    optimization_context['inv_dep_graph'] = inv_dep_graph
    # print(">>> Inverse Dependency Graph: ", inv_dep_graph)

def dead_rule_elimination(program):
    global optimization_context
    inv_dep_graph = optimization_context['inv_dep_graph']
    indices_to_remove = []
    for i, lst in inv_dep_graph.items():
        if len(lst) == 0:
            indices_to_remove.append(i)
            if DEBUG_MODE:
                print("\033[94m>>> Dead Rule Elimination Success. Rule: ", i, "\033[0m")

    program.rules = [program.rules[i] for i in range(len(program.rules)) if i not in indices_to_remove]
    build_program_stats(program)

    return program

def inline_i1_to_i2(i1, i2):
    global optimization_context

    ## Inlining ##################################################


    v1 = optimization_context['current_program'].rules[i1]
    v1_rel_access = v1.rel_access
    v2 = optimization_context['current_program'].rules[i2]

    target_rel_access_in_v2_id = None
    for i in range(len(v2.body)):
        if isinstance(v2.body[i], RelationAccess) and v2.body[i].name == v1_rel_access.name:
            target_rel_access_in_v2_id = i
            break
    v2_target_rel_access = v2.body[target_rel_access_in_v2_id]


    v1_vars = v1_rel_access.vars
    v2_vars = v2_target_rel_access.vars

    if len(v1_vars) != len(v2_vars):
        raise Exception("Inlining Error: Number of Variables Mismatch")

    var_mapping = {}
    for i in range(len(v1_vars)):
        var_mapping[v1_vars[i].name] = v2_vars[i].name
    
    v1_body_deep_copy = copy.deepcopy(optimization_context['current_program'].rules[i1].body)

    def inline_visitor_func(visit_stack):
        last_expr = visit_stack[-1]
        if isinstance(last_expr, Variable):
            if last_expr.name in var_mapping:
                last_expr.name = var_mapping[last_expr.name]

    traverse_tondir_expr(v1_body_deep_copy, inline_visitor_func, [])

    if target_rel_access_in_v2_id is None:
        raise Exception("Inlining Error: Target Relation Access Not Found in Rule")


    v2.body = v2.body[:target_rel_access_in_v2_id] + v1_body_deep_copy + v2.body[target_rel_access_in_v2_id+1:]

    ## Inlining Assignments ##################################################

    assignments = [x for x in v2.body if isinstance(x, Theta) and x.type == ThetaType.ASSIGN]
    for assignment in assignments:
        var = assignment.term1
        
        if isinstance(assignment.term2, External) and assignment.term2.type == ExternalType.UID:
            continue

        # visit the body and replace the var with the term2
        def replace_visitor_func(visit_stack):

            last_expr = visit_stack[-1]
            parent_expr = None
            parent_parent_expr = None
            parent_parent_parent_expr = None

            if len(visit_stack) > 1:
                parent_expr = visit_stack[-2]
            if len(visit_stack) > 2:
                parent_parent_expr = visit_stack[-3]
            if len(visit_stack) > 3:
                parent_parent_parent_expr = visit_stack[-4]

            if isinstance(last_expr, Variable) and last_expr.name == var.name:

                if isinstance(parent_expr, list) and \
                    isinstance(parent_parent_expr, str) and \
                        parent_parent_expr.startswith("head"):
                    if isinstance(assignment.term2, Variable):
                        idx = parent_expr.index(last_expr)
                        parent_expr[idx].name = assignment.term2.name
                    return
                
                if isinstance(parent_expr, list) and \
                    isinstance(parent_parent_expr, RelationAccess) and \
                        isinstance(parent_parent_parent_expr, str) and \
                            parent_parent_parent_expr == "head_rel_access":
                    return
                
                if isinstance(parent_expr, Theta) and parent_expr.type == ThetaType.ASSIGN:
                    if parent_expr.term1 == last_expr:
                        return

                if isinstance(parent_expr, External):
                    for i in range(len(parent_expr.args)):
                        if isinstance(parent_expr.args[i], Variable) and parent_expr.args[i].name == var.name:
                            parent_expr.args[i] = assignment.term2
                        elif isinstance(parent_expr.args[i], list):
                            for j in range(len(parent_expr.args[i])):
                                if isinstance(parent_expr.args[i][j], Variable) and parent_expr.args[i][j].name == var.name:
                                    parent_expr.args[i][j] = assignment.term2
                            
                elif isinstance(parent_expr, Theta):
                    if parent_expr.type == ThetaType.ASSIGN:
                        parent_expr.term2 = assignment.term2
                    else:
                        # generic case
                        if parent_expr.term1 == last_expr:
                            parent_expr.term1 = assignment.term2
                        elif parent_expr.term2 == last_expr:
                            parent_expr.term2 = assignment.term2
                elif isinstance(parent_expr, BinOp):
                    if parent_expr.term1 == last_expr:
                        parent_expr.term1 = assignment.term2
                    elif parent_expr.term2 == last_expr:
                        parent_expr.term2 = assignment.term2
                elif isinstance(parent_expr, list):
                    exception = False
                    if parent_parent_expr is not None and isinstance(parent_parent_expr, External) and parent_parent_expr.type == ExternalType.UID:
                        exception = True
                    if parent_parent_expr is not None and isinstance(parent_parent_expr, RelationAccess):
                        exception = True
                    if not exception:
                        index = parent_expr.index(last_expr)
                        parent_expr[index] = assignment.term2
                elif isinstance(parent_expr, Aggregate):
                    parent_expr.term = assignment.term2
                elif isinstance(parent_expr, If):
                    if parent_expr.condition == last_expr:
                        parent_expr.condition = assignment.term2
                    elif parent_expr.term1 == last_expr:
                        parent_expr.term1 = assignment.term2
                    elif parent_expr.term2 == last_expr:
                        parent_expr.term2 = assignment.term2
                else:
                    print(parent_expr)
                    raise Exception("Unknown Parent Expression Type")

        traverse_tondir_expr(v2, replace_visitor_func, [])

    ## Eliminating Specific Self-Join Case ##################################################
    theta_for_uid_comparison = None
    for i in range(len(v2.body)):
        if isinstance(v2.body[i], Theta) and v2.body[i].type == ThetaType.EQ:
            if isinstance(v2.body[i].term1, Variable) and isinstance(v2.body[i].term2, Variable):
                assignments = [x for x in v2.body if isinstance(x, Theta) and x.type == ThetaType.ASSIGN]
                uid_assign_1 = [x for x in assignments if x.term1.name == v2.body[i].term1.name and isinstance(x.term2, External) and x.term2.type == ExternalType.UID]
                uid_assign_2 = [x for x in assignments if x.term1.name == v2.body[i].term2.name and isinstance(x.term2, External) and x.term2.type == ExternalType.UID]
                if len(uid_assign_1) == 1 and len(uid_assign_2) == 1:
                    theta_for_uid_comparison = v2.body[i]
                    continue
                elif (len(uid_assign_1) == 1 and len(uid_assign_2)==0) or (len(uid_assign_1) == 0 and len(uid_assign_2)==1):
                    theta_for_uid_comparison = v2.body[i]
                    continue
    v2.body = [x for x in v2.body if x != theta_for_uid_comparison]

    ## Constant Relation Join Handling ################################
    rel_accesses = [x for x in v2.body if isinstance(x, RelationAccess)]
    cons_rel_acesses = [x for x in v2.body if isinstance(x, ConstantRelation)]
    if len(cons_rel_acesses) == 1 and len(rel_accesses) == 1:
        left = rel_accesses[0]
        right = cons_rel_acesses[0]
        if len(right.values) == 1:
            if isinstance(right.values[0], list):
                if isinstance(right.values[0][0], Constant):
                    uid = [x for x in v2.body if isinstance(x, Theta) and \
                            x.type == ThetaType.ASSIGN and \
                            isinstance(x.term2, External) and \
                            x.term2.type == ExternalType.UID and \
                            x.term1.name == v2.rel_access.vars[0].name]
                    v2.body = [x for x in v2.body if x not in uid]
                    v2.body.append(Theta(ThetaType.ASSIGN, v2.rel_access.vars[0], left.vars[0]))

                    def replace_visitor_func(visit_stack):
                        if len(visit_stack) > 1:
                            last_expr = visit_stack[-1]
                            parent_expr = visit_stack[-2]
                        else:
                            return
                        if isinstance(last_expr, Variable):
                            if isinstance(parent_expr, BinOp):
                                if isinstance(parent_expr.term1, Variable) and parent_expr.term1.name == last_expr.name:
                                    if last_expr.name in mapping:
                                        parent_expr.term1 = mapping[last_expr.name]
                                if isinstance(parent_expr.term2, Variable) and parent_expr.term2.name == last_expr.name:
                                    if last_expr.name in mapping:
                                        parent_expr.term2 = mapping[last_expr.name]
                    mapping = {}
                    for v in range(len(right.vars)):
                        mapping[right.vars[v].name] = right.values[0][v]
                    traverse_tondir_expr(v2, replace_visitor_func, [])

                v2.body = [x for x in v2.body if x != right]


    #####################################################################################
    ## Eliminating Repeated Relation Accesses ###########################################

    rule_copy = copy.deepcopy(v2)

    rel_access_ids = []

    for i in range(len(rule_copy.body)):
        if isinstance(rule_copy.body[i], RelationAccess):
            rel_access_ids.append(i)
    
    def var_usage_visitor_func(visit_stack):
        last_expr = visit_stack[-1]
        if isinstance(last_expr, Variable):
            if last_expr.name == lookup_var.name:
                optimization_context['is_var_used'] = True
    
    rel_access_usages = {}
    for i in range(len(rel_access_ids)):
        rel_access = v2.body[rel_access_ids[i]]
        rel_access_usages[rel_access_ids[i]] = []
        for j in range(len(rel_access.vars)):
            var = rel_access.vars[j]
            new_deep_copy = copy.deepcopy(rule_copy)
            same_name_rels = []
            for x in rel_access_ids:
                if rel_access.name == new_deep_copy.body[x].name:
                    same_name_rels.append(x)
            new_deep_copy.body = [new_deep_copy.body[x] for x in range(len(new_deep_copy.body)) if x not in same_name_rels]
            lookup_var = rel_access.vars[j]
            optimization_context['is_var_used'] = False
            traverse_tondir_expr(new_deep_copy, var_usage_visitor_func, [])
            if optimization_context['is_var_used']:
                rel_access_usages[rel_access_ids[i]].append(j)
            del optimization_context['is_var_used']

    rel_access_ids_to_remove = []

    # remove the rel_accesses where one usage is subset of the other
    for i in range(len(rel_access_ids)):
        for j in range(len(rel_access_ids)):
            rel_access_i = rule_copy.body[rel_access_ids[i]]
            rel_access_j = rule_copy.body[rel_access_ids[j]]
            if  (rel_access_ids[i] in rel_access_ids_to_remove) or \
                (rel_access_ids[j] in rel_access_ids_to_remove) or \
                rel_access_i.name != rel_access_j.name or \
                i >= j:
                continue
            
            usages_i = rel_access_usages[rel_access_ids[i]]
            usages_j = rel_access_usages[rel_access_ids[j]]

            i_set = set(usages_i)
            j_set = set(usages_j)

            if i_set.issubset(j_set) or j_set.issubset(i_set):
                shared_used_vars = i_set.intersection(j_set)
                share_vars_i_names = sorted([rel_access_i.vars[k].name for k in shared_used_vars])
                share_vars_j_names = sorted([rel_access_j.vars[k].name for k in shared_used_vars])
                if share_vars_i_names == share_vars_j_names:
                    # remove the rel_access that has less usages
                    if len(usages_i) < len(usages_j):
                        rel_access_ids_to_remove.append(rel_access_ids[i])
                    else:
                        rel_access_ids_to_remove.append(rel_access_ids[j])

    # handle fusable rel_accesses
    for i in range(len(rel_access_ids)):
        for j in range(len(rel_access_ids)):
            rel_access_i = v2.body[rel_access_ids[i]]
            rel_access_j = v2.body[rel_access_ids[j]]

            if  (rel_access_ids[i] in rel_access_ids_to_remove) or \
                (rel_access_ids[j] in rel_access_ids_to_remove) or \
                rel_access_i.name != rel_access_j.name or \
                i >= j:
                continue
            
            usages_i = rel_access_usages[rel_access_ids[i]]
            usages_j = rel_access_usages[rel_access_ids[j]]
            shared_used_vars = set(usages_i).intersection(set(usages_j))

            involvements_in_theta_with_other_rel_access_i = []
            involvements_in_theta_with_other_rel_access_j = []

            for usages in [i, j]:
                for u in rel_access_usages[rel_access_ids[usages]]:
                    used_var = v2.body[rel_access_ids[usages]].vars[u]
                    if u in shared_used_vars:
                        if rel_access_i.vars[u].name == rel_access_j.vars[u].name:
                            continue

                    # check if var is used in a non-assignment theta with at least one other rel_access than rel_access_i
                    for k in range(len(rule_copy.body)):
                        # TODO FUTURE: This check can be done using a more sophisticated visitor function
                        if isinstance(v2.body[k], Theta) and v2.body[k].type != ThetaType.ASSIGN:
                            if isinstance(v2.body[k].term1, Variable) and isinstance(v2.body[k].term2, Variable):
                                if used_var.name in [v2.body[k].term1.name, v2.body[k].term2.name]:
                                    # put the name of the other rel_access
                                    other_name = v2.body[k].term1.name if used_var.name == v2.body[k].term2.name else v2.body[k].term2.name
                                    # find the relevant rel_access of the other var name
                                    for l in range(len(rel_access_ids)):
                                        if (l in rel_access_ids_to_remove): 
                                            continue
                                        if other_name in [v2.body[rel_access_ids[l]].vars[m].name for m in range(len(v2.body[rel_access_ids[l]].vars))]:
                                            if usages == i:
                                                involvements_in_theta_with_other_rel_access_i.append(rel_access_ids[l])
                                            else:
                                                involvements_in_theta_with_other_rel_access_j.append(rel_access_ids[l])
                                            break
            
            fuse_needed = None
            if len(involvements_in_theta_with_other_rel_access_i) > 0 and len(involvements_in_theta_with_other_rel_access_j) == 0:
                fuse_needed = (j, i)
            elif len(involvements_in_theta_with_other_rel_access_i) == 0 and len(involvements_in_theta_with_other_rel_access_j) > 0:
                fuse_needed = (i, j)
            elif len(involvements_in_theta_with_other_rel_access_i) == 0 and len(involvements_in_theta_with_other_rel_access_j) == 0:
                fuse_needed = (i, j)
            elif len(involvements_in_theta_with_other_rel_access_i) > 0 and len(involvements_in_theta_with_other_rel_access_j) > 0:
                if involvements_in_theta_with_other_rel_access_i == involvements_in_theta_with_other_rel_access_j:
                    fuse_needed = (i, j)

            if fuse_needed:
                rel_access_ids_to_remove.append(rel_access_ids[fuse_needed[0]])

                name_mapping = {}
                usages_1 = rel_access_usages[rel_access_ids[fuse_needed[0]]]
                usages_2 = rel_access_usages[rel_access_ids[fuse_needed[1]]]
                shared_used_vars = set(usages_1).intersection(set(usages_2))

                for k in usages_1:
                    if k not in shared_used_vars:
                        name_mapping[v2.body[rel_access_ids[fuse_needed[1]]].vars[k].name] = \
                            v2.body[rel_access_ids[fuse_needed[0]]].vars[k].name
                    else:
                        name_mapping[v2.body[rel_access_ids[fuse_needed[0]]].vars[k].name] = \
                            v2.body[rel_access_ids[fuse_needed[1]]].vars[k].name

                def replace_name_visitor_func(visit_stack):
                    last_expr = visit_stack[-1]
                    if isinstance(last_expr, Variable):
                        if last_expr.name in name_mapping:
                            last_expr.name = name_mapping[last_expr.name]
                traverse_tondir_expr(v2, replace_name_visitor_func, [])

                vars = sorted(list(set(usages_i + usages_j)))
                rel_access_usages[rel_access_ids[fuse_needed[1]]] = vars


    # remove the rel_accesses that are not used
    rel_access_ids_to_remove = list(set(rel_access_ids_to_remove))
    v2.body = [v2.body[i] for i in range(len(v2.body)) if i not in rel_access_ids_to_remove]

    #####################################################################################
    ## Removing repeated body elements ##################################################
    elements_to_remove = []
    for i in range(len(v2.body)):
        for j in range(i+1, len(v2.body)):
            hash_i = hash(v2.body[i])
            hash_j = hash(v2.body[j])
            # print("Hash I: ", hash_i, v2.body[i])
            # print("Hash J: ", hash_j, v2.body[j])
            if hash_i == hash_j:
                if DEBUG_MODE:
                    print(">>>> Repeated Element Removed: ", v2.body[j])
                elements_to_remove.append(j)

    v2.body = [v2.body[i] for i in range(len(v2.body)) if i not in elements_to_remove]

    #####################################################################################

    return True

def rule_inlining(rule_id):
    global optimization_context

    v1 = optimization_context['current_program'].rules[rule_id]
    v1_rel_access = v1.rel_access

    used_rules = optimization_context['dep_graph'][rule_id]
    rules_to_inline = []
    for r in used_rules:
        breakers = optimization_context['breakers']
        if not (r in breakers):
            rules_to_inline.append(r)

    if len(rules_to_inline) == 0:
        return False
    
    for r in rules_to_inline:
        v2 = optimization_context['current_program'].rules[r]
        v2_rel_access = v2.rel_access
        if DEBUG_MODE:
            print("\033[92m>>> Inlining: ", v1_rel_access.name, " to ", v2_rel_access.name, "\033[0m")
        
        inline_i1_to_i2(r, rule_id)
        build_program_stats(optimization_context['current_program'])

    # Removing the Assignments that are not used in the rule
    for atom in optimization_context['current_program'].rules[rule_id].body:
        if isinstance(atom, Theta) and atom.type == ThetaType.ASSIGN:
            if atom.term1.name not in [x.name for x in optimization_context['current_program'].rules[rule_id].rel_access.vars]:
                optimization_context['current_program'].rules[rule_id].body = [x for x in optimization_context['current_program'].rules[rule_id].body if x != atom]

    return True

#####################################################################

def optimize(tondir_program, data_context):
    global optimization_context

    optimization_context = {
        'program': tondir_program,
        'data_context': data_context
    }

    ### OPTIMIZATION LOOP #################################################################
    print("\033[94m>>> Starting Optimization\033[0m")

    optimization_context['current_program'] = copy.deepcopy(optimization_context['program'])
    program_hash = hash_program(optimization_context['current_program'])
    further_optimization = True
    
    opt_counter = 1
    while further_optimization:

        ## ENABLE FOR STEP-BY-STEP DEBUGGING ################################
        # with open('x.sql', 'w') as f:
        #     f.write(str(optimization_context['current_program'].to_sql(data_context)))
        # with open('y.sql', 'w') as f:
        #     f.write(str(optimization_context['current_program']))
        # read = input("Press Enter to Continue")
        #####################################################################
        print("\033[94m>>> Optimization Iteration: ", opt_counter, "\033[0m")
        # Preprocessing ####################################################
        preprocessing(optimization_context['current_program'])
        build_program_stats(optimization_context['current_program'])
        ####################################################################
        # Group-By Elimination
        for i in range(len(optimization_context['current_program'].rules)):
            done = group_elimination(i)
            if done:
                if DEBUG_MODE:
                    print("\033[94m>>> Group-By Elimination Success. Rule: ", i, "\033[0m")
        ####################################################################
        # Self-Join Elimination
        for i in range(len(optimization_context['current_program'].rules)):
            found = self_join_elimination(i)
            if found:
                if DEBUG_MODE:
                    print("\033[94m>>> Self-Join Elimination Success. Rule: ", i, "\033[0m")
                build_program_stats(optimization_context['current_program'])
        ####################################################################
        # Unused Variable Elimination
        for i in range(len(optimization_context['current_program'].rules)):
            vars_eliminated = unused_var_elimination(i)
            if len(vars_eliminated) > 0:
                if DEBUG_MODE:
                    print("\033[94m>>> Unused Variable Elimination Success. Rule: ", i, "Eliminated Vars: ", vars_eliminated, "\033[0m")
        ####################################################################
        # Rule Inlining
        for i in range(len(optimization_context['current_program'].rules)):
            inlined = rule_inlining(i)
            build_program_stats(optimization_context['current_program'])
            if inlined:
                if DEBUG_MODE:
                    print("\033[94m>>> Rule Inlining Success. Rule: ", i, "\033[0m")
                break

        ####################################################################
        # Dead Rule Elimination
        dead_rule_elimination(optimization_context['current_program'])
        ####################################################################
        ## Optimization Termination Condition
        if hash_program(optimization_context['current_program']) == program_hash:
            further_optimization = False
            print("\033[94m>>> Fully Optimized. \033[0m")
            break
        else:
            program_hash = hash_program(optimization_context['current_program'])

        opt_counter += 1

    #######################################################################################

    return optimization_context['current_program']