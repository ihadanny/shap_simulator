from __future__ import division

def ExtendPath(unique_path, unique_depth, zero_fraction, one_fraction, feature_index):
    pe = {'feature_index': feature_index, 'zero_fraction': zero_fraction,
          'one_fraction': one_fraction, 'pweight': 1 if unique_depth == 0 else 0}
    unique_path.append(pe)
    for i in range(unique_depth - 1, -1, -1):
        unique_path[i+1]['pweight'] += one_fraction * unique_path[i]['pweight'] * (i+1) / (unique_depth+1) 
        unique_path[i]['pweight'] += zero_fraction * unique_path[i]['pweight'] * (unique_depth - i) / (unique_depth+1)
    print("ExtendPath end", unique_path, unique_depth)

def UnwoundPathSum(unique_path, unique_depth, path_index):
    print("UnwoundPathSum", unique_path[path_index]['feature_index'], unique_depth)
    one_fraction = unique_path[path_index]['one_fraction']
    zero_fraction = unique_path[path_index]['zero_fraction']
    next_one_portion = unique_path[unique_depth]['pweight']
    total = 0;
    for i in range(unique_depth-1, -1, -1):
        if one_fraction != 0:
            tmp = next_one_portion*(unique_depth+1) / ((i+1)*one_fraction)
            total += tmp
            next_one_portion = unique_path[i]['pweight'] - tmp*zero_fraction*((unique_depth-i) / (unique_depth+1))
        else:
            total += (unique_path[i]['pweight']/zero_fraction)/((unique_depth-i) / (unique_depth+1))
    return total

# recursive computation of SHAP values for a decision tree
def TreeShap(feat, phi, node_index, unique_depth, parent_unique_path, parent_zero_fraction,
             parent_one_fraction, parent_feature_index):
    node = nodes[node_index]
    print("TreeShap", node["split_index"])
    unique_path = [] 
    for el in parent_unique_path:
        unique_path.append(el.copy())
    ExtendPath(unique_path, unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index)
    #leaf node
    if node['is_leaf']:
        for i in range(1, unique_depth + 1):
            w = UnwoundPathSum(unique_path, unique_depth, i)
            el = unique_path[i]
            print(node['split_index'], "adding to", el['feature_index'], 'w', w, 'one_fraction', el['one_fraction'], 'zero_fraction', el['zero_fraction'],
                  'leaf_value', node['leaf_value'])
            phi[el['feature_index']] += w*(el['one_fraction']-el['zero_fraction'])*node['leaf_value']
    #internal node
    else:
        split_index = node['split_index']
        #find which branch is "hot" (meaning x would follow it)
        if feat[split_index]['fvalue'] < node['split_cond']:
            hot_index, cold_index = node['cleft'], node['cright']
        else:
            hot_index, cold_index = node['cright'], node['cleft']
        w = node['sum_hess']
        hot_zero_fraction = nodes[hot_index]['sum_hess']/w
        cold_zero_fraction = nodes[cold_index]['sum_hess']/w
        incoming_zero_fraction = 1
        incoming_one_fraction = 1
        TreeShap(feat, phi, hot_index, unique_depth+1, unique_path, hot_zero_fraction*incoming_zero_fraction, incoming_one_fraction, split_index)
        TreeShap(feat, phi, cold_index, unique_depth+1, unique_path, cold_zero_fraction*incoming_zero_fraction, 0, split_index);

def CalculateContributions(feat, root_id):
    #find the expected value of the tree's predictions
    base_value = 0.0
    total_cover = 0
    for node in nodes.values():
        if node['is_leaf']:
            cover = node['sum_hess']
            base_value += cover*node['leaf_value']
            total_cover += cover;
    out_contribs = {k: 0 for k in feat.keys()}
    out_contribs['bias'] = base_value / total_cover

    unique_path_data = []
    TreeShap(feat, out_contribs, root_id, 0, unique_path_data, 1, 1, -1)
    return out_contribs

def GetNext(feat, my_node):
    cur_feat = my_node['split_index']
    my_val = feat[cur_feat]['fvalue']
    if my_val < my_node['split_cond']:
        return my_node['cleft']
    else:
        return my_node['cright']
    
def Predict(feat, node_id):
    my_node = nodes[node_id]
    if my_node['is_leaf']:
        return my_node['leaf_value']
    else:
        return Predict(feat, GetNext(feat, my_node))

node_mean_values = {}
def FillNodeMeanValue(node_id):
    my_node = nodes[node_id]
    if my_node['is_leaf']:
        res = my_node['leaf_value']
    else:
        res = FillNodeMeanValue(my_node['cleft']) * nodes[my_node['cleft']]['sum_hess']
        res += FillNodeMeanValue(my_node['cright']) * nodes[my_node['cright']]['sum_hess']
        res /= my_node['sum_hess']
    node_mean_values[node_id] = res    
    return node_mean_values[node_id]

def CalculateContributionsApprox(feat, root_id):
    FillNodeMeanValue(root_id)
    out_contribs = {k: 0 for k in feat.keys()}
    pid = root_id
    node_value = node_mean_values[root_id]
    out_contribs['bias'] = node_value 
    if nodes[pid]['is_leaf']:
        return
    while(not nodes[pid]['is_leaf']):
        split_index = nodes[pid]['split_index']
        pid = GetNext(feat, nodes[pid])
        new_value = node_mean_values[pid]
        out_contribs[split_index] += new_value - node_value
        node_value = new_value
    leaf_value = nodes[pid]['leaf_value']
    out_contribs[split_index] += leaf_value - node_value
    return out_contribs
    
nodes = {
    'A': {'split_index': 'A', 'cleft': 'X1', 'cright': 'B', 'is_leaf': False, 'sum_hess': 1000, 'split_cond': 3},
    'B': {'split_index': 'B', 'cleft': 'X2', 'cright': 'C', 'is_leaf': False, 'sum_hess': 100, 'split_cond': 3},
    'C': {'split_index': 'C', 'cleft': 'X3', 'cright': 'X4', 'is_leaf': False, 'sum_hess': 10, 'split_cond': 3},

    'X1': {'split_index': 'X1', 'is_leaf': True, 'sum_hess': 900, 'leaf_value': 1},
    'X2': {'split_index': 'X2', 'is_leaf': True, 'sum_hess': 90, 'leaf_value': 2},
    'X3': {'split_index': 'X3', 'is_leaf': True, 'sum_hess': 9, 'leaf_value': 3},
    'X4': {'split_index': 'X4', 'is_leaf': True, 'sum_hess': 1, 'leaf_value': 4},
}

nodes = {
    'A': {'split_index': 'A', 'cdefault': 'X1', 'cleft': 'X1', 'cright': 'X2', 'is_leaf': False, 'sum_hess': 1000, 'split_cond': 3},
    'X1': {'split_index': 'X1', 'is_leaf': True, 'sum_hess': 500, 'leaf_value': 1},
    'X2': {'split_index': 'X2', 'is_leaf': True, 'sum_hess': 500, 'leaf_value': 2},
}

feat = {
    'A': {'fvalue': 1},
    'B': {'fvalue': 1},
    'C': {'fvalue': 1},
    'D': {'fvalue': 9999},
    'E': {'fvalue': 9999},
    'F': {'fvalue': 9999},
}

print('Predict', Predict(feat, 'A'))

out_contribs = CalculateContributionsApprox(feat, 'A')
print("CalculateContributionsApprox", out_contribs, sum(out_contribs.values()))

out_contribs = CalculateContributions(feat, 'A')
print("CalculateContributions", out_contribs, sum(out_contribs.values()))
