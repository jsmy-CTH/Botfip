import networkx as nx
from collections import deque


def serialize_tree(tree, root):
    """利用广度优先搜索序列化有向树"""
    serialized = [root]

    # 使用队列实现BFS
    queue = deque([root])
    while queue:
        node = queue.popleft()

        children = list(tree.successors(node))
        serialized.extend(children)

        for child in children:
            if child is not None:
                queue.append(child)

        # 添加'end'来标记当前节点的子节点列表结束
        serialized.append('end')

    return serialized

def serialize_tree_with_value(tree, root, value_key:str):
    """序列化有向树"""
    serialized = [root]

    # 获取当前节点的子节点
    children = list(tree.successors(root))
    for child in children:
        serialized.extend(serialize_tree(tree, child))

    # 添加'end'来标记当前节点的子节点列表结束
    serialized.append('end')

    serialized_with_value = []
    for node in serialized:
        if node != 'end':
            serialized_with_value.append(tree.nodes[node][value_key]+1)
        else:
            serialized_with_value.append(0)
    return serialized

def parse_tree_node_op_split(seq):
    n = len(seq)
    num_non_zero = sum(1 for val in seq if val != 0)
    seq_no = [0] * n  # 初始化存放编号的序列
    symbol_seq = [0] * num_non_zero  # 初始化存放符号信息的序列

    cur_no = 1  # 当前编号
    for i, val in enumerate(seq):
        if val != 0:  # 非空节点
            seq_no[i] = cur_no  # 分配编号
            symbol_seq[cur_no-1] = val-1  # 存储符号信息
            cur_no += 1  # 更新当前编号

    return seq_no, symbol_seq


def deserialize_tree(serialized):
    tree = nx.DiGraph()
    parent_queue = deque()

    iterator = iter(serialized)
    root = next(iterator)
    tree.add_node(root)
    parent_queue.append(root)

    for item in iterator:
        if item == 'end':
            parent_queue.popleft()
        else:
            parent = parent_queue[0]
            tree.add_edge(parent, item)
            parent_queue.append(item)

    return tree


def tree_seq_split(tree_seq):
    tree_without_end_flag = []
    node_index = []
    for i in range(len(tree_seq)):
        if tree_seq[i] != 0:
            tree_without_end_flag.append(1)
            node_index.append(tree_seq[i]-1)
        else:
            tree_without_end_flag.append(0)
    return tree_without_end_flag, node_index


def splitseq_to_treeseq(tree_without_end_flag,node_index):
    current_count_ind = 0
    tree_seq = []
    for i in range(len(tree_without_end_flag)):
        if tree_without_end_flag[i] == 1:
            tree_seq.append(node_index[current_count_ind]+1)
            current_count_ind += 1
        else:
            tree_seq.append(0)
    return tree_seq

def nposseq2intseq(tree_without_end_flag):
    assert tree_without_end_flag[0] == 1, 'tree_without_end_flag[0] must be 1'
    int_seq = []
    current_count = 1
    for i in range(1,len(tree_without_end_flag)):
        if tree_without_end_flag[i] == tree_without_end_flag[i-1]:
            current_count += 1
        else:
            int_seq.append(current_count)
            current_count = 1
    int_seq.append(current_count)
    return int_seq

def intseq2nposseq(int_seq):
    npos_seq = []
    for i in range(len(int_seq)):
        if i%2 ==0:
            npos_seq.extend([1]*int_seq[i])
        else:
            npos_seq.extend([0]*int_seq[i])
    return npos_seq