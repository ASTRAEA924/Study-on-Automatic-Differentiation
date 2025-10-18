from math import log as math_log
from math import sin as math_sin
from math import cos as math_cos


class Node:
    verbose = False
    input_count = 0
    intermediate_count = 0

    def __init__(self, value, parent_nodes=[], operator="input"):
        self.value = value
        self.parent_nodes = parent_nodes
        self.child_nodes = []
        self.operator = operator
        self.grad_wrt_parents = []
        self.partial_derivative = 0  
        self.gradient = 0            # For reverse mode 

        if self.operator == "input":
            Node.input_count += 1
            self.name = f"x{Node.input_count}"
        else:
            Node.intermediate_count += 1
            self.name = f"v{Node.intermediate_count}"

        if Node.verbose:
            print(f"{self.name:<3}= {self.operator:<5}{[p.name for p in self.parent_nodes]} = {self.value:.4f}")


def add(node1, node2):
    value = node1.value + node2.value
    new = Node(value, [node1, node2], "add")
    new.grad_wrt_parents = [1, 1]
    node1.child_nodes.append(new)
    node2.child_nodes.append(new)
    return new


def sub(node1, node2):
    value = node1.value - node2.value
    new = Node(value, [node1, node2], "sub")
    new.grad_wrt_parents = [1, -1]
    node1.child_nodes.append(new)
    node2.child_nodes.append(new)
    return new


def mul(node1, node2):
    value = node1.value * node2.value
    new = Node(value, [node1, node2], "mul")
    new.grad_wrt_parents = [node2.value, node1.value]
    node1.child_nodes.append(new)
    node2.child_nodes.append(new)
    return new


def log(node):
    value = math_log(node.value)
    new = Node(value, [node], "log")
    new.grad_wrt_parents = [1 / node.value]
    node.child_nodes.append(new)
    return new


def sin(node):
    value = math_sin(node.value)
    new = Node(value, [node], "sin")
    new.grad_wrt_parents = [math_cos(node.value)]
    node.child_nodes.append(new)
    return new


def topological_order(rootNode):
    visited = set()
    order = []

    def visit(node):
        if node not in visited:
            visited.add(node)
            for p in node.parent_nodes:
                visit(p)
            order.append(node)

    visit(rootNode)
    return order

def forward(rootNode):
    rootNode.partial_derivative = 1
    ordering = topological_order(rootNode)
    for node in ordering[1:]:
        partial_derivative = 0
        for i in range(len(node.parent_nodes)):
            dnode_dparent = node.grad_wrt_parents[i]
            dparent_droot = node.parent_nodes[i].partial_derivative
            partial_derivative += dnode_dparent * dparent_droot
        node.partial_derivative = partial_derivative

        if Node.verbose == True:
            symbol_process = ""
            value_process = ""
            for i in range(len(node.parent_nodes)):
                dnode_dparent = node.grad_wrt_parents[i]
                symbol_process += "(d" + node.name + "/d" + node.parent_nodes[i].name + ")"\
                                  + "(d" + node.parent_nodes[i].name + "/d" + rootNode.name + ") + "
                value_process += "(" + str(dnode_dparent.__round__(3)) + ")(" + \
                    str(node.parent_nodes[i].partial_derivative.__round__(
                        3)) + ") + "
            print('d{:<2}/d{:<2} = {:<45} \n\t= {:<30} = {:<5}'.format(
                node.name,
                rootNode.name,
                symbol_process.strip(" + "),
                value_process.strip(" + "),
                str(node.partial_derivative.__round__(3)))
            )

def reverse(rootNode):
    # 建立拓撲排序
    order = topological_order(rootNode)

    rootNode.gradient = 1  # dy(root)/dy = 1
    # 反轉（由輸出往輸入）逐一傳遞
    for node in reversed(order):
        for i, parent in enumerate(node.parent_nodes):
            old_gradient = parent.gradient
            #根據chain rule更新parent的gradient
            parent.gradient += node.gradient * node.grad_wrt_parents[i]

            #print出執行狀況
            if Node.verbose:
                print(f"d{rootNode.name}/d{parent.name} += "
                      f"(d{rootNode.name}/d{node.name}) * (d{node.name}/d{parent.name}) "
                      f"= {old_gradient:.3f} + {node.gradient:.3f} * {node.grad_wrt_parents[i]:.3f} "
                      f"-> {parent.gradient:.3f}")

