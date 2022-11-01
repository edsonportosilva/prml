# This script was originally writen by jdmcpee (https://github.com/jdmcpeek/pretty-print-binary-tree)
# I rewrite some parts for bug fixing and ported it to Python 3.
#
# For a detailed explanation please read the Readme.
# 
# For an example see the test_nodes.py.
#
# 13.02.2019 by therealpeterpython 
# Script retrived from
#   https://github.com/therealpeterpython/pretty-print-binary-tree

from __future__ import print_function   # necessary for py2/3 compatibility

try:    # necessary for py2/3 compatibility
    from queue import Queue
except:
    from Queue import Queue

from copy import deepcopy as deepcopy
import sys

# 'data_attr_name' is the name of the data attribute
# 'left_attr_name' is the name of the left sub node attribute 
# 'right_attr_name' is the name of the right sub node attribute
#
def pprint(external_node, data_attr_name = "data", left_attr_name = "left", right_attr_name = "right"):
    internal_node = convert_tree(external_node, data_attr_name, left_attr_name, right_attr_name)
    internal_node.prettyPrint()
    
def convert_tree(external_node, data_attr_name = "data", left_attr_name = "left", right_attr_name = "right"):
    if not external_node:
        return None
    node_data = getattr(external_node, data_attr_name)
    node_left = getattr(external_node, left_attr_name)
    node_right = getattr(external_node, right_attr_name)
    return Node(
        node_data,
        convert_tree(
            node_left, data_attr_name, left_attr_name, right_attr_name
        ),
        convert_tree(
            node_right, data_attr_name, left_attr_name, right_attr_name
        ),
    )

class Node:
    def __init__(self, data = None, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right

    def getHeight(self):
        return Node.getHeightHelper(self)

    @staticmethod
    def getHeightHelper(node):
        return (
            max(Node.getHeightHelper(node.left), Node.getHeightHelper(node.right))
            + 1
            if node
            else 0
        )

    def fillTree(self, height):
        Node.fillTreeHelper(self, height)

    def fillTreeHelper(self, height):
        if height <= 1:
            return
        if self:
            if not self.left:
                self.left = Node(' ')
            if not self.right:
                self.right = Node(' ')
            Node.fillTreeHelper(self.left, height - 1)
            Node.fillTreeHelper(self.right, height - 1)


    def prettyPrint(self):
        """
        """
        # get height of tree
        total_layers = self.getHeight()

        tree = deepcopy(self)

        tree.fillTree(total_layers)
        # start a queue for BFS
        queue = Queue()
        # add root to queue
        queue.put(tree) # self = root
        # index for 'generation' or 'layer' of tree
        gen = 1
        # BFS main
        extra_spaces_next = 1
        init_padding = 2
        while not queue.empty():
            # copy queue
            # 
            copy = Queue()
            while not queue.empty():
                copy.put(queue.get())
            # 
            # end copy queue 

            first_item_in_layer = True
            edges_string = ""
            extra_spaces_next_node = False

            # modified BFS, layer by layer (gen by gen)
            while not copy.empty():

                node = copy.get()

                # -----------------------------
                # init spacing
                spaces_front = pow(2, total_layers - gen + 1) - 2
                spaces_mid   = pow(2, total_layers - gen + 2) - 2
                dash_count   = pow(2, total_layers - gen) - 2
                dash_count = max(dash_count, 0)
                spaces_mid = spaces_mid - (dash_count*2)
                spaces_front = spaces_front - dash_count
                spaces_front += init_padding
                if first_item_in_layer:
                    edges_string += " " * init_padding
                # ----------------------------->

                # -----------------------------
                # construct edges layer
                edge_sym = "/" if node.left and node.left.data is not " " else " "
                if first_item_in_layer:
                    edges_string += " " * (pow(2, total_layers - gen) - 1) + edge_sym
                else:
                    edges_string += " " * (pow(2, total_layers - gen + 1) + 1) + edge_sym
                edge_sym = "\\" if node.right and node.right.data is not " " else " "
                edges_string += " " * (pow(2, total_layers - gen + 1) - 3) + edge_sym
                # ----------------------------->

                # -----------------------------
                # conditions for dashes
                dash_left = " " if node.left and node.left.data == " " else "_"
                dash_right = " " if node.right and node.right.data == " " else "_"
                # ----------------------------->

                # -----------------------------
                # handle condition for extra spaces when node lengths don't match or are even:
                if extra_spaces_next_node:
                    extra_spaces = extra_spaces_next
                    extra_spaces_next_node = False
                else:
                    extra_spaces = 0
                # ----------------------------->

                # -----------------------------
                # account for longer data
                data_length = len(str(node.data))
                if data_length > 1:
                    if data_length % 2 == 1: # odd
                        if dash_count > 0:
                            dash_count -= int((data_length - 1)/2)
                        else:
                            spaces_mid -= int((data_length - 1)/2)
                            spaces_front -= int((data_length - 1)/2)
                            if data_length is not 1:
                                extra_spaces_next = int((data_length - 1)/2)
                                extra_spaces_next_node = True
                    else: # even
                        if dash_count > 0:
                            dash_count -= data_length // 2 - 1
                                                #dash_count += 1
                        else:
                            spaces_mid -= data_length // 2
                            spaces_front -= data_length // 2
                        extra_spaces_next_node = True
                        extra_spaces_next = data_length // 2 - 1
                # ----------------------------->

                # -----------------------------
                # print node with/without dashes
                if first_item_in_layer:
                    print(" " * spaces_front + dash_left * dash_count + str(node.data) + dash_right * dash_count, end=" ")
                    first_item_in_layer = False
                else:
                    print(" " * (spaces_mid-extra_spaces) + dash_left * dash_count + str(node.data) + dash_right * dash_count, end=" ")
                # ----------------------------->

                if node.left: queue.put(node.left)
                if node.right: queue.put(node.right)

            if not queue.empty():
                    print("\n" + edges_string)

            # increase layer index
            gen += 1

