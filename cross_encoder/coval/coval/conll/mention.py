import hashlib


class Mention:
    def __init__(self, doc_name, sent_num, start, end, words):
        self.doc_name = doc_name
        self.sent_num = sent_num
        self.start = start
        self.end = end
        self.words = words 
        self.gold_parse_is_set = False
        self.gold_parse = None
        self.min_spans = set()
        
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.min_spans:
                return self.doc_name == other.doc_name and self.sent_num == other.sent_num \
                       and self.min_spans==other.min_spans
            else:
                return self.doc_name == other.doc_name and self.sent_num == other.sent_num \
                   and self.start == other.start and self.end == other.end 
        return NotImplemented
    
    def __neq__(self, other):
        if isinstance(other, self.__class__):
            return self.__eq__(other)
        
        return NotImplemented
    
    def __str__(self):
        return str("DOC: " +self.doc_name+ ", sentence number: " + str(self.sent_num) 
                   + ", ("+str(self.start)+", " + str(self.end)+")" +
                   (str(self.gold_parse) if self.gold_parse else "") + ' ' + ' '.join(self.words))

    def __hash__(self):
        if self.min_spans:
            return self.sent_num * 1000000 + hash(frozenset(self.min_spans))
        else:
            return self.sent_num * 1000000 + hash(frozenset((self.start, self.end)))

    def get_span(self):
        if self.min_spans:
            ordered_words=[e[0] for e in sorted(self.min_spans, key=lambda e: e[1])]
            return ' '.join(ordered_words)
        else:
            return ' '.join([w[1] for w in self.words])
         
            
    def set_gold_parse(self, tree):
        self.gold_parse = tree
        self.gold_parse_is_set = True

    def are_nested(self, other):
        if isinstance(other, self.__class__):
            if self.__eq__(other):
                return -1
            if True:
                #self is nested in other
                if self.sent_num == other.sent_num and \
                   self.start >= other.start and self.end <= other.end:
                    return 0
                #other is nested in self
                elif self.sent_num == other.sent_num and \
                   other.start >= self.start and other.end <= self.end:
                    return 1
                else:
                    return -1
       
        return NotImplemented


    '''
    This function is for specific cases in which the nodes 
    in the top two level of the mention parse tree do not contain a valid tag.
    E.g., (TOP (S (NP (NP one)(PP of (NP my friends)))))
    '''
    def get_min_span_no_valid_tag(self, root):
        if not root:
            return
        
        terminal_shortest_depth = float('inf')
        queue = [(root, 0)]

        accepted_tags = None
    
        while queue:
            node, depth = queue.pop(0)

            if not accepted_tags:
                if node.tag[0:2] in ['NP', 'NM']:
                    accepted_tags=['NP', 'NM', 'QP', 'NX']
                elif node.tag[0:2]=='VP':
                    accepted_tags=['VP']

            if node.isTerminal and depth <= terminal_shortest_depth:
                if self.is_a_valid_terminal_node(node.tag, node.pos):
                    self.min_spans.add((node.tag, node.index))
                    terminal_shortest_depth = min(terminal_shortest_depth, depth)
                    
            elif (not self.min_spans or depth < terminal_shortest_depth )and node.children and \
                 (depth== 0 or not accepted_tags or node.tag[0:2] in accepted_tags): 
                for child in node.children:
                    if not child.isTerminal or (accepted_tags and node.tag[0:2] in accepted_tags):
                        queue.append((child, depth+1))    


    """
    Exluding terminals like comma and paranthesis
    """
    def is_a_valid_terminal_node(self, tag, pos):
        if len(tag.split()) == 1:
            if (any(c.isalpha() for c in tag) or \
                any(c.isdigit() for c in tag) or tag == '%') \
                  and (tag != '-LRB-' and tag != '-RRB-') \
                  and pos[0] != 'CC' and pos[0] != 'DT' and pos[0] != 'IN':# not in conjunctions:
                return True
            return False
        else: # for exceptions like ", and"
            for i, tt in enumerate(tag.split()):
                if self.is_a_valid_terminal_node(tt, [pos[i]]):
                    return True
            return False
   

    def get_valid_node_min_span(self, root, valid_tags, min_spans):
        if not root:
            return

        terminal_shortest_depth = float('inf')
        queue = [(root, 0)]

        while queue:
            node, depth = queue.pop(0)

            if node.isTerminal and depth <= terminal_shortest_depth:
                if self.is_a_valid_terminal_node(node.tag, node.pos):
                    min_spans.add((node.tag, node.index))
                    terminal_shortest_depth = min(terminal_shortest_depth, depth)

            elif (not min_spans or depth < terminal_shortest_depth )and node.children and \
                 (depth== 0 or not valid_tags or node.tag[0:2] in valid_tags):
                for child in node.children:
                    if not child.isTerminal or (valid_tags and node.tag[0:2] in valid_tags):
                        queue.append((child, depth+1))


    def get_top_level_phrases(self, root, valid_tags):
        terminal_shortest_depth = float('inf')
        top_level_valid_phrases = []
        min_spans = set()
        
        if root and root.isTerminal and self.is_a_valid_terminal_node(root.tag, root.pos):
            self.min_spans.add((root.tag, root.index))

        elif root and root.children:
            for node in root.children:
                if node:
                    if node.isTerminal and self.is_a_valid_terminal_node(node.tag, node.pos):
                        self.min_spans.add((node.tag, node.index))
            if not self.min_spans:           
                for node in root.children:
                    if node.children and node.tag[0:2] in valid_tags:
                        top_level_valid_phrases.append(node)

        return top_level_valid_phrases

    def get_valid_tags(self, root):
        valid_tags = None
        NP_tags = ['NP', 'NM', 'QP', 'NX']
        VP_tags = ['VP']

        if root.tag[0:2]=='VP':
            valid_tags = VP_tags
        elif root.tag[0:2] in ['NP', 'NM']:
            valid_tags = NP_tags
        else:
            if root.children: ## If none of the first level nodes are either NP or VP, examines their children for valid mention tags
                all_tags = []
                for node in root.children:
                    all_tags.append(node.tag[0:2])
                if 'NP' in all_tags or 'NM' in all_tags:
                    valid_tags = NP_tags
                elif 'VP' in all_tags:
                    valid_tags = VP_tags
                else:
                    valid_tags = NP_tags

        return valid_tags


    def set_min_span(self):

        if not self.gold_parse_is_set:
            print('The parse tree should be set before extracting minimum spans')
            return NotImplemented

        root = self.gold_parse

        if not root:
            return

        terminal_shortest_depth = float('inf')
        queue = [(root, 0)]

        valid_tags = self.get_valid_tags(root)


        top_level_valid_phrases = self.get_top_level_phrases(root, valid_tags)
        
        if self.min_spans:
            return
        '''
        In structures like conjunctions the minimum span is determined independently
        for each of the top-level NPs
        '''
        if top_level_valid_phrases:
            for node in top_level_valid_phrases:
                self.get_valid_node_min_span(node, valid_tags, self.min_spans) 

        else:
            self.get_min_span_no_valid_tag(root)


        """
        If there was no valid minimum span due to parsing errors return the whole span
        """
        if len(self.min_spans)==0:
            self.min_spans.update([(word, index) for index, word in enumerate(self.words)])


    
class TreeNode:
    def __init__(self, tag, pos, index, isTerminal):
        self.tag = tag
        self.pos = pos
        self.index = index
        self.isTerminal = isTerminal
        self.children = []
        
    def __str__(self, level=0):
        ret = "\t"*level+(self.tag)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def get_terminals(self, terminals):
        if self.isTerminal:
            terminals.append(self.tag)
        else:
            for child in self.children:
                child.get_terminals(terminals)

    def refined_get_children(self):    
        children = []
        for child in self.children:
            if not child.isTerminal and child.children and len(child.children)==1 and child.children[0].isTerminal:
                children.append(child.children[0])
            else:
                children.append(child)
        return children

            
