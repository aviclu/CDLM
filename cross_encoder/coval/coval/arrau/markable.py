class Markable:
    def __init__(self, doc_name, start, end, MIN, is_referring, words):
        self.doc_name = doc_name
        self.start = start
        self.end = end
        self.MIN = MIN
        self.is_referring = is_referring
        self.words = words

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # MIN is only set for the key markables
            if self.MIN:
                return (self.doc_name == other.doc_name
                        and other.start >= self.start
                        and other.start <= self.MIN[0]
                        and other.end <= self.end
                        and other.end >= self.MIN[1])
            elif other.MIN:
                return (self.doc_name == other.doc_name
                        and self.start >= other.start
                        and self.start <= other.MIN[0]
                        and self.end <= other.end
                        and self.end >= other.MIN[1])
            else:
                return (self.doc_name == other.doc_name
                        and self.start == other.start
                        and self.end == other.end)
        return NotImplemented

    def __neq__(self, other):
        if isinstance(other, self.__class__):
            return self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(frozenset((self.start, self.end)))

    def __str__(self):
        return ('DOC: %s SPAN: (%d, %d) String: %r MIN: %s Referring tag: %s'
                % (
                    self.doc_name, self.start, self.end, ' '.join(self.words),
                    '(%d, %d)' % self.MIN if self.MIN else '',
                    self.is_referring))