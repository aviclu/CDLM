def parse_key_file(key_file):
    try:
        from nltk.parse.stanford import StanfordParser
        parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz",
                                java_options='-Xmx8G')
        print("Starting to parse key_file!")
        print("This might take a while...")
        new_file = open(key_file + ".parsed", "w")
        with open(key_file) as f:
            tmp_sentence = [[]]
            tmp_conll_lines = []
            for line in f:
                if line.startswith("#begin"):
                    new_file.write(line)
                    continue
                elif len(line.strip()) == 0 or (line.startswith("#end") and len(tmp_conll_lines) > 0):
                    parse = parser.parse_sents(tmp_sentence)
                    for tree in parse:
                        for tree_line in tree:  # line is a Tree
                            parse_string = ' '.join(str(tree_line).split())
                            treecomp = parse_string.split()
                            currlowestindex = 0
                            token_index = 0
                            for idx, val in enumerate(treecomp):
                                if not val.startswith("("):
                                    firstindexofbracket = val.index(")")
                                    lastindex = val.__len__() - 1
                                    tag_components = []
                                    pos_tag = treecomp[idx - 1].replace("(", "")
                                    if currlowestindex == idx - 1:
                                        if firstindexofbracket == lastindex:
                                            tag_components.append("*")
                                        else:
                                            parsecol = "*" + val[firstindexofbracket:lastindex]
                                            tag_components.append(parsecol)
                                    else:
                                        for i in range(currlowestindex, idx - 1):
                                            tag_components.append(treecomp[i])
                                        if firstindexofbracket == lastindex:
                                            tag_components.append("*")
                                        else:
                                            parsecol = "*" + val[firstindexofbracket:lastindex]
                                            tag_components.append(parsecol)
                                    currlowestindex = idx + 1

                                    new_file.write('\t'.join(
                                        tmp_conll_lines[token_index].split()[0:4]) + "\t" + pos_tag + "\t" + ''.join(
                                        tag_components) + '\t' + '\t'.join(
                                        tmp_conll_lines[token_index].split()[4:]) + '\n')
                                    token_index += 1

                    tmp_sentence[0] = []
                    tmp_conll_lines = []
                    new_file.write("\n")

                elif not line.startswith("#"):
                    word = line.split()[3]
                    word_uc = word  # .decode(encoding='UTF-8')
                    tmp_sentence[0].append(word_uc)
                    tmp_conll_lines.append(line)
                if line.startswith("#end"):
                    new_file.write(line)


    except:
        print("You need to set the CLASSPATH environment variable to point to the Stanford parser!")
        print(
            "Example: export CLASSPATH=/path/to/stanford-parser-full-YYYY-MM-DD/stanford-parser.jar:/path/to/stanford-parser-full-YYYY-MM-DD/stanford-parser-X.X.X-models.jar")
        print("")
        raise


def check_gold_parse_annotation(key_file):
    has_gold_parse = False
    with open(key_file) as f:
        for line in f:
            if not line.startswith("#"):
                if len(line.split()) > 6:
                    parse_col = line.split()[5]
                    if not parse_col == "-":
                        has_gold_parse = True
                        break
                    else:
                        break
    return has_gold_parse