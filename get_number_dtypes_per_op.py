import os, re

def get_strings_from_parens_one_level(input_text):
    strings = []
    startIdx = 0
    endIdx = len(input_text) - 1
    inString = False
    for i in range(len(input_text)):
        char = input_text[i]
        if char == '(' and (not inString):
            startIdx = i+1
            inString = True
        if char == ')' and inString:
            endIdx = i+1
            inString = False
            if endIdx >= len(input_text):
                strings.append(input_text[startIdx:endIdx-1])
            else:
                strings.append(input_text[startIdx:endIdx])
    return strings

def make_ops_dict(input_text):
    dtype_counts = dict()
    #print("input_text: " + str(input_text))
    for op_string in input_text:
        op_name = re.sub(r'\([^)]*\)', '', op_string).replace("&&", "").strip()
        #print("op_name: " + str(op_name))
        dtypes = get_strings_from_parens_one_level(op_string)
        #print("dtypes: " + str(dtypes))
        assert(len(dtypes) == 1)
        if "||" in dtypes[0]:
            num_dtypes = len(dtypes[0].split("||"))
            dtype_counts[op_name] = num_dtypes
        else:
            dtype_counts[op_name] = 1
    return dtype_counts

def collate_dtype_counts(ops_dict):
    one_dtype_count = 0
    two_dtype_count = 0
    three_plus_dtype_count = 0
    for k,v in ops_dict.items():
        if v == 1:
            one_dtype_count += 1
        elif v == 2:
            two_dtype_count += 1
        else:
            assert(v >= 3)
            three_plus_dtype_count += 0
    resultStr = str(one_dtype_count)
    resultStr += "," + str(two_dtype_count)
    resultStr += "," + str(three_plus_dtype_count)
    return resultStr

        
with open("cmake-out/examples/dtype_selective_build/select_build_lib/selected_op_variants.h") as headerFile:
    text = headerFile.read()
    pattern = r'/\*.*?\*/'
    no_comment_text = re.sub(pattern, '', text, flags=re.DOTALL)
    extracted_content = re.findall(r'\{([^}]+)\}', no_comment_text)
    text = extracted_content[0].replace("(std::string_view(operator_name).compare(\"", "")
    text = text.replace("\") == 0)", "")
    text = text.replace("scalar_type == executorch::aten::ScalarType::","")
    text = text.replace("return","").replace(";","").strip()
    #print("input_text:" + str(text))
    ops = get_strings_from_parens_one_level(text)
    dtypes_dict = make_ops_dict(ops)
    print(collate_dtype_counts(dtypes_dict))

