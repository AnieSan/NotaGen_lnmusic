import os
import re
import jellyfish
import music21 as m21
from unidecode import unidecode

def find_all_abc(directory):
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if file_path.endswith('.abc') or file_path.endswith('txt'):
                yield file_path


def extract_melody(abc):
    lines = abc.split("\n")
    melody = []

    # for line in lines:
    #     if line.startswith("V:") or line.startswith("%%"):
    #         if line=="V:2":
    #             break
    #         continue
    #     else:
    #         melody.append(line)

    # 240129处理Lieder时修改：保留V:2行以上所有内容
    for line in lines:
        if line=="V:2":
            break
        else:
            melody.append(line)
    
    return "\n".join(melody)

def has_lyrics(abc):

    lines = abc.split("\n")
    lyrics_flag = False
    no_lyrics_melody = []

    for line in lines:
        if line.startswith("w:") or line.startswith("W:"):
            lyrics_flag = True
            continue
        else:
            no_lyrics_melody.append(line)
    
    no_lyrics_melody = "\n".join(no_lyrics_melody)

    return lyrics_flag, no_lyrics_melody


def has_harmony(abc):

    score = m21.converter.parse(abc)
    harmony_flag = False
    
    for e in score.recurse():
        if isinstance(e, m21.harmony.ChordSymbol):
            harmony_flag = True
            break
    
    if harmony_flag:
        # 匹配所有括号括起来的内容
        matches_quote = re.findall(r'\".*?\"', abc)
        no_harmony_melody = abc
        # 检查第二位是否为 ^ _ < > @，如果是，则不被去掉，如果不是，则为和弦，要被去掉
        for match in matches_quote:
            if not match[1] in ['^', '_', '<', '>', '@']:
                no_harmony_melody = re.sub(match, '', no_harmony_melody)
        no_harmony_melody = re.sub(r'[ \t]+', ' ', no_harmony_melody)   # 合并空格
        no_harmony_melody = no_harmony_melody.strip()
    else:
        no_harmony_melody = abc

    return harmony_flag, no_harmony_melody

def add_control_codes(abc):
    meta_data, merged_body_data = split_abc_original(abc)
    control_codes, abc = add_tokens(meta_data, merged_body_data)

    return control_codes, abc

def strip_empty_bars(abc):
    meta_data, merged_body_data = split_abc(abc)
    if merged_body_data==None:
        return None
    else:
        body = split_bars(merged_body_data)
        left_idx = find_invalid_bars_idx(body)
        right_idx = len(body)-find_invalid_bars_idx(body[::-1])
        body = body[left_idx:right_idx]
        body = check_final_barline(body)
    
    stripped_abc = meta_data+''.join(body)
    return stripped_abc

## 麻烦检查以上的函数是否正常运行，辛苦啦！


def remove_breath(abc_body):
    pattern = r'!breath!'
    result = re.sub(pattern, '', abc_body)

    if result == abc_body:
        fermata_flag = False
    else:
        fermata_flag = True

    return fermata_flag, result


def remove_melody(abc):
    # For melodization task, remove melody and keep the chords.
    score = m21.converter.parse(abc)
    harmony_flag = False
    no_harmony_melody = []

    for e in score.recurse():
        if isinstance(e, m21.harmony.ChordSymbol):
            harmony_flag = True
            break


def check_final_barline(body):
    try:
        final_bar = body[-1]
        final_bar = replace_bracket_content(remove_wrapped_content(final_bar.replace("\n", ""))).split(" ")
        if final_bar[-1]=="|":
            body[-1] = body[-1]+"]"
    except:
        pass
    return body

def remove_wrapped_content(string):
    # 移除花括号包裹的内容
    string = re.sub(r'\{.*?\}', '', string)

    # 移除双引号包裹的内容
    string = re.sub(r'"(.*?)"', '', string)
    
    # 移除双感叹号包裹的内容
    string = re.sub(r'!(.*?)!', '', string)

    # 去除连续空格（保留换行）
    string = re.sub(r' +', ' ', string)

    return string.strip()
    
def replace_bracket_content(string, pos="top"):
    pattern = r'\[(.*?)\]'  # 匹配方括号内的内容
    matches = re.findall(pattern, string)

    for match in matches:
        if len(match) >= 2 and match[0].isupper() and match[1] == ':':
            continue  # 不需要替换，跳过当前内容
        replace_text = extract_notes(match)

        if len(replace_text) != 0:
            if pos=="top":
                replace_text = replace_text[-1]
            elif pos=="bottom":
                replace_text = replace_text[0]
        else:
            replace_text = ""

        # 将原始字符串中的方括号包裹的内容替换为处理后的内容
        string = string.replace('[' + match + ']', replace_text)

    return string

def extract_notes(input_string):
    # Regular expression pattern for single notes, rests, and decorated notes
    note_pattern = r"(x[0-9]*/*[0-9]*|z[0-9]*/*[0-9]*|[\^_=]*[A-G][,']*[0-9]*/*[0-9]*\.*|[\^_=]*[a-g][']*/*[0-9]*/*[0-9]*\.*)"
    
    # Regular expression pattern for chord notes
    chord_note_pattern = r"(?<!:)\[[^\]]*\]"
    
    # Regular expression pattern for headers
    header_pattern = r"\[[A-Za-z]:[^\]]*\]"
    
    # Regular expression pattern for decorations
    decoration_pattern = r"!.*?!"
    
    # Regular expression pattern for quoted content
    quoted_pattern = r"\".*?\""

    # Remove quoted content from input
    input_string = re.sub(quoted_pattern, '', input_string)
    
    # Remove decoration information from input
    input_string = re.sub(decoration_pattern, '', input_string)
    
    # Remove header information from input
    input_string = re.sub(header_pattern, '', input_string)
    
    # Extract notes, rests, and decorated notes using regex
    note_matches = re.findall(note_pattern, input_string)
    
    # Extract chord notes using regex
    chord_notes = re.findall(chord_note_pattern, input_string)
    
    # Combine single notes, rests, decorated notes, and chord notes
    notes = [note for note in note_matches if note.strip() != '']
    
    notes = notes + chord_notes

    return notes

def find_invalid_bars_idx(body):
    invalid_idx = 0
    rest_flag = True
    for bar in body:
        bar = replace_bracket_content(remove_wrapped_content(bar.replace("\n", ""))).replace("&", "").split(" ")
        for token in bar[:-1]:
            if token and token[0]!="z" and token[0]!="x":
                rest_flag = False
                break
        if rest_flag:
            invalid_idx += 1
        else:
            break
    return invalid_idx

def split_bars(body):
    delimiters = "|:", "::", ":|", "[|", "||", "|]", "|"
    regexPattern = '('+'|'.join(map(re.escape, delimiters))+')'
    body = "".join(body)
    bars = re.split(regexPattern, body)
    while("" in bars):
        bars.remove("")
    if bars[0] in delimiters:
        bars[1] = bars[0]+bars[1]
        bars = bars[1:]
    bars = [bars[i*2]+bars[i*2+1] for i in range(int(len(bars)/2))]

    return bars

def ld_sim(str_a, str_b):
    ld = jellyfish.levenshtein_distance(str_a, str_b)
    sim = 1-ld/(max(len(str_a), len(str_b)))
    return sim

def num_alph(line):
    num_flag = False
    alpha_flag = False
    valid_flag = False

    for char in line:
        if char.isnumeric() and alpha_flag==False and valid_flag==False:
            return True
        elif char.isalpha() and num_flag==False:
            return False
        elif char=='(' or char=='\"' or char=='!':
            valid_flag = True

def split_abc(abc):
    lines = re.split('(\n)', abc)
    if lines[-1] != '\n':
        lines.append('\n')

    lines = [lines[i*2]+lines[i*2+1] for i in range(int(len(lines)/2))]
    lines = [line.lstrip() for line in lines]
    abc = ''.join(lines)
    meta_flag = False
    meta_idx = 0

    for line in lines:
        if len(line) > 1 and line[0].isalpha() and line[1] == ':':
            meta_idx += 1
            meta_flag = True
        else:
            if meta_flag:
                break
            else:
                meta_idx += 1

    meta_data = ''.join(lines[:meta_idx])
    body_data = abc[len(meta_data):]
    
    delimiters = ":|", "||", "|]", "::", "|:", "[|"
    regexPattern = '('+'|'.join(map(re.escape, delimiters))+')'
    body_data = re.split(regexPattern, body_data)
    body_data = list(filter(lambda a: a !='', body_data))
    if len(body_data) == 0:
        return None, None
    if len(body_data)==1:
        body_data = [abc[len(meta_data):][::-1].replace('|', ']|', 1)[::-1]]
    else:
        if body_data[0] in delimiters:
            body_data[1] = body_data[0]+body_data[1]
            body_data = body_data[1:]
        if len(body_data) % 2 == 0:
            body_data = [body_data[i*2]+body_data[i*2+1] for i in range(int(len(body_data)/2))]
        else:
            body_data_temp = [body_data[i*2]+body_data[i*2+1] for i in range(int(len(body_data)/2))]
            body_data_temp.append(body_data[-1])
            body_data = body_data_temp

    merged_body_data = []
    
    for line in body_data:
        if num_alph(line):
            try:
                merged_body_data[-1] += line
            except:
                return None, None
        else:
            merged_body_data.append(line)
    
    return meta_data, merged_body_data


def split_abc_original(abc):
    lines = re.split('(\n)', abc)
    lines = [lines[i * 2] + lines[i * 2 + 1] for i in range(int(len(lines) / 2))]
    meta_flag = False
    meta_idx = 0

    for line in lines:
        if len(line) > 1 and line[0].isalpha() and line[1] == ':':
            meta_idx += 1
            meta_flag = True
        else:
            if meta_flag:
                break
            else:
                meta_idx += 1

    meta_data = ''.join(lines[:meta_idx])
    body_data = abc[len(meta_data):]

    delimiters = ":|", "||", "|]", "::", "|:", "[|"
    regexPattern = '(' + '|'.join(map(re.escape, delimiters)) + ')'
    body_data = re.split(regexPattern, body_data)
    body_data = list(filter(lambda a: a != '', body_data))
    if len(body_data) == 1:
        body_data = [abc[len(meta_data):][::-1].replace('|', ']|', 1)[::-1]]
    else:
        if body_data[0] in delimiters:
            body_data[1] = body_data[0] + body_data[1]
            body_data = body_data[1:]
        body_data = [body_data[i * 2] + body_data[i * 2 + 1] for i in range(int(len(body_data) / 2))]

    merged_body_data = []

    for line in body_data:
        if num_alph(line):
            try:
                merged_body_data[-1] += line
            except:
                return None, None
        else:
            merged_body_data.append(line)

    return meta_data, merged_body_data


def run_strip(line, delimiters):
    for delimiter in delimiters:
        line = line.strip(delimiter)
        line = line.replace(delimiter, '|')
    return line

def add_tokens(meta_data, merged_body_data):
    if merged_body_data==None:
        return "", ""
    delimiters = ":|", "||", "|]", "::", "|:", "[|"
    sec = len(merged_body_data)
    bars = []
    sims = []

    for line in merged_body_data:
        line = run_strip(line, delimiters)
        bars.append(line.count('|')+1)

    for anchor_idx in range(1, len(merged_body_data)):
        sim = []
        for compar_idx in range(anchor_idx):
            sim.append(ld_sim(merged_body_data[anchor_idx], merged_body_data[compar_idx]))
        sims.append(sim)

    header = "S:"+str(sec)+"\n"
    for i in range(len(bars)):
        if i>0:
            for j in range(len(sims[i-1])):
                header += "E:"+str(round(sims[i-1][j]*10))+"\n"
        header += "B:"+str(bars[i])+"\n"
    return unidecode(header), unidecode(meta_data+''.join(merged_body_data))

class Patchilizer:
    """
    A class for converting music bars to patches and vice versa.
    """
    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def split_bars(self, body):
        """
        Split a body of music into individual bars.
        """
        bars = re.split(self.regexPattern, ''.join(body))
        bars = list(filter(None, bars))  # remove empty strings
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        return bars

    def encode(self, abc_code):
        """
        Encode music into patches of specified length.
        """
        lines = unidecode(abc_code).split('\n')
        lines = list(filter(None, lines))  # remove empty lines

        body = ""
        patches = []

        for line in lines:
            if len(line) > 1 and ((line[0].isalpha() and line[1] == ':') or line.startswith('%%')):
                if body:
                    bars = self.split_bars(body)
                    patches.extend(bar + '\n' if idx == len(bars) - 1 else bar for idx, bar in enumerate(bars))
                    body = ""
                patches.append(line + '\n')
            else:
                body += line + '\n'

        if body:
            patches.extend(bar for bar in self.split_bars(body))

        return patches

def check_bar_patch_num_equality(abc_1, abc_2):
    # 检查两个abc经过bar patch后，小节数是否相等
    # 同时检查最后一个patch的小节线是否一致
    equality_flag = True

    abc_1 = abc_1.strip()
    abc_2 = abc_2.strip()

    patchilizer = Patchilizer()

    patches_1 = patchilizer.encode(abc_1)
    patches_2 = patchilizer.encode(abc_2)

    if len(patches_1) != len(patches_2):
        equality_flag = False

    regex_with_lookahead = "(" + "|".join(re.escape(s) for s in patchilizer.delimiters) + ")$"

    barline_1 = re.search(regex_with_lookahead, patches_1[-1]).group(0)
    barline_2 = re.search(regex_with_lookahead, patches_2[-1]).group(0)

    if barline_1 != barline_2:
        equality_flag = False

    return equality_flag


def judge_has_harmony(abc):
    # 只用于判断有无和声
    score = m21.converter.parse(abc)
    harmony_flag = False

    for e in score.recurse():
        if isinstance(e, m21.harmony.ChordSymbol):
            harmony_flag = True
            break

    return harmony_flag


def strip_harmony(abc):
    # 保证有和声的情况下，对tunebody做处理，去掉里面的和声
    matches_quote = re.findall(r'\".*?\"', abc)
    no_harmony_melody = abc
    # 检查第二位是否为 ^ _ < > @，如果是，则不被去掉，如果不是，则为和弦，要被去掉
    for match in matches_quote:
        if not match[1] in ['^', '_', '<', '>', '@']:
            no_harmony_melody = re.sub(match, '', no_harmony_melody)
    no_harmony_melody = re.sub(r'[ \t]+', ' ', no_harmony_melody)  # 合并空格
    no_harmony_melody = no_harmony_melody.strip()

    return no_harmony_melody


class Transposer:
    """
    A class for transposing abc
    """

    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'

    def split_bars(self, body):
        """
        Split a body of music into individual bars.
        """
        bars = re.split(self.regexPattern, ''.join(body))
        bars = list(filter(None, bars))  # remove empty strings
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        for j in range(len(bars)):
            # strip，去掉\n，去掉$
            bars[j] = bars[j].strip().replace('\n', '').replace('$', '')
            # 如果以数字开头，则提取数字之后的字符串，直到非数字/,/./-出现，把它加到上一个patch末尾
            if re.match(r'\d', bars[j]):
                k = 0
                for k in range(len(bars[j])):
                    if not bars[j][k] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.', '-']:
                        break
                affix = bars[j][:k]
                bars[j] = bars[j][k:].strip()
                bars[j - 1] = bars[j - 1] + affix

        return bars

transposer = Transposer()

def transpose_abc(abc_text):
    # 转置abc
    reserved_info_field = ['L:', 'K:', 'M:', 'Q:', 'V:', 'I:']

    abc_text_lines = abc_text.split('\n')

    global_M = ''
    global_L = ''
    # 滤掉除 Q:K:M:L:V: 以外的 information field
    # 滤掉除 %%score 以外的 %%行
    filtered_abc_text_lines = []
    for i, line in enumerate(abc_text_lines):
        save_state = True
        if re.search(r'^[A-Za-z]:', line) and line[:2] not in reserved_info_field:
            save_state = False
        if line.startswith("%") and not line.startswith('%%score'):
            save_state = False
        if line.startswith('M:'):
            global_M = line.strip()
        if line.startswith('L:'):
            global_L = line.strip()
        if save_state:
            filtered_abc_text_lines.append(line)

    if global_M.lstrip('M:') == 'none':
        return None

    # 分割为各个声部
    part_symbol_list = []

    tunebody_index = None
    for i, line in enumerate(filtered_abc_text_lines):
        if line.strip() == 'V:1':
            tunebody_index = i
            break
    if tunebody_index is None:
        raise Exception('tunebody index not found.')

    tunebody_lines = filtered_abc_text_lines[tunebody_index:]
    metadata_lines = filtered_abc_text_lines[:tunebody_index]
    part_text_list = []

    last_start_index = None
    for i, line in enumerate(tunebody_lines):
        if i == 0:
            last_start_index = 1
            part_symbol_list.append(line[:3])
            continue
        if line.startswith('V:'):
            last_end_index = i
            part_text_list.append(''.join(tunebody_lines[last_start_index:last_end_index]))
            part_symbol_list.append(line[:3])
            last_start_index = i + 1
    part_text_list.append(''.join(tunebody_lines[last_start_index:]))

    # 240206添加：通过bar patch检查每个声部的小节数能否对得上
    bar_equality_flag = True
    for i in range(1, len(part_text_list)):
        if not check_bar_patch_num_equality(part_text_list[0], part_text_list[i]):
            print('Warning: unequal bar number ', i)
            bar_equality_flag = False
            break
    if not bar_equality_flag:
        return

    part_patches_list = []
    for i in range(len(part_text_list)):
        part_patches = transposer.split_bars(part_text_list[i])
        part_patches = ['[' + part_symbol_list[i] + ']' + patch for patch in part_patches]
        part_patches_list.append(part_patches)

    transpose_abc_text = ''
    for j in range(len(part_patches_list[0])):
        for i in range(len(part_symbol_list)):
            transpose_abc_text = transpose_abc_text + part_patches_list[i][j].strip()
        transpose_abc_text += '\n'

    transpose_abc_text = '\n'.join(metadata_lines) + '\n' + transpose_abc_text

    return transpose_abc_text


if __name__ == '__main__':
    abc1 = """
X:1
%%score 1 ( 2 3 ) ( 4 5 ) 6 ( 7 8 ) 9 ( 10 11 ) 12 13 ( 14 15 ) ( 16 17 ) 18 19 ( 20 21 ) ( 22 23 ) ( 24 25 )
L:1/8
Q:1/4=128
M:4/4
K:C
V:1 perc nm="Percussion, D_8301"
K:none
I:percmap ^G G 64 x
I:percmap ^b b 57 x
I:percmap f f 58 normal
V:2 bass nm="Fingered Bass"
L:1/4
V:3 bass 
L:1/4
V:4 treble nm="Steel Gtr."
V:5 treble 
V:6 bass nm="Muted Guitar"
L:1/16
V:7 bass nm="Jazz Guitar"
V:8 bass 
L:1/4
V:9 treble nm="Orchestra Hit"
V:10 treble nm="Grand Piano"
V:11 treble 
V:12 treble nm="Harpsichord"
V:13 treble nm="Saw Wave"
L:1/16
V:14 perc nm="Percussion"
K:none
I:percmap A A 41 normal
I:percmap ^b b 57 x
I:percmap ^e e 46 x
I:percmap ^g g 42 x
I:percmap c c 38 normal
I:percmap d d 45 normal
I:percmap f f 82 normal
L:1/16
V:15 perc 
K:none
I:percmap F F 36 normal
V:16 treble nm="Strings 1"
V:17 treble 
V:18 treble nm="Ocarina"
V:19 treble nm="Synth Drum"
V:20 treble nm="Brightness"
V:21 treble 
L:1/4
V:22 bass nm="Grand Piano"
L:1/4
V:23 bass 
L:1/4
V:24 treble nm="Syn.Strings 1"
L:1/4
V:25 treble 
L:1/4
V:1
 z8 | z8 | z8 | z8 | z8 | z4 z2 z f | z f z f z f z2 | ff z f z ^G z f | ff f2 f2 z2 | z8 |
 z2 ff ff z f | z f z f f2 z2 | z2 ff f^G z ^G | z ^G ^G2 f2 z2 | z2 ff ff z f | z f z f ^G2 z2 |
 z2 ^G^G ^Gf z f | z f z f f2 z2 | z8 | z4 z fff | ff z2 ffff | ff z2 z fff | ff z2 ^bfff |
 ff z2 f^G z ^G | ^G2 z f z f z f | f2 z2 ff z f | f2 ff f^b f2 | f2 z2 z fff | ff z2 ffff |
 ff z2 z fff | ff z2 ^bfff | ff z2 z ^G z2 |]
V:2
[K:F] z4 | z4 | z2 C,,2 | .C,, .C,,/>A,,,/- [A,,,C,,-] C,, | z z/ D,,/ z2 | z z/ F,,/ z2 |
 z z/ F,,/ z2 | .B,,, .B,,,/>B,,,/- [B,,,C,,-] C,, | z z/ D,,/ z2 |
 .F,, .F,,/>D,,/- [D,,F,,-] F,,- | F,, z F,,2 | .F,, F,, B,,,2 | B,,, .B,,, C,,2 | .C,, z D,,2 |
 z D,, z2 | .B,,, z C,,2 | z z/ C,,/- C,, z | .B,,, z C,,2 | C,, .C,,/>A,,,/- [A,,,C,,-] C,, |
 z z/ D,,/ z2 | .F,, .F,,/>D,,/- [D,,F,,-] F,, | .F,, .F,,/>C,,/- [C,,F,,-] F,, |
 .F,, .F,,/>D,,/- [C,,-D,,] C,, | .C,, .C,,/>A,,,/- [A,,,C,,-] C,, |
 .C,, .C,,/>A,,,/- [A,,,C,,-] C,, | .C,, .C,,/>G,,,/- [G,,,C,,-] C,, |
 .C,, .C,,/>G,,,/- [G,,,F,,-] F,, | .F,, .F,,/>C,,/- [C,,F,,-] F,, |
 .F,, .F,,/>D,,/- [D,,F,,-] F,, | .F,, .F,,/>C,,/- [C,,F,,-] F,, | .F,, .F,,/>D,,/- [C,,-D,,] C,, |
 .C,, C,,/G,,,/ .A,,, z |]
V:3
[K:F] x4 | x4 | x4 | z z/ G,,,/ z2 | .C,, .C,,/>C,,/- [C,,F,,-] F,, |
 .F,, .C,,/>E,,/- [D,,-E,,] D,, | .D,, .D,,/>D,,/- [B,,,-D,,] B,,, | z z/ F,,,/ z2 |
 .C,, .C,,/>C,,/- [C,,F,,-] F,, | z z/ C,,/ z2 | x4 | x4 | z z/ B,,,/- B,,, z | z C,,2 z |
 D,, z/ C,,/- [B,,,-C,,] B,,, | z B,,,2 z | C,, .C,, B,,,2 | z B,,,2 z | z z/ G,,,/ z2 |
 C,, .C,,/>C,,/- [C,,F,,-] F,, | z z/ C,,/ z2 | z z/ D,,/ z2 | z z/ C,,/ z2 | z z/ G,,,/ z2 |
 z z/ G,,,/ z2 | z z/ A,,,/ z2 | z z/ A,,,/ z2 | z z/ D,,/ z2 | z z/ C,,/ z2 | z z/ D,,/ z2 |
 z z/ C,,/ z2 | x4 |]
V:4
[K:F] z8 | z8 | z4 z [G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][K:bass][F,CFA][F,CFA].[F,CF] |
 [F,CFA][F,CFA][F,CFA].[F,CF] [F,CFA][D,A,DF][D,A,DF].[D,A,D] |
 [D,A,DF][D,A,DF][D,A,DF].[D,A,D][K:treble] [D,A,DF][F,B,FB][F,B,FB].[F,B,F] |
 [F,B,FB][F,B,FB][F,B,FB].[F,B,F] [F,B,FB][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][F,CFA][F,CFA].[F,CF] | z2 z z/ [CFA]/- [CFA] C3- | C2 z2 z4 |
 [F,CFA][F,CFA][F,CFA].[F,CF] [F,CFA][F,B,FB][F,B,FB].[F,B,F] |
 [F,B,FB][F,B,FB][F,B,FB].[F,B,F] [F,B,FB][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][K:bass][D,A,DF][D,A,DF].[D,A,D] |
 [D,A,DF][D,A,DF][D,A,DF].[D,A,D][K:treble] [D,A,DF][F,B,FB][F,B,FB].[F,B,F] |
 [F,B,FB][F,B,FB][F,B,FB].[F,B,F] [F,B,FB][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][F,B,FB][F,B,FB].[F,B,F] |
 [F,B,FB][F,B,FB][F,B,FB].[F,B,F] [F,B,FB][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][F,CFA][F,CFA].[F,CF] |
 [F,CFA][F,CFA][F,CFA].[F,CF] [F,CFA][F,B,FB][F,B,FB].[F,B,F] |
 [F,B,FB][F,CFA][F,CFA].[F,CF] [F,CFA][F,CFA][F,CFA].[F,CF] |
 [F,CFA][F,CFA][F,CFA].[F,CF] [F,CFA][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] [G,CGc][F,CFA][F,CFA].[F,CF] |
 [F,CFA][F,CFA][F,CFA].[F,CF] [F,CFA][F,CFA][F,CFA].[F,CF] |
 [F,CFA][F,CFA][F,CFA].[F,CF] [F,CFA][F,B,FB][F,B,FB].[F,B,F] |
 [F,B,FB][F,CFA][F,CFA].[F,CF] [F,CFA][F,CFA][F,CFA].[F,CF] |
 [F,CFA][F,CFA][F,CFA].[F,CF] [F,CFA][G,CGc][G,CGc].[G,CG] |
 [G,CGc][G,CGc][G,CGc].[G,CG] .[G,CGc]2 z2 |]
V:5
[K:F] x8 | x8 | x8 | x8 | x5[K:bass] x3 | x8 | x4[K:treble] x4 | x8 | x8 |
 [F,CFA][F,CFA] [F,CFA].[F,CF]/F,/ F,2- [F,F]-[F,FA]- | [F,FA]2 z2 z [F,CFA][F,CFA].[F,CF] | x8 |
 x8 | x5[K:bass] x3 | x4[K:treble] x4 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 |
 x8 | x8 | x8 | x8 | x8 |]
V:6
[K:F] z16 | z16 | z8 z2 C,2 z .C,3 | .C,4 .C,4 z2 C,2 z .C,3 | .C,4 .C,4 z2 F,,2 z .F,,3 |
 .F,,4 .F,,4 z2 D,2 z .D,3 | .D,4 .D,4 z2 B,,2 z .B,,3 | .B,,4 .B,,4 B,,2C,2 z .C,3 |
 .C,4 .C,4 z2 F,,2 z .F,,3 | .F,,4 .F,,4 z8 | z8 z2 F,,2 z .F,,3 | .F,,4 .F,,4 z2 B,,2 z .B,,3 |
 .B,,4 .B,,4 B,,2C,2 z .C,3 | .C,4 .C,4 z2 D,2 z .D,3 | .D,4 .D,4 z2 B,,2 z .B,,3 |
 .B,,4 .B,,4 B,,2C,2 z .C,3 | .C,4 .C,4 z2 B,,2 z .B,,3 | .B,,4 .B,,4 B,,2C,2 z .C,3 |
 .C,4 .C,4 z2 C,2 z .C,3 | .C,4 .C,4 z2 F,,2 z .F,,3 | .F,,4 .F,,4 z2 F,,2 z .F,,3 |
 .F,,4 .F,,4 z2 F,,2 z .F,,3 | .F,,4 .F,,4 z2 C,2 z .C,3 | .C,4 .C,4 z2 C,2 z .C,3 |
 .C,4 .C,4 z2 C,2 z .C,3 | .C,4 .C,4 z2 C,2 z .C,3 | .C,4 .C,4 z2 F,,2 z .F,,3 |
 .F,,4 .F,,4 z2 F,,2 z .F,,3 | .F,,4 .F,,4 z2 F,,2 z .F,,3 | .F,,4 .F,,4 z2 F,,2 z .F,,3 |
 .F,,4 .F,,4 z2 C,2 z .C,3 | .C,4 .C,4 z2 .C,2 z4 |]
V:7
[K:F] z8 | z8 | .A,A, .G,>F,- [F,G,-]2 G,2 | .G,G, A,2 B,4 | .B,B,.A,A, .G,2 A,2 |
 CA, G,2 F,3 F,- | F, G,2 A,- A, B,3 | .D, D,2 F,- F,E, z A, | .A,A, z z/ F,/ F,4- | F,6 z2 | z8 |
 z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z2 z[K:treble] C- [CF]4- | [CF-]2 F2 z4 |
 z2 z B,- [B,G]4- | [B,G-]2 G2 z4 | z8 | z8 | z8 | z8 | z2 z C- [CF]4- | [CF-]2 F2 z4 |
 z4 B,G z2 |]
V:8
[K:F] x4 | x4 | z z/ G,/ z2 | x4 | x4 | x4 | x4 | x4 | z G, z2 | x4 | x4 | x4 | x4 | x4 | x4 | x4 |
 x4 | x4 | x4 | x4 | x4 | z F,3[K:treble] | x4 | z C3- | C z z2 | x4 | x4 | x4 | x4 | z F,3- |
 F, z z2 | z C2 z |]
V:9
[K:F] z8 | z8 | z2 z G z c z2 | z8 | z2 z c z F z2 | z8 | z8 | z8 | z8 | z2 z c z f z2 | z8 | z8 |
 z8 | z8 | z8 | z8 | z8 | z8 | z8 | z2 z c z f z2 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 |
 z8 | z8 | z8 |]
V:10
[K:F] z8 | z8 | z2 z g z4 | .[Bg][Bg] [ca]2 [db]4 | .[db].[db] .[ca]>[Bg]- [Bc-ga-]2 [ca]2 | z8 |
 z2 z [ca]- [ca]2 z2 | .[Bd] [Bd]2 [df]- [df] [ce]2 .[ca] | .[ca][ca] z [Af] [Af]4- | [Af]4 z4 |
 z8 | z4[K:bass] z2 F,-[F,B,]- | [F,B,F]-[F,B,FB]- [F,B,FB-f-]2 [Bf]2 z2 |
 z4[K:treble] z2 A,-[A,D]- | [A,DE]-[A,-D-EF-] [A,DF-A-]2 [FA]2 z2 | z4[K:bass] z2 G,-[G,C]- |
 [G,CD]-[G,-C-DE-] [G,CE-G-]2 [EG]2 z2 | z4[K:treble] z2 G,-[G,C]- |
 [G,CD]-[G,-C-DE-] [G,CE-G-]2 [EG]C-[C-D][CE]- | [CEG]-[CEGc]- [C-EG-c-g-]2 [CGcg]2 z2 | z8 | z8 |
 z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 |]
V:11
[K:F] x8 | x8 | .[ca][ca] .[Bg]B/[Af]/- [AB-fg-]2 [Bg]2 | x8 | z2 z [ca] z4 |
 .[ca].[ca] [Bg]2 [Af]2 z [Af]- | [Af] .[Bg]3 [db]4 | x8 | z2 [Bg]2 z4 | x8 | x8 |
 z4[K:bass] B,,4- | B,,4 z4 | z4[K:treble] D,4- | D,4 z4 | z4[K:bass] C,4- | C,4 z4 |
 z4[K:treble] C,4- | C,4 G,4- | G,4 z4 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 |
 x8 |]
V:12
[K:F] z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 |
 z8 | z4 z .A.AA | .AA z2 AB.BB | BA z2 z .AA.A | .AA z2 FAAA | AG z2 c c2 .c | c2 z B z B2 .A |
 A2 z2 A B2 B | A2 .AA GF G2 | A2 z2 z .A.AA | .AA z2 AB.BB | BA z2 z .AA.A | .AA z2 FAAA |
 AG z2 z4 |]
V:13
[K:F] z16 | z16 | z8 z4 .[CEG].[CE].[CE]2 |
 .[CEG]4 .[CG].[CE].[CE].[CE] .[CEG]4 .[CEG].[CE].[CE]2 |
 .[CEG]4 .[CEG].[CE].[CE].[CE] .[CEG]4 .[CFA].[CF].[CF]2 |
 .[CFA]4 .[CFA].[CF].[CF].[CF] .[CFA]4 .[DFA].[DF].[DF]2 |
 .[DFA]4 .[DFA].[DF].[DF].[DF] .[DFA]4 .[DFB].[DF].[DF]2 |
 .[DFB]4 .[DFB].[DF].[DF].[DF] .[DFB]4 .[CEG].[CE].[CE]2 |
 .[CEG]4 .[CEG].[CE].[CE].[CE] .[CEG]4 .[CFA].[CF].[CF]2 |
 .[CFA]4 .[CFA].[CF].[CF].[CF] .[CFA]4 z4 | z8 z4 .[CF]4 | .[CF]4 .[CF]4 .[DF]4 .[DF]4 |
 .[DF]4 .[DF]4 .[DF]4 .[CE]4 | .[CE]4 .[CE]4 .[DF]4 .[DF]4 | .[DF]4 .[DF]4 .[DF]4 .[DF]4 |
 .[DF]4 .[DF]4 .[CE]4 .[CE]4 | .[CE]4 .[CE]4 .[CE]4 .[DF]4 | .[DF]4 .[DF]4 .[CG]4 .[EG]4 |
 .[EG]4 .[EG]4 .[EG]4 .[EG]4 | .[EG]4 .[EG]4 .[EG]4 .[CF].[CF].[CF]2 |
 .[CF]4 .[CF].[CF].[CF].[CF] .[CF]4 .[DF].[DF].[DF]2 |
 .[^CDF]4 .[=CF].[CF].[CF].[CF] .[CF]4 .[FA].[CF].[CF]2 |
 .[FA]4 .[FA].[CF].[CF].[CF] .[CFA]4 .[CEG].[CE].[CE]2 |
 .[CEG]4 .[CEG].[CE].[CE].[CE] .[CEG]4 .[CEG].[CE].[CE]2 |
 .[CEG]4 .[CEG].[CE].[CE].[CE] .[CEG]4 .[CEG].[CE].[CE]2 |
 .[CG]4 .[CEG].[CE].[CE].[CE] .[CEG]4 .[CEG].[CE].[CE]2 |
 .[CEG]4 .[CEG].[CE].[CE].[CE] .[CEG]4 .[CFA].[CF].[CF]2 |
 .[CFA]4 .[CFA].[CF].[CF].[CF] .[CFA]4 .[CF].[CF].[CF]2 |
 .[CF]4 .[CF].[CF].[CF].[CF] .[CF]4 .[DF].[DF].[DF]2 |
 .[^CDF]4 .[=CF].[CF].[CF].[CF] .[CF]4 .[CFA].[CF].[CF]2 |
 .[CFA]4 .[CFA].[CF].[CF].[CF] .[CFA]4 .[CEG].[CE].[CE]2 |
 .[CEG]4 .[CEG].[CE].[CE].[CE] .[CEG]4 z4 |]
V:14
 z8 z2 ^g2 z2 ^g2 | z2 ^g2 z2 ^g2 z2 ^g2 z2 ^g2 |
 f2[f^g]2[cd]2[Ac^g]2 [Ac^e]2[f^g^b]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g^b]2 z4 |
 z8 z2 [f^g^b]2[ff^g]2[ff^g]2 | [ff^g]2[f^g]2[ff^g]2[ff^g]2 [ff^g]2[f^g]2[ff^g]2[ff^g]2 |
 [ff^g]2[f^g]2[ff^g]2[ff^g]2 [ff^g]2[f^g]2[ff^g]2[ff^g]2 |
 [ff^g]2[f^g]2[ff^g]2[ff^g]2 [ff^g]2[f^g]2[ff^g]2[ff^g]2 |
 [ff^g]2[f^g]2[ff^g]2[ff^g]2 [ff^g]2[f^g]2[ff^g]2[ff^g]2 |
 [ff^g]2[f^g]2[ff^g]2[ff^g]2 [ff^g]2[f^g]2[ff^g]2[ff^g]2 |
 [ff^g]2[f^g]2[ff^g]2[ff^g]2 [ff^g]2[f^g]2[ff^g]2[ff^g]2 |
 [ff^g]2[f^g]2[ff^g]2[ff^g]2 [ff^g]2[f^g]2[ff^g]2[ff^g]2 |
 [ff^g]2[f^g]2[ff^g]2[ff^g]2 [ff^g]2[f^g]2[ff^g]2[cff^g]2 |
 [fff^g]2[ff^g]2[dff^g]2[Aff^g]2 [cff^g]c[f^g^b]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f/8c3/4[cff^g]2 [c^eff]3/4 z/8 ^g/8c[f^g^b]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8 z [f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 |
 [^eff]^g[f^g]2 [^eff]3/4 z/8 ^g/8 z/8 f3/8 z/ [cff^g]2 [^eff]3/4 z/8 ^g/8-^g3 z4 |]
V:15
 x8 | x8 | z4 z F z F | z F z F z F z F | z F z F z F z F | z F z F z F z F | z F z F z F z F |
 z F z F z F z F | z F z F z F z F | z F z F z F z2 | z4 z F z F | z F z F z F z F |
 z F z F z F z F | z F z F z F z F | z F z F z F z F | z F z F z F z F | z F z F z F z F |
 z F z F z F z F | z F z F z F z F | z F z F z F z F | z F z F z F z F | z F z F z F z F |
 z F z F z F z F | z F z F z F z F | z F z F z F z F | z F z F z F z F | z F z F z F z F |
 z F z F z F z F | z F z F z F z F | z F z F z F z F | z F z F z F z F | z F z F z4 |]
V:16
[K:F] z8 | z8 | .[ac'][ac'].[gb]g [gb]4 | .[gb][gb] [ac']2 [bd']4 |
 .[bd'][bd'] .[ac']>[gb]- [ga-bc'-]2 [ac']2 | z8 | z2 z [ac']- [ac']2 z2 |
 [db] [db]2 [fd']- [fd'] [ec']2 .[ac'] | .[ac'][ac'] z [fa] [fa]4- | [fa]6 z2 | z8 | z8 | z8 | z8 |
 z8 | z8 | z8 | z8 | z8 | z8 | z4 [db]4 | .[^cb]2 z2 z4 | z8 | [Bg]4- [Be-gc'-]2 [ec']2- |
 [ec']4 z2 [ca]2- | [ca]4- [cd-ab-]2 [db]2 | [ca]4 [Bg][Af] [Bg]2 | z8 | z4 z b3- | b2 z2 z4 |
 z4 [ca]4- | [ca]2 z2 z4 |]
V:17
[K:F] x8 | x8 | z2 z b/[fa]/- [fa]2 z2 | x8 | z2 z [ac'] z4 | .[ac'][ac'] [gb]2 [fa]2 z [fa]- |
 [fa] .[gb]3 [bd']4 | x8 | z2 [gb]2 z4 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 | x8 |
 z4 [ca]4- | [ca]6 z2 | z2 [=ca]6- | [ca]8- | [ca]2 z2 z4 | z4 [db]4 | x8 | x8 | [ca]4 [ca]4- |
 [ca]4- [ca] d3 | .^c2 [=ca]6- | [ca]4 z4 | [Bg]4- [Bg].[cc'] z2 |]
V:18
[K:F] z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z4 z FCF | GFCF GFCF | GFCF GF z F | z FCF GF z F |
 z F z F z E z E | z ECE FF z F | z F z F z D z D | z DCD EE z E | z E z E z F z F | z FDF GG z G |
 z GCG AG z G | z G z G z4 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 |]
V:19
[K:F] z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 |
 z4 z2 .c/4>.c/4 z/4 .c/4.c | .G.D[K:bass].A,.D, .A,,2 z2 | z8 | z8 | z8 | z8 | z8 | z8 | z8 | z8 |
 z8 | z8 | z8 | z8 |]
V:20
[K:F] z8 | z8 | .[Ac].[Ac].[GB].[GB] .[FA]2 .[GB]2 | .[GB].[GB] .[Ac]2 [Bd]4 |
 .[Bd].[Bd].[Ac].[Ac] [Ac]4 | .[Ac].[Ac] .[GB]2 [FA]2 z [FA]- | [FA][GB] z [Ac] z .[Bd]3 |
 .[DB] [DB]2 [Fd]- [Fd][Ec] z .[Ac] | .[Ac].[Ac] .[GB]2 [FA]4- | [FA]2 z2 z4 | z8 | z8 | z8 | z8 |
 z8 | z8 | z8 | z8 | z8 | z4 z .[CA].[CA].[CA] | .[CA][CA] z2 .[CA].[DB].[DB].[DB] |
 .[^CB][=CA] z2 z .[CA].[CA].[CA] | .[CA][CA] z2 .[A,F].[CA].[CA].[CA] |
 .[CA][B,G] z2 .[Cc] [Ec]2 .[Ec] | [Ec]2 z [DB] z [DB]2 .[CA] | [CA]2 z2 .[CA] [DB]2 [DB] |
 .[CA]2 .[CA].[CA] .[B,G][A,F] [B,G]2 | [CA]2 z2 z .[CA].[CA].[CA] |
 .[CA][CA] z2 .[CA].[DB].[DB].[DB] | .[^CB][=CA] z2 z .[CA].[CA].[CA] |
 .[CA][CA] z2 .[A,F].[CA].[CA].[CA] | .[CA][B,G] z2 z4 |]
V:21
[K:F] x4 | x4 | x4 | x4 | z z/ z/4 [GB]/4- [GB] z | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 |
 x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 |]
V:22
[K:F] z4 | z4 | C z/ A,/- [A,B,-] B,- | B,[K:treble] C D2- | D z/ B,/- [B,C-] C- |
 C[K:bass] B, z2 | B,2[K:treble] C/ D3/2 | z D- [C-D] C- | C B, A,2- | A,3 z | z4 | z4 | z4 | z4 |
 z4 | z4 | z4 | z4 | z4 | z4 | z2 B2- | B z z2 | z F2 z | G2- [Gc-] c- | c2 z A- | A2- A/ B3/2 |
 A2 G/F/- F/<G/ | A4- | A3 z | A4- | A2 A2- | A z z2 |]
V:23
[K:F] x4 | x4 | z B, z2 | x[K:treble] x3 | z C z2 | z2[K:bass] A,2- | A, z[K:treble] z2 | B,2 z2 |
 x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | z2 A2- | A3 z | A4- | A2 A2- | A z z2 |
 z2 B2 | x4 | x4 | x4 | z2 B2- | B z z2 | z F2 z | G3 z |]
V:24
[K:F] z4 | z4 | z2 C2- | C4- | C2 C2- | C3 z | z2 [B,DF]2- | [B,DF]3 z | z4 | z4 | z2 [A,C]2- |
 [A,C]3 z | z2 [G,CE]2- | [G,CE]2- [G,A,-CEF-] [A,F]- | [A,F]3[K:bass] z | z2[K:treble] [G,CE]2- |
 [G,CE]3 z | z2 [CEG]2- | [CEG]4- | [CE-G-]2 [EF-G] F- | F2 [B,-D-F]2 | [A,-B,DF-] [A,F]3- |
 [A,F]2- [G,-A,E-F] [G,E]- | [G,E]4- | [G,E]4- | [G,E]4- | [G,E]2- [G,EF-] F- | F4- |
 F2 [B,-D-F]2 | [A,-B,DF-] [A,F]3- | [A,F]2- [G,-A,C-E-F] [G,CE]- | [G,CE]3 z |]
V:25
[K:F] x4 | x4 | z2 [G,E]2- | [G,E]4- | [G,E]2- [G,A,-EF-] [A,F]- | [A,F]2 [A,DF]2- |
 [A,-DF]2 A, z | z2 [G,CE]2- | [G,-CE-]2 [G,A,-C-EF-] [A,CF]- | [A,CF]3 z | z2 F2- | F2 [B,DF]2- |
 [B,DF]3 z | z2 D2- | D2[K:bass] [F,B,D]2- | [F,B,D]3[K:treble] z | z2 [B,DF]2- | [B,DF]3 z | x4 |
 z2 [A,C]2- | [A,C]3 z | C4- | C2 C2- | C4- | C4- | C4- | C2 [A,C]2- | [A,C]4- | [A,C]3 z | C4- |
 C2 z2 | x4 |]

    """

    abc = transpose_abc(abc1)
    print(abc)