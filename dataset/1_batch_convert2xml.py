import os
import math
import json
import subprocess
from tqdm import trange
from multiprocessing import Pool

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

cmd = 'cd '+os.getcwd()
output = os.popen(cmd).read()
# cmd = 'cmd /u /c python xml2abc.py -m 2 -c 6 -x '
cmd = 'cmd /u /c python abc2xml.py '

def run_filter(lines):
    score = ""
    for line in lines.replace("\r", "").split("\n"):
        if line[:2] in ['A:', 'B:', 'C:', 'D:', 'F:', 'G', 'H:', 'N:', 'O:', 'R:', 'r:', 'S:', 'T:', 'W:', 'w:', 'X:', 'Z:'] \
        or line=='\n' \
        or line.startswith('%'):
            continue
        else:
            if '%' in line:
                line = line.split('%')
                bar = line[-1]
                line = ''.join(line[:-1])
            score += line + '\n'
    score = score.strip()
    if score.endswith(" |"):
        score += "]"     
    return score

def convert_abc(file_list):
    for file_idx in trange(len(file_list)):
        file = file_list[file_idx]
        filename = file.split('\\')[-1]
        try:
            # output = os.popen(cmd+file).read()
            p = subprocess.Popen(cmd+file, stdout=subprocess.PIPE)
            result = p.communicate()
            output = result[0].decode('utf-8')
            # output = run_filter(output)

            if output=='':
                continue
            else:
                with open('01_xml\\MAD\\'+filename[:-4]+'.xml', 'w', encoding='utf-8') as f:
                    f.write(output)
        except Exception as e:
            print(e)
            pass

if __name__ == '__main__':
    
    file_list = []
    abc_list = []
    
    # traverse folder
    # for root, dirs, files in os.walk('abcs'):
    #     for file in files:
    #         abc_list.append("mxls\\"+file.replace('.txt', '.mxl'))

    # traverse folder
    for root, dirs, files in os.walk('00_raw/MAD'):
        for file in files:
            filename = os.path.join(root, file)
            file_list.append(filename)

    # file_list = list(set(file_list).difference(set(abc_list)))

    # convert_abc(file_list)
    file_lists = []
    for i in range(os.cpu_count()):
        start_idx = int(math.floor(i*len(file_list)/os.cpu_count()))
        end_idx = int(math.floor((i+1)*len(file_list)/os.cpu_count()))
        file_lists.append(file_list[start_idx:end_idx])
    
    # convert_abc(file_list)
    # cnt = 0

    # for file_list in file_lists:
    #     cnt += len(file_list)
        
    pool = Pool(processes=os.cpu_count())
    pool.map(convert_abc, file_lists)