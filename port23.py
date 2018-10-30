import os
import subprocess

os.chdir('./clairvoyante/')
py_sripts = [p for p in os.listdir() if p.endswith('.py')]

# use python provided 2to3 to convert package scripts to python3 versions
subprocess.call(['2to3', '-nw']+py_sripts)

# fix relative imports, overwrite to package folder
for fn in os.listdir():
    if fn.endswith('.py') and not fn.startswith('__'):
        with open(fn, 'r') as f:
            s = f.read()
        s = s.replace('from . ', '')
        with open(fn, 'w') as f:
            f.write(s)
        print('Fix import of %s' % fn)
