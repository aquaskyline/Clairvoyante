import os
import subprocess


pkg_dir = os.path.dirname(os.path.abspath(__file__))+os.sep+'clairvoyante'
os.chdir(pkg_dir)
py2_scripts = [p for p in os.listdir() if p.endswith('.py')]

print('Call 2to3 on {}'.format(', '.join(py2_scripts)))
subprocess.call(['2to3', '-nw', '--no-diffs']+py2_scripts)

for fn in os.listdir():
    if fn.endswith('.py') and not fn.startswith('__'):
        with open(fn, 'r') as f:
            s = f.read()
        s = s.replace('from . ', '')
        with open(fn, 'w') as f:
            f.write(s)
        print('Restore absolute imports of %s' % fn)
