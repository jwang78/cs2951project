import time
from subprocess import Popen, PIPE
with open("hosts.txt", "r") as fi:
  machines = fi.read().split()

proc_count = "$(($(grep -c ^processor /proc/cpuinfo) / 2))"
cmds = [["yes", "|", "/bin/ssh", "jwang78@%s 'cd /home/jwang78/course/cs2951/cs2951project; source /nbu/cuda10/teef-2.0/bin/activate; nohup python worker.py 35.188.1.232 %s'" % (machine, proc_count)] for machine in machines]
#cmds = [["/bin/ssh", "jwang78@%s 'killall -9 python'" % machine] for machine in machines]
print(" ".join(cmds[0]))
lines = [" ".join(cmd) + " &" for cmd in cmds]
with open("cmds.sh", "w") as fi:
  fi.write("\n".join(lines))
