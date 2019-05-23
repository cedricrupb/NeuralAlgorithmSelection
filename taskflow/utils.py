import subprocess as sp
import sys
from time import time


def run_command(command, **kwargs):

    kwargs['stdout'] = sp.PIPE
    kwargs['stderr'] = sp.PIPE
    kwargs['universal_newlines'] = True
    kwargs['bufsize'] = 1

    timeout = -1
    set_timeout = False

    if 'timeout' in kwargs:
        timeout = kwargs['timeout']
        set_timeout = True
        del kwargs['timeout']

    start = time()
    timer = timeout

    with sp.Popen(args=command, **kwargs) as process:

        while not set_timeout or timer > 0:

            try:
                line = process.stdout.readline()

                if line == '' and process.poll() is not None:
                    break

                err = process.stderr.readline()

                sys.stdout.write(line)
                sys.stderr.write(err)

                timer = timeout - (time() - start)
            except Exception:
                process.kill()
                raise

        if set_timeout and timer <= 0:
            process.kill()
            process.wait()
            raise sp.TimeoutExpired("Timeout of %i seconds is expired" % timeout)

        if process.poll():
            raise sp.CalledProcessError(
                process.returncode, command,
            )
