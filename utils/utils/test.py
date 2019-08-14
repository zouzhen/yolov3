# import os

# print('Process (%s) start...' % os.getpid())
# # Only works on Unix/Linux/Mac:
# pid = os.fork()
# if pid == 0:
#     print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
# else:
#     print('I (%s) just created a child process (%s).' % (os.getpid(), pid))


import asyncio
import threading
@asyncio.coroutine
def hello():
    print('Hello world! (%s)' % threading.currentThread())
    yield from asyncio.sleep(10)
    print('Hello again! (%s)' % threading.currentThread())

@asyncio.coroutine
def hello1():
    print('Hello world! (%s)' % threading.currentThread())
    yield from asyncio.sleep(10)
    print('Hello again! (%s)' % threading.currentThread())

@asyncio.coroutine
def hello2():
    print('Hello world! (%s)' % threading.currentThread())
    yield from asyncio.sleep(10)
    print('Hello again! (%s)' % threading.currentThread())

# 获取EventLoop:
loop = asyncio.get_event_loop()
# 执行coroutine
tasks = [hello(), hello1(), hello2()]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
