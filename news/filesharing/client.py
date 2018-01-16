#coding=utf-8
from xmlrpclib import ServerProxy, Fault
from cmd import Cmd
from random import choice
from string import lowercase
from server import Node, UNHANDLED
from threading import Thread
from time import sleep
import sys

HEAD_START = 0.1
PWD_LEN = 100

def randomString(length):
    char = []
    letters = lowercase[:26]
    while length > 0:
        length -= 1
        char.append(choice(letters))
    return ''.join(char)

class Client(Cmd):
    prompt = '>'

    def __init__(self, url, dir, urlfile):
        Cmd.__init__(self)
        self.pwd = randomString(PWD_LEN)
        n = Node(url, dir, self.pwd)
        t = Thread(target=n._start)              # 启动线程
        t.setDaemon(1)
        t.start()
        # 启动服务器
        sleep(HEAD_START)
        self.server = ServerProxy(url)
        for line in open(urlfile):              # 每行包括一个已知node的url
            line = line.strip()
            self.server.hello(line)

    def do_fetch(self, query):
        try:
            self.server.fetch(query, self.pwd)
        except Fault, f:
            if f.faultCode != UNHANDLED: raise
            print('Could not find the file', query)

    def do_exit(self, arg):
        print
        sys.exit()

    do_EOF = do_exit

if __name__ == '__main__':
    urlfile, dir, url = sys.argv[1:]
    client = Client(url, dir, urlfile)
    client.cmdloop()