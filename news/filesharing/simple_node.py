#coding=utf-8
from xmlrpclib import ServerProxy
from os.path import join, isfile
from SimpleXMLRPCServer import SimpleXMLRPCServer
from urlparse import urlparse
import sys

MAX_HISTORY_LENGTH = 6

OK = 1
FAIL = 2
EMPTY = ''

def getPort(url):
    name = urlparse(url=url)[1]
    parts = name.split(':')
    return int(parts[-1])

class Node:

    def __init__(self, url, dirname, pwd):
        self.url = url
        self.dirname = dirname
        self.pwd = pwd
        self.known = set()

    def query(self, query, history=[]):
        '''查询文件，可能向相邻node寻求帮助，将文件作为字符串返回'''
        code, data = self._handle(query)
        if code == OK:
            return code, data
        else:
            history.append(self.url)
            if len(history) >= MAX_HISTORY_LENGTH:
                return FAIL, EMPTY
            return self._broadcast(query, history)

    def hello(self, other):
        '''介绍给其他节点'''
        self.known.add(other)
        return OK

    def fetch(self, query, pwd):
        if pwd != self.pwd:
            return FAIL
        code, data = self.query(query)
        if code == OK:
            f = open(join(self.dirname, query), 'w')
            f.write(data)
            f.close()
            return OK
        else:
            return FAIL

    def _start(self):
        '''启动XML-RPC服务器'''
        s = SimpleXMLRPCServer(('', getPort(self.url)), logRequests=False)
        s.register_instance(self)
        s.serve_forever()

    def _handle(self, query):
        dir = self.dirname
        filename = join(dir, query)
        if isfile(filename):
            return OK, open(filename).read()
        else:
            return FAIL, EMPTY

    def _broadcast(self, query, history):
        for other in self.known.copy():                        # copy???
            if other in history:
                continue
            try:
                s = ServerProxy(other)                         # 获取远程的instance
                code, data = s.query(query, history)           # 调用该instance的函数
                if code == OK:
                    return code, data
            except:
                self.known.remove(other)                        # 连接失败
        return FAIL, EMPTY

def main():
    url, dir, pwd = sys.argv[1:]
    n = Node(url, dir, pwd)
    n._start()                                                  # 启动server，等待命令服务

if __name__ == '__main__':
    main()