#coding=utf-8
from xmlrpclib import ServerProxy, Fault
from os.path import join, abspath, isfile
from SimpleXMLRPCServer import SimpleXMLRPCServer
from urlparse import urlparse
import sys

SimpleXMLRPCServer.allow_reuse_address = 1
MAX_HISTORY_LENGTH = 6

UNHANDLED = 100
ACCESS_DENIED = 200

class UnhandledQuery(Fault):
    '''无法处理的查询异常'''
    def __init__(self, msg = 'Could not handle the query'):
        Fault.__init__(self, UNHANDLED, msg)

class AccessDenied(Fault):
    '''为授权资源'''
    def __init__(self, msg = 'Access denied'):
        Fault.__init__(self, ACCESS_DENIED, msg)


def inside(dir, file):
    '''检查目录中是否有该文件'''
    dir = abspath(dir)
    file = abspath(file)
    return file.startswith(join(dir, ''))

def getPort(url):
    name = urlparse(url)[1]
    parts = name.split(':')
    return int(parts[-1])


class Node:
    def __init__(self, url, dir, pwd):
        self.url = url
        self.dir = dir
        self.pwd = dir
        self.known = set()

    def query(self, query, history=[]):
        try:
            return self._handle(query)
        except UnhandledQuery:
            history += self.url
            if len(history) >= MAX_HISTORY_LENGTH: raise
            return self._broadcast(query, history)

    def hello(self, other):
        self.known.add(other)
        return 0

    def fetch(self, query, pwd):
        if pwd != self.pwd: raise AccessDenied
        result = self.query(query)
        f = open(join(self.dir, query), 'w')
        f.write(result)
        f.close()
        return 0

    def _start(self):
        s = SimpleXMLRPCServer(('', getPort(self.url)), logRequests=False)
        s.register_instance(self)
        s.serve_forever()

    def _handle(self, query):
        dir = self.dir
        file = join(dir, query)
        if not isfile(file): raise UnhandledQuery
        if not inside(dir, file): raise AccessDenied
        return open(file).read()

    def _broadcast(self, query, history):
        for other in self.known.copy():
            if other in history:continue
            try:
                s = ServerProxy(other)
                return other.query(query, history)
            except Fault, f:
                if f.faultCode == UNHANDLED: pass
                else: self.known.remove(other)
            except:
                self.known.remove(other)
        raise UnhandledQuery

if __name__ == '__main__':
    url, dir, pwd = sys.argv[1:]
    n = Node(url, dir, pwd)
    n._start()