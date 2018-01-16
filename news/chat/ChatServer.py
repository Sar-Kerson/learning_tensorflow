# coding=utf-8
from asyncore import dispatcher
from asynchat import async_chat
import socket, asyncore

PORT = 5006
NAME = 'TestChat'

class ChatSession(async_chat):

    def __init__(self, server, socket):
        async_chat.__init__(self, socket)
        self.server = server
        self.set_terminator("\r\n")
        self.data = []
        # 向客户端写入数据
        self.push('Welcome to %s \r\n' % self.server.name)

    def collect_incoming_data(self, data):
        self.data.append(data)

    def found_terminator(self):
        '''每次发现终结符都进行广播'''
        line = ''.join(self.data)
        self.data = []
        self.server.broadcast(line)

    def handle_close(self):
        async_chat.handle_close(self)
        self.server.disconnect(self)

class ChatServer(dispatcher):

    def __init__(self, port, name):
        dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()            # 当服务器崩溃之后，不占用端口
        self.bind(('', port))
        self.listen(5)
        self.name = name
        self.sessions = []               # 连接的会话

    def disconnect(self, session):
        self.sessions.remove(session)

    def handle_accept(self):
        conn, addr = self.accept()
        print('Connection attempt from ', addr[0])      # conn 是套接字数据
        self.sessions.append(ChatSession(self, conn))

    def broadcast(self, line):
        for session in self.sessions:
            session.push(line + '\r\n')




if __name__ == '__main__':
    s = ChatServer(PORT, NAME)
    try:asyncore.loop()
    except KeyboardInterrupt: pass