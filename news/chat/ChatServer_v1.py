#coding=utf-8
from asyncore import dispatcher
from asynchat import async_chat
import socket, asyncore

PORT = 5005
NAME = 'TestChat'
class EndSession(Exception): pass

class CommandHandler:

    def unknown(self, session, cmd):
        session.push('Unknown command : %s' % cmd)

    def handle(self, session, line):
        '''处理接受到的line'''
        if not line.strip():
            return
        parts = line.split(' ', 1)
        cmd = parts[0]
        try:
            line = parts[1].strip()
        except IndexError:
            line = ''
        meth = getattr(self, 'do_' + cmd, None)
        try:
            meth(session, line)
        except TypeError:    # 若不可以调用
            self.unknown(session, cmd)

class Room(CommandHandler):

    '''负责基本命令处理以及广播'''

    def __init__(self, server):
        self.server = server
        self.sessions = []

    def add(self, session):
        '''进入房间'''
        self.sessions.append(session)

    def remove(self, session):
        '''离开房间'''
        self.sessions.remove(session)

    def broadcast(self, line):
        for session in self.sessions:
            session.push(line)

    def do_logout(self, session, line):
        raise EndSession

class LoginRoom(Room):

    '''for clients who just log in'''

    def add(self, session):
        Room.add(self, session)
        self.broadcast('Welcome to %s \r\n' % self.server.name)

    def unknown(self, session, cmd):
        session.push('Please log in\n Use "login <nick>"\r\n')

    def do_login(self, session, line):
        name = line.strip()
        if not name:
            session.push('Please enter a name\r\n')
        elif name in self.server.users:
            session.push('The name %s is taken.\r\n' % name)
            session.push('Please input another name\r\n')
        else:
            session.name = name
            session.enter(self.server.main_room)

class ChatRoom(Room):

    def add(self, session):
        self.broadcast(session.name + ' has entered the room.\r\n')
        self.server.users[session.name] = session
        Room.add(self, session)    # append to room's sessions

    def remove(self, session):
        Room.remove(self, session)
        self.broadcast(session.name + ' has leaved the room\r\n')

    def do_say(self, session, line):
        self.broadcast(session.name + ': ' + line + '\r\n')

    def do_look(self, session, line):
        session.push('The following are in the room:\r\n')
        for session in self.sessions:
            session.push(session.name + '\t')
        session.push('\r\n')

    def do_who(self, session, line):
        session.push('The following are logged in :\r\n')
        for name in self.server.users:
            session.push(name + '\t')
        session.push('\r\n')

class LogoutRoom(Room):
    def add(self, session):
        try:
            del self.server.users[session.name]
        except KeyError: pass

class ChatSession(async_chat):

    def __init__(self, server, sock):
        async_chat.__init__(self, socket)
        self.server = server
        self.set_terminator('\r\n')
        self.data = []
        self.name = None
        self.enter(LoginRoom(server))

    def enter(self, room):
        try:
            cur = self.room
        except AttributeError:
            pass
        else:
            cur.remove(self)
        self.room = room
        room.add(self)

    def collect_incoming_data(self, data):
        self.data.append(data)

    def found_terminator(self):
        line = ''.join(self.data)
        self.data = []
        try:
            self.room.handle(self, line)
        except EndSession:
            self.handle_close()

    def handle_close(self):
        async_chat.handle_close(self)
        self.enter(LogoutRoom(self.server))


class ChatServer(dispatcher):

    def __init__(self, port, name):
        dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind(('', port))
        self.listen(5)
        self.name = name
        self.users = {}
        self.main_room = ChatRoom(self)

    def handle_accept(self):
        conn, addr = self.accept()
        ChatSession(self, conn)

if __name__ == '__main__':
    s = ChatServer(PORT, NAME)
    try: asyncore.loop()
    except KeyboardInterrupt: print