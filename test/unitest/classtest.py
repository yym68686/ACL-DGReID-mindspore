class Base(object):
    def __init__(self):
        print("Base init")

class A(Base):
    def __init__(self):
        print("A init")
        super(A, self).__init__()

class B(Base):
    def __init__(self):
        print("B init")
        super(B, self).__init__()

class C(B, A):
    def __init__(self):
        print("C init")
        super().__init__()

c = C()