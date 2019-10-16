class A:
    def __init__(self):
        self.a = 0
        self.set_a()
    def set_a(self):
        self.a = 1

class B(A):
    def __init__(self):
        super(B, self).__init__()
    def set_a(self):
        self.a = 10

b = B()
print(b.a)


