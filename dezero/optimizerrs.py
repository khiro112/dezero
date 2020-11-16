class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []
        
    def setup(self, target):
        self.target = target
        return self
    
    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        
        for f in self.hooks:
            f(params)
            
        for param in params()