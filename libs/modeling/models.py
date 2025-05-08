import os

# backbone (e.g., conv / transformer)如果找不到，去init的地方看看是不是没导入py文件
backbones = {}
def register_backbone(name):
    def decorator(cls):
        backbones[name] = cls
        return cls
    return decorator

# neck (e.g., FPN)
necks = {}
def register_neck(name):
    def decorator(cls):
        necks[name] = cls
        return cls
    return decorator

# location generator (point, segment, etc)
generators = {}
def register_generator(name):
    def decorator(cls):
        generators[name] = cls
        return cls
    return decorator

# meta arch (the actual implementation of each model)
meta_archs = {}
def register_meta_arch(name):
    def decorator(cls):
        meta_archs[name] = cls
        return cls
    return decorator
first_stage_archs = {}
def register_first_stage_arch(name):
    def decorator(cls):
        first_stage_archs[name] = cls
        return cls
    return decorator
layers={}
def register_layer(name):
    def decorator(cls):
        layers[name]=cls
        return cls
    return decorator
blocks={}
def register_block(name):
    def decorator(cls):
        blocks[name]=cls
        return cls
    return decorator
losses={}
def register_loss(name):
    def decorator(cls):
        losses[name]=cls
        return cls
    return decorator
# builder functions
def make_backbone(name, **kwargs):
    backbone = backbones[name](**kwargs)
    return backbone

def make_neck(name, **kwargs):
    neck = necks[name](**kwargs)
    return neck

def make_meta_arch(name, **kwargs):
    meta_arch = meta_archs[name](**kwargs)
    return meta_arch
def make_layer(name,**kwargs):
    layer=layers[name](**kwargs)
    return layer
def make_block(name,**kwargs):
    block=blocks[name](**kwargs)
    return block
def make_loss(name,**kwargs):
    loss=losses[name](**kwargs)
    return loss
def make_generator(name, **kwargs):
    generator = generators[name](**kwargs)
    return generator
def make_first_stage_arch(name, **kwargs):
    first_stage_arch = first_stage_archs[name](**kwargs)
    return first_stage_arch