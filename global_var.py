def _init():
     global _global_cache
     _global_cache = {}

def set_value(name, value):
     _global_cache[name] = value

def get_value(name, defValue=None):
     try:
        return _global_cache[name]
     except KeyError:
        return defValue